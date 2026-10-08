"""Independent 3B base load/evaluation. Never loads an adapter or trains."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

BASE_DIR = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    result = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=False)
    return [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')] if result.returncode == 0 else None


class GPUSampler:
    def __init__(self):
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample,daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            current = memory()
            if current: self.peak = max(self.peak,current[0])
            self.stop_event.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop_event.set()
        self.thread.join(timeout=5)


def block(answer):
    fence = chr(96) * max(4,max((len(x) for x in re.findall(chr(96)+'+',answer)),default=0)+1)
    return fence+'text\n'+answer+'\n'+fence


def flags(answer, ids, stop_ids, limit):
    found = []
    if not answer.strip(): found.append('empty')
    if '\ufffd' in answer or '\x00' in answer: found.append('invalid_character')
    if len(ids)>=limit and (not ids or ids[-1] not in stop_ids): found.append('token_limit_reached_possible_truncation')
    lines = [x.strip() for x in answer.splitlines() if x.strip()]
    # Same line-repeat heuristic as v1; code lines may be false positives.
    if len(lines)>=6 and Counter(lines).most_common(1)[0][1]>=4: found.append('repeated_lines_manual_review')
    if len(answer.strip())<10: found.append('very_short_manual_review')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE_DIR/'configs'/'base_3b.json')
    parser.add_argument('--mode',choices=['download','load','evaluate'],default='load')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    out = BASE_DIR/config['output_dir']
    out.mkdir(parents=True,exist_ok=True)
    baseline_path = BASE_DIR/config['baseline_file']
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if args.mode=='download':
        info = HfApi().model_info(config['model_name'],files_metadata=True)
        weight_bytes = sum(file.size or 0 for file in info.siblings if file.rfilename.endswith('.safetensors'))
        print('Model:',config['model_name'],'revision:',info.sha,'weight bytes:',weight_bytes,flush=True)
        cached = snapshot_download(config['model_name'],revision=info.sha,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'],max_workers=2)
        report = {'model':config['model_name'],'revision':info.sha,'snapshot':cached,'weight_bytes':weight_bytes}
        (out/'download_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return
    download = json.loads((out/'download_report.json').read_text(encoding='utf-8'))
    snapshot = Path(download['snapshot'])
    if args.mode=='evaluate':
        load_test = json.loads((out/'load_report.json').read_text(encoding='utf-8'))
        if load_test.get('status')!='success': raise RuntimeError('Load-only test must succeed first.')
        if any((out/name).exists() for name in ['base_3b_answers.json','base_3b_answers.md']):
            raise FileExistsError('Preserve existing evaluation results; do not overwrite them.')
    protected_root = BASE_DIR/'output'/'formal_v1'
    protected = {str(p.relative_to(protected_root)):digest(p) for p in protected_root.rglob('*') if p.is_file()}
    old_comparison_sha = digest(baseline_path)
    report = {'mode':args.mode,'model':config['model_name'],'revision':download['revision'],'status':'started','started_at':datetime.now().isoformat(timespec='seconds'),'cuda_available':torch.cuda.is_available(),'memory_before_mib':memory(),'adapter_loaded':False}
    report_path = out/('load_report.json' if args.mode=='load' else 'evaluation_report.json')
    start = time.monotonic()
    sampler = GPUSampler()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; stop without changing model.')
        torch.cuda.reset_peak_memory_stats()
        with sampler:
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
            report['tokenizer_loaded'] = True
            old_tokenizer = AutoTokenizer.from_pretrained(baseline['metadata']['model'],local_files_only=True)
            if hashlib.sha256(old_tokenizer.chat_template.encode()).hexdigest()!=baseline['metadata']['chat_template_sha256']:
                raise ValueError('Cached v1 chat template differs from original comparison.')
            native_template_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
            tokenizer.chat_template = old_tokenizer.chat_template
            quantization = BitsAndBytesConfig(load_in_4bit=config['load_in_4bit'],bnb_4bit_quant_type=config['bnb_4bit_quant_type'],bnb_4bit_compute_dtype=getattr(torch,config['bnb_4bit_compute_dtype']))
            model = AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,quantization_config=quantization,device_map=config['device_map'],dtype=torch.bfloat16,attn_implementation='sdpa')
            model.eval()
            torch.cuda.synchronize()
            device_map = getattr(model, 'hf_device_map', None)
            if device_map is None:
                devices = sorted({str(t.device) for t in list(model.parameters()) + list(model.buffers())})
                device_map = {f'actual_tensor_device_{i}': device for i, device in enumerate(devices)}
            report.update({'model_loaded':True,'loaded_in_4bit':bool(getattr(model,'is_loaded_in_4bit',False)),'gpu':torch.cuda.get_device_name(0),'bf16_supported':torch.cuda.is_bf16_supported(),'hf_device_map':device_map,'model_memory_footprint_mib':round(model.get_memory_footprint()/1024**2,2),'memory_after_load_mib':memory(),'native_template_sha256':native_template_sha,'effective_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()})
            if not report['loaded_in_4bit']: raise RuntimeError('Model was not loaded in 4bit as requested.')
            if any(str(device) in ['cpu','disk'] for device in device_map.values()):
                raise RuntimeError('Auto placement offloaded to CPU/disk; stop and report instead of silently running a different hardware test.')
            # CUDA arithmetic only; the load-only phase performs no generation.
            assert (torch.ones(1,device='cuda')+1).item()==2
            print('LOAD TEST:',json.dumps(report,ensure_ascii=False),flush=True)
            if args.mode=='evaluate':
                generation = GenerationConfig.from_dict(baseline['metadata']['generation_config'])
                if generation.do_sample is not False or generation.max_new_tokens!=512:
                    raise ValueError('v1 generation settings are unexpected.')
                inputs_list = []
                for row in baseline['results']:
                    prompt = tokenizer.apply_chat_template([{'role':'user','content':row['question']}],tokenize=False,add_generation_prompt=True)
                    if hashlib.sha256(prompt.encode()).hexdigest()!=row['prompt_sha256']:
                        raise ValueError('Prompt differs from v1 for question '+str(row['id']))
                    encoded = tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
                    old_encoded = old_tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
                    if not torch.equal(encoded['input_ids'],old_encoded['input_ids']):
                        raise ValueError('Encoded prompt token IDs differ from v1.')
                    inputs_list.append(encoded)
                report['all_18_prompt_text_and_token_ids_match_v1'] = True
                rows = []
                payload = {'metadata':{'model':config['model_name'],'revision':download['revision'],'adapter_loaded':False,'baseline_comparison_file':str(baseline_path),'baseline_sha256':old_comparison_sha,'generation_config':generation.to_dict(),'native_template_sha256':native_template_sha,'effective_template_sha256':baseline['metadata']['chat_template_sha256'],'all_prompts_and_token_ids_match_v1':True,'quantization':config,'automatic_scoring':False},'results':rows}
                answer_path = out/'base_3b_answers.json'
                with answer_path.open('x',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False,indent=2)
                inference_start = time.monotonic()
                stop_ids = generation.eos_token_id if isinstance(generation.eos_token_id,list) else [generation.eos_token_id]
                for original, encoded in zip(baseline['results'],inputs_list):
                    inputs = encoded.to(model.device)
                    begun = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(**inputs,generation_config=deepcopy(generation))
                    torch.cuda.synchronize()
                    ids = generated.sequences[0,inputs['input_ids'].shape[1]:].tolist()
                    answer = tokenizer.decode(ids,skip_special_tokens=True)
                    row = {'id':original['id'],'topic':original['topic'],'question':original['question'],'base_answer':answer,'generated_tokens_including_stop':len(ids),'answer_characters':len(answer),'seconds':round(time.monotonic()-begun,2),'flags':flags(answer,ids,stop_ids,512),'prompt_sha256':original['prompt_sha256']}
                    rows.append(row)
                    answer_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
                    print(f'[{len(rows)}/18] {row["topic"]}: tokens={len(ids)}, chars={len(answer)}, flags={row["flags"]}',flush=True)
                summary = {'successful_generations':len(rows),'total_questions':len(baseline['results']),'mean_generated_tokens_including_stop':round(statistics.mean(r['generated_tokens_including_stop'] for r in rows),2),'mean_answer_characters':round(statistics.mean(r['answer_characters'] for r in rows),2),'answers_reaching_512_tokens':sum(r['generated_tokens_including_stop']>=512 for r in rows),'empty_answers':sum(not r['base_answer'].strip() for r in rows),'flag_counts':dict(Counter(flag for r in rows for flag in r['flags'])),'total_inference_seconds':round(time.monotonic()-inference_start,2)}
                payload['summary'] = summary
                answer_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
                lines = ['# Qwen2.5-3B Base 原始答案','','复用 v1 的 18 道问题，逐题核对 prompt 文本和 token IDs 完全相同；同一 chat template 和完整生成参数，do_sample=False、max_new_tokens=512。4bit NF4、bfloat16 compute、device_map=auto；未加载 LoRA、未训练、未自动评分。','','## 统计','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','']
                for row in rows:
                    lines.extend([f'## {row["id"]}. {row["topic"]}','','**问题**','',row['question'],'','**3B Base 原始回答**','',block(row['base_answer']),'',f'tokens（含停止 token）：{row["generated_tokens_including_stop"]}；字符数：{row["answer_characters"]}；标记：{row["flags"]}。',''])
                with (out/'base_3b_answers.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
                report['summary'] = summary
            report['status'] = 'success'
    except Exception as error:
        report.update({'status':'failed','error_type':type(error).__name__,'error':str(error),'oom':'out of memory' in str(error).lower(),'memory_at_error_mib':memory()})
        raise
    finally:
        current = {str(p.relative_to(protected_root)):digest(p) for p in protected_root.rglob('*') if p.is_file()}
        report.update({'formal_v1_all_files_unchanged':current==protected,'v1_comparison_unchanged':digest(baseline_path)==old_comparison_sha,'elapsed_seconds_including_load':round(time.monotonic()-start,2),'torch_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,2) if torch.cuda.is_available() else None,'torch_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/1024**2,2) if torch.cuda.is_available() else None,'whole_gpu_peak_sampled_mib':sampler.peak,'memory_after_mib':memory()})
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('REPORT:',json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
