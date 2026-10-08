"""Diagnose EOS, then evaluate the same 18 questions with a concise prompt."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import torch
from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from evaluate_base_v2 import GPUSampler, block, digest, flags, memory

BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BASE_DIR/'configs'/'base_3b_concise.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    base = json.loads((BASE_DIR/config['base_config']).read_text(encoding='utf-8-sig'))
    source = BASE_DIR/base['output_dir']
    out = BASE_DIR/config['output_dir']
    if any((out/name).exists() for name in ['base_3b_answers.json', 'base_3b_answers.md', 'eos_diagnostics.json']):
        raise FileExistsError('Preserve existing results; choose a new output directory.')
    out.mkdir(parents=True, exist_ok=True)
    assert HF_HOME == r'D:\Code\huggingface_cache'
    assert HF_HUB_CACHE == r'D:\Code\huggingface_cache\hub'
    download = json.loads((source/'download_report.json').read_text(encoding='utf-8'))
    assert json.loads((source/'load_report.json').read_text(encoding='utf-8'))['status'] == 'success'
    snapshot = Path(download['snapshot'])
    assert snapshot.is_relative_to(Path(HF_HUB_CACHE))
    baseline_path = BASE_DIR/base['baseline_file']
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    previous = json.loads((source/'base_3b_answers.json').read_text(encoding='utf-8'))
    assert len(baseline['results']) == len(previous['results']) == 18
    protected_files = [baseline_path, source/'base_3b_answers.json', source/'base_3b_answers.md']
    protected_files += [p for p in (BASE_DIR/'output'/'formal_v1').rglob('*') if p.is_file()]
    protected = {str(p): digest(p) for p in protected_files}
    report = {'status':'started', 'started_at':datetime.now().isoformat(timespec='seconds'), 'adapter_loaded':False,
              'model':base['model_name'], 'hf_home':HF_HOME, 'hf_hub_cache':HF_HUB_CACHE, 'snapshot':str(snapshot)}
    start = time.monotonic()
    sampler = GPUSampler()
    try:
        assert torch.cuda.is_available(), 'CUDA unavailable.'
        torch.cuda.reset_peak_memory_stats()
        with sampler:
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            old_tokenizer = AutoTokenizer.from_pretrained(baseline['metadata']['model'], local_files_only=True)
            template_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
            assert template_sha == baseline['metadata']['chat_template_sha256']
            assert tokenizer.chat_template == old_tokenizer.chat_template
            quantization = BitsAndBytesConfig(load_in_4bit=base['load_in_4bit'], bnb_4bit_quant_type=base['bnb_4bit_quant_type'],
                                             bnb_4bit_compute_dtype=getattr(torch, base['bnb_4bit_compute_dtype']))
            model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, quantization_config=quantization,
                                                       device_map=base['device_map'], dtype=torch.bfloat16, attn_implementation='sdpa')
            model.eval()
            assert getattr(model, 'is_loaded_in_4bit', False)
            devices = sorted({str(t.device) for t in list(model.parameters())+list(model.buffers())})
            assert devices == ['cuda:0'], devices
            assert (torch.ones(1, device='cuda')+1).item() == 2
            generation = GenerationConfig.from_dict(baseline['metadata']['generation_config'])
            assert generation.max_new_tokens == 512 and generation.do_sample is False
            stop_ids = generation.eos_token_id if isinstance(generation.eos_token_id, list) else [generation.eos_token_id]
            native = json.loads((snapshot/'generation_config.json').read_text(encoding='utf-8'))
            assert stop_ids == native['eos_token_id'] and tokenizer.eos_token_id in stop_ids
            assert generation.forced_eos_token_id is None

            def generate(content, settings):
                prompt = tokenizer.apply_chat_template([{'role':'user', 'content':content}], tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
                begun = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**encoded, generation_config=deepcopy(settings))
                torch.cuda.synchronize()
                ids = generated.sequences[0, encoded['input_ids'].shape[1]:].tolist()
                answer = tokenizer.decode(ids, skip_special_tokens=True)
                eos_positions = [i for i, token in enumerate(ids) if token in stop_ids]
                return {'base_answer':answer, 'generated_tokens_including_stop':len(ids), 'generated_token_ids':ids,
                        'answer_characters':len(answer), 'non_whitespace_characters':len(re.sub(r'\s', '', answer)),
                        'last_token_id':ids[-1] if ids else None, 'last_token_text':tokenizer.convert_ids_to_tokens(ids[-1]) if ids else None,
                        'eos_positions':eos_positions, 'ended_with_eos':bool(ids and ids[-1] in stop_ids),
                        'stop_reason':'eos' if ids and ids[-1] in stop_ids else 'max_new_tokens' if len(ids)>=settings.max_new_tokens else 'other',
                        'seconds':round(time.monotonic()-begun, 2), 'flags':flags(answer, ids, stop_ids, settings.max_new_tokens),
                        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(), 'effective_user_content':content}

            diagnostic = {'eos_ids':stop_ids, 'eos_tokens':[tokenizer.convert_ids_to_tokens(i) for i in stop_ids],
                          'tokenizer_eos_id':tokenizer.eos_token_id, 'native_generation_config':native,
                          'evaluation_generation_config':generation.to_dict(), 'chat_template_sha256':template_sha,
                          'forced_eos':False, 'checks':[]}
            simple_config = deepcopy(generation)
            simple_config.max_new_tokens = 32
            simple = generate('请只回复一个字：好', simple_config)
            diagnostic['checks'].append({'name':'short_response_eos_sanity', 'max_new_tokens':32, **simple})
            print('EOS sanity:', simple['generated_tokens_including_stop'], simple['stop_reason'], repr(simple['base_answer']), flush=True)
            extended_config = deepcopy(generation)
            extended_config.max_new_tokens = config['diagnostic_max_new_tokens']
            original = generate(baseline['results'][0]['question'], extended_config)
            original['first_512_tokens_match_previous_answer'] = tokenizer.decode(original['generated_token_ids'][:512], skip_special_tokens=True) == previous['results'][0]['base_answer']
            diagnostic['checks'].append({'name':'original_question_1_extended_limit', 'max_new_tokens':extended_config.max_new_tokens, **original})
            (out/'eos_diagnostics.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
            print('Original Q1 extended:', original['generated_tokens_including_stop'], original['stop_reason'], 'prefix matches:', original['first_512_tokens_match_previous_answer'], flush=True)

            rows = []
            payload = {'metadata':{'model':base['model_name'], 'revision':download['revision'], 'adapter_loaded':False,
                                  'prompt_suffix':config['prompt_suffix'], 'prompt_changed_from_previous':True,
                                  'generation_config':generation.to_dict(), 'chat_template_sha256':template_sha,
                                  'baseline_file':str(baseline_path), 'previous_3b_file':str(source/'base_3b_answers.json'),
                                  'automatic_scoring':False, 'character_count_definition':'Unicode characters excluding whitespace; punctuation and code included'}, 'results':rows}
            answer_path = out/'base_3b_answers.json'
            with answer_path.open('x', encoding='utf-8') as f: json.dump(payload, f, ensure_ascii=False, indent=2)
            inference_start = time.monotonic()
            for old in baseline['results']:
                original_prompt = tokenizer.apply_chat_template([{'role':'user','content':old['question']}], tokenize=False, add_generation_prompt=True)
                assert hashlib.sha256(original_prompt.encode()).hexdigest() == old['prompt_sha256']
                content = old['question']+'\n\n'+config['prompt_suffix']
                prompt = tokenizer.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
                assert tokenizer.encode(prompt, add_special_tokens=False) == old_tokenizer.encode(prompt, add_special_tokens=False)
                row = {'id':old['id'], 'topic':old['topic'], 'question':old['question'], **generate(content, generation)}
                rows.append(row)
                answer_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'[{len(rows)}/18] tokens={row["generated_tokens_including_stop"]}, chars={row["non_whitespace_characters"]}, stop={row["stop_reason"]}', flush=True)
            summary = {'successful_generations':len(rows), 'natural_eos_count':sum(r['ended_with_eos'] for r in rows),
                       'max_new_tokens_stop_count':sum(r['stop_reason']=='max_new_tokens' for r in rows),
                       'mean_generated_tokens_including_stop':round(statistics.mean(r['generated_tokens_including_stop'] for r in rows), 2),
                       'mean_answer_characters':round(statistics.mean(r['answer_characters'] for r in rows), 2),
                       'mean_non_whitespace_characters':round(statistics.mean(r['non_whitespace_characters'] for r in rows), 2),
                       'within_200_350_characters_count':sum(200<=r['non_whitespace_characters']<=350 for r in rows),
                       'empty_answers':sum(not r['base_answer'].strip() for r in rows),
                       'flag_counts':dict(Counter(flag for r in rows for flag in r['flags'])),
                       'total_inference_seconds':round(time.monotonic()-inference_start, 2)}
            payload['summary'] = summary
            answer_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            lines = ['# Qwen2.5-3B Base：简洁提示评估', '', '同一组 18 道题；每道 user 内容末尾追加：'+config['prompt_suffix'], '',
                     '模板及全部生成参数沿用原评估：do_sample=False、max_new_tokens=512、NF4/bfloat16。提示内容已改变，本轮用于观察简洁要求的效果。答案原样保留，未强制截字或强制 EOS，未自动评分。', '',
                     '字数统计为去除空白后的 Unicode 字符数，标点和代码计入。', '', '## 统计', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '']
            for row in rows:
                lines += [f'## {row["id"]}. {row["topic"]}', '', '**原问题**', '', row['question'], '', '**3B Base 回答**', '', block(row['base_answer']), '',
                          f'tokens：{row["generated_tokens_including_stop"]}；去空白字符数：{row["non_whitespace_characters"]}；停止原因：{row["stop_reason"]}；末尾 token：{row["last_token_id"]}；标记：{row["flags"]}。', '']
            with (out/'base_3b_answers.md').open('x', encoding='utf-8') as f: f.write('\n'.join(lines))
            report.update({'summary':summary, 'cuda_available':True, 'loaded_in_4bit':True, 'actual_tensor_devices':devices,
                           'same_18_original_questions_and_template_verified':True, 'generation_config_unchanged':True, 'eos_diagnostics_file':str(out/'eos_diagnostics.json')})
        report['status'] = 'success'
    except Exception as error:
        report.update({'status':'failed', 'error_type':type(error).__name__, 'error':str(error), 'oom':'out of memory' in str(error).lower()})
        raise
    finally:
        report.update({'previous_results_and_formal_v1_unchanged':all(p.exists() and digest(p)==protected[str(p)] for p in protected_files),
                       'elapsed_seconds_including_load_and_diagnostics':round(time.monotonic()-start, 2),
                       'torch_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2, 2) if torch.cuda.is_available() else None,
                       'torch_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/1024**2, 2) if torch.cuda.is_available() else None,
                       'whole_gpu_peak_sampled_mib':sampler.peak})
        (out/'evaluation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT:', json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
