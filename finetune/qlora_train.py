import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data' / 'train.jsonl'
VAL_FILE = BASE_DIR / 'data' / 'val.jsonl'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'


def length(tokenizer, row):
    ids = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
    return len(ids['input_ids'] if hasattr(ids, 'keys') else ids)


class MemoryMonitor(TrainerCallback):
    def __init__(self, run_dir, smoke):
        self.gpu_used_mib = []
        self.run_dir = run_dir
        self.smoke = smoke

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {'global_step':state.global_step,'epoch':state.epoch, **(logs or {})}
        with (self.run_dir / 'trainer_logs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print('TRAINER_LOG:', json.dumps(record, ensure_ascii=False), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            self.gpu_used_mib.append(int(result.stdout.strip().splitlines()[0]))
        if not self.smoke and state.global_step == state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='64 train + 16 validation examples, one epoch; isolated output.')
    parser.add_argument('--resume-from-checkpoint', type=Path, help='Resume formal_v1 from its saved checkpoint; do not restart an existing run.')
    options = parser.parse_args()
    smoke = options.smoke_test
    run_dir = OUTPUT_DIR / ('smoke_1024_' + datetime.now().strftime('%Y%m%d_%H%M%S')) if smoke else OUTPUT_DIR / 'formal_v1'
    if smoke and options.resume_from_checkpoint:
        parser.error('Checkpoint resume is for the formal run only.')
    if options.resume_from_checkpoint:
        checkpoint = options.resume_from_checkpoint.resolve()
        if checkpoint.parent != run_dir.resolve() or not (checkpoint / 'trainer_state.json').exists():
            parser.error('Choose an existing checkpoint inside output/formal_v1.')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'{run_dir} already contains a run; use --resume-from-checkpoint to resume.')
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {'mode':'smoke' if smoke else 'formal_v1', 'max_length':1024, 'num_train_epochs':1, 'status':'started', 'model':MODEL_NAME,
              'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA_FILE,VAL_FILE)},
              'resume_from_checkpoint':str(options.resume_from_checkpoint) if options.resume_from_checkpoint else None}
    monitor = MemoryMonitor(run_dir, smoke)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; a CPU test cannot validate RTX 4060 OOM behavior.')
        dataset = load_dataset('json', data_files={'train':str(DATA_FILE),'validation':str(VAL_FILE)})
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        train, val = dataset['train'], dataset['validation']
        if smoke:
            lengths = [length(tokenizer,row) for row in train]
            longest = sorted(range(len(train)),key=lambda i:lengths[i],reverse=True)[:16]
            remaining = [i for i in range(len(train)) if i not in longest]
            indices = longest + random.Random(42).sample(remaining,48)
            train = train.select(indices)
            val_lengths = [length(tokenizer,row) for row in val]
            val_indices = sorted(range(len(val)),key=lambda i:val_lengths[i],reverse=True)[:16]
            val = val.select(val_indices)
            report.update({'train_indices':indices,'validation_indices':val_indices,'untruncated_selected_lengths':[lengths[i] for i in indices], 'padding':'pad_to_multiple_of=1024 ensures actual 1024-token training tensors'})
        print('训练数据数量:',len(train),'验证数据数量:',len(val),'smoke:',smoke,flush=True)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)
        lora_config = LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM')
        training_args = SFTConfig(
            output_dir=str(run_dir), num_train_epochs=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=2e-4,
            logging_steps=1 if smoke else 10, max_length=1024,
            eval_strategy='epoch' if smoke else 'steps', eval_steps=250,
            prediction_loss_only=True,
            bf16=True, gradient_checkpointing=True,
            packing=False, pad_to_multiple_of=1024 if smoke else None,
            report_to='none', save_strategy='no' if smoke else 'steps',
            save_steps=250, save_total_limit=3, save_only_model=False,
            model_init_kwargs={'local_files_only':True,'device_map':{'':0},'attn_implementation':'sdpa','dtype':torch.bfloat16},
        )
        torch.cuda.reset_peak_memory_stats()
        trainer = SFTTrainer(model=MODEL_NAME,train_dataset=train,eval_dataset=val,args=training_args,processing_class=tokenizer,peft_config=lora_config,quantization_config=quantization_config,callbacks=[monitor])
        (run_dir / 'training_args.json').write_text(trainer.args.to_json_string(), encoding='utf-8')
        batch = next(iter(trainer.get_train_dataloader()))
        actual_length = batch['input_ids'].shape[-1]
        if smoke and actual_length != 1024:
            raise RuntimeError(f'Smoke batch does not exercise 1024 tokens: {actual_length}')
        del batch
        trainer.model.print_trainable_parameters()
        result = trainer.train(resume_from_checkpoint=str(options.resume_from_checkpoint) if options.resume_from_checkpoint else None)
        eval_metrics = {key:value for item in trainer.state.log_history for key,value in item.items() if key.startswith('eval_')}
        report.update({'status':'success','train_rows':len(train),'val_rows':len(val),'optimizer_steps':trainer.state.global_step,'actual_batch_sequence_length':actual_length,'train_metrics':result.metrics,'eval_metrics':eval_metrics,'gpu':torch.cuda.get_device_name(0),'torch_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/1024**2,2),'torch_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/1024**2,2),'nvidia_smi_max_step_sample_used_mib':max(monitor.gpu_used_mib,default=None),'elapsed_seconds':round(time.monotonic()-start,2)})
        trainer.save_model(str(run_dir / ('smoke_adapter' if smoke else 'final_adapter')))
        trainer.save_state()
        (run_dir / 'trainer_log_history.json').write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2), encoding='utf-8')
        evaluation_logs = [item for item in trainer.state.log_history if 'eval_loss' in item]
        train_logs = [item for item in trainer.state.log_history if 'loss' in item]
        best = min(evaluation_logs, key=lambda item:item['eval_loss']) if evaluation_logs else None
        report.update({'epoch':trainer.state.epoch,'final_logged_train_loss':train_logs[-1]['loss'] if train_logs else None,
                       'final_logged_train_loss_step':train_logs[-1]['step'] if train_logs else None,
                       'mean_train_loss':result.metrics.get('train_loss'),
                       'final_eval_loss':evaluation_logs[-1]['eval_loss'] if evaluation_logs else None,
                       'lowest_eval_loss':best['eval_loss'] if best else None,'lowest_eval_loss_step':best['step'] if best else None,
                       'adapter_path':str(run_dir / ('smoke_adapter' if smoke else 'final_adapter')),
                       'elapsed_seconds':round(time.monotonic()-start,2)})
    except Exception as error:
        report.update({'status':'failed','error_type':type(error).__name__,'error':str(error),'elapsed_seconds':round(time.monotonic()-start,2)})
        raise
    finally:
        (run_dir/'run_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
