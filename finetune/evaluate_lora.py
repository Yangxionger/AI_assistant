"""Paired deterministic evaluation; original answers only, no scoring or training."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
ADAPTER_DIR = BASE_DIR / 'output' / 'formal_v1' / 'final_adapter'
OUTPUT_DIR = BASE_DIR / 'evaluation'
MAX_NEW_TOKENS = 512
QUESTIONS = [
    ('Python', '一个 Python 函数用 def collect(item, bucket=[]) 收集元素，多次调用后却保留了上次的结果。为什么会这样？如何修正，并说明修正后如何允许调用方主动共享同一个列表？'),
    ('Python', '处理百万条日志时，列表推导式和生成器表达式应该如何选择？比较内存占用、重复遍历和延迟执行的区别，并给出一个生成器更适合的场景。'),
    ('数据结构与算法', '一个服务需要经常按用户 ID 查找记录，同时按插入顺序遍历。单独使用链表或哈希表分别有什么代价？你会如何选择或组合数据结构？说明平均时间复杂度和内存方面的权衡。'),
    ('数据结构与算法', '给一组可能包含负数的整数，求连续子数组的最大和。为什么不能简单累加所有正数？解释一种 O(n) 的做法，并说明当数组全为负数时该如何处理。'),
    ('MySQL/数据库', 'MySQL 的订单表有联合索引 (user_id, created_at)。比较只按 user_id 查、只按 created_at 查、同时按两列查时利用索引的情况，并说明为什么不能直接把没有命中索引的查询判定为数据库故障。'),
    ('MySQL/数据库', '两个并发请求都读取到某商品库存为 1，然后各自下单并把库存改成 0，导致卖出两件。为什么普通的先查后改会出问题？比较原子条件 UPDATE 和事务内 SELECT FOR UPDATE 的处理思路。'),
    ('Linux', 'Linux 服务器磁盘显示已满，但删除一个很大的日志文件后 df 的可用空间没有变化。可能是什么原因？如何检查文件是否仍被进程打开，并给出谨慎的处理步骤。'),
    ('Linux', '你需要让一个部署目录中的脚本可以执行，为什么不应该直接 chmod -R 777？解释读、写、执行权限对文件和目录的不同含义，并给出更合理的授权思路。'),
    ('Git', '一个错误提交已经推送到多人共享的 main 分支。比较 git revert 与 git reset 后强制推送的影响，并解释此时通常为什么优先使用 revert。'),
    ('Git', '你在功能分支上开发，而 main 已增加多个提交。比较 merge 和 rebase 对提交历史的影响；如果功能分支已被同事共同使用，rebase 会带来什么协作风险？'),
    ('计算机网络', '在浏览器输入一个 HTTPS 域名后，为什么 DNS 解析成功仍不代表网页一定能打开？按 DNS、TCP、TLS、HTTP 的顺序解释可能的故障位置，以及如何逐层定位。'),
    ('计算机网络', '实时语音和文件下载为什么可能选择不同的传输方案？比较 TCP 与 UDP 在可靠性、顺序、延迟方面的特点，并解释 UDP 本身不可靠是否意味着应用层不能实现可靠传输。'),
    ('操作系统', '一个程序启动很多线程处理 CPU 密集计算，吞吐量却没有持续提高。解释 CPU 核数、线程调度和上下文切换的影响，并与 I/O 密集任务比较。不要把所有编程语言的线程实现都当作相同。'),
    ('操作系统', '两个线程分别持有锁 A 和锁 B，并等待对方释放另一把锁，程序卡住了。这为什么构成死锁？解释统一加锁顺序如何预防这种问题，并说明它与简单增加等待时间有什么不同。'),
    ('前端基础', '前端同时使用 localStorage 和 HttpOnly Cookie 保存登录凭据时，面对 XSS 和 CSRF 的风险有什么区别？说明 HttpOnly、SameSite、Secure 各自解决什么问题，为什么不能只靠一个属性保证安全。'),
    ('前端基础', '网页请求自己的后端时遇到 CORS 错误，为什么用 Postman 请求却成功？解释同源策略由谁执行、预检请求的作用，以及为什么前端添加 Access-Control-Allow-Origin 请求头通常不能解决问题。'),
    ('AI/机器学习基础', '一个分类模型训练准确率很高，验证准确率却明显较低。除增加模型参数外，你会检查哪些原因？解释过拟合、数据泄漏、训练验证分布差异，并比较正则化和增加高质量数据的作用。'),
    ('AI/机器学习基础', '为计算机学习助手引入新版本软件知识时，比较 RAG 和 LoRA 微调的作用、成本与局限。为什么微调后的模型仍可能需要检索？请用一个涉及版本更新的问题说明。'),
]


def answer_block(answer):
    # A generated answer may end inside a code fence at the token limit.
    # An outer text fence preserves the exact answer and isolates its Markdown.
    fence = '`' * max(4, max((len(run) for run in re.findall(r'`+', answer)), default=0) + 1)
    return fence + 'text\n' + answer + '\n' + fence


def inspect_answer(answer, ids, tokenizer):
    flags = []
    if not answer.strip(): flags.append('empty')
    if '\ufffd' in answer or '\x00' in answer: flags.append('invalid_character')
    if len(ids) >= MAX_NEW_TOKENS: flags.append('token_limit_reached_possible_truncation')
    eos = tokenizer.eos_token_id
    if ids and ids[-1] == eos and len(ids) >= MAX_NEW_TOKENS:
        flags = [flag for flag in flags if flag != 'token_limit_reached_possible_truncation']
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) >= 6 and Counter(lines).most_common(1)[0][1] >= 4: flags.append('repeated_lines_manual_review')
    if len(answer.strip()) < 10: flags.append('very_short_manual_review')
    return flags


def main():
    if not (ADAPTER_DIR / 'adapter_model.safetensors').exists():
        raise FileNotFoundError(ADAPTER_DIR)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / 'formal_v1_comparison.json'
    md_path = OUTPUT_DIR / 'formal_v1_comparison.md'
    if json_path.exists() or md_path.exists():
        raise FileExistsError('Comparison exists; preserve it instead of overwriting.')
    originals = {}
    for split in ('train', 'val'):
        path = BASE_DIR / 'data' / (split + '.jsonl')
        originals[split] = {json.loads(line)['messages'][0]['content'].strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}
    for _, q in QUESTIONS:
        if any(q.strip() in values for values in originals.values()):
            raise ValueError('Evaluation question exactly matches existing train/val data.')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,local_files_only=True,quantization_config=quantization,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    generation = deepcopy(base.generation_config)
    generation.update(max_new_tokens=MAX_NEW_TOKENS,do_sample=False,num_beams=1,temperature=1.0,top_p=1.0,repetition_penalty=1.0,use_cache=True,pad_token_id=tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=False)
    model = PeftModel.from_pretrained(base,str(ADAPTER_DIR),is_trainable=False)
    model.eval()
    results = []
    payload = {'metadata': {'created_at':datetime.now().isoformat(timespec='seconds'),'model':MODEL_NAME,'base_revision':getattr(base.config,'_commit_hash',None),'adapter':str(ADAPTER_DIR),'adapter_sha256':hashlib.sha256((ADAPTER_DIR/'adapter_model.safetensors').read_bytes()).hexdigest(),'precision':'Both models share the same 4bit NF4 base, bfloat16 compute/dtype and SDPA; base uses disable_adapter(), LoRA uses adapter enabled.','generation_config':generation.to_dict(),'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),'question_count':len(QUESTIONS),'novelty_check':'Newly written questions, no exact question match against train or validation; semantic topic overlap not excluded.','automatic_scoring':False}, 'results':results}
    start = time.monotonic()
    with json_path.open('x',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
    for i, (topic, question) in enumerate(QUESTIONS,1):
        prompt = tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
        inputs = tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
        item = {'id':i,'topic':topic,'question':question,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
        for label in ('base','lora'):
            begun = time.monotonic()
            context = model.disable_adapter() if label=='base' else nullcontext()
            with context, torch.inference_mode():
                output = model.generate(**inputs,generation_config=deepcopy(generation))
            ids = output.sequences[0,inputs['input_ids'].shape[1]:].tolist()
            answer = tokenizer.decode(ids,skip_special_tokens=True)
            item[label+'_answer'] = answer
            item[label+'_generated_tokens_including_stop'] = len(ids)
            item[label+'_answer_characters'] = len(answer)
            item[label+'_seconds'] = round(time.monotonic()-begun,2)
            item[label+'_flags'] = inspect_answer(answer,ids,tokenizer)
            print(f'[{i}/{len(QUESTIONS)}] {topic} {label}: tokens={len(ids)}, chars={len(answer)}, flags={item[label+"_flags"]}',flush=True)
        results.append(item)
        # Preserve partial raw answers if a later generation fails.
        json_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    summary = {'completed_questions':len(results),'elapsed_seconds':round(time.monotonic()-start,2)}
    for label in ('base','lora'):
        summary[label] = {'successful_generations':len(results),'mean_generated_tokens_including_stop':round(statistics.mean(r[label+'_generated_tokens_including_stop'] for r in results),2),'mean_answer_characters':round(statistics.mean(r[label+'_answer_characters'] for r in results),2),'empty_answers':sum(not r[label+'_answer'].strip() for r in results),'flag_counts':dict(Counter(flag for r in results for flag in r[label+'_flags']))}
    payload['summary'] = summary
    json_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# formal_v1 Base / LoRA 原始答案对比','','18 个新编计算机问题；已检查与 train/val 问题无完全相同项。概念重叠不代表没有见过相关知识。','',f'模型：{MODEL_NAME}；Adapter：`{ADAPTER_DIR}`。','',f'同一个 4bit NF4 基础模型，bfloat16 compute，同一 chat template、相同输入与生成配置；do_sample=False、max_new_tokens={MAX_NEW_TOKENS}。Base 禁用 adapter，LoRA 启用 adapter。','', '不使用另一 LLM，不自动评分。下方保留完整输出；异常标记仅提示人工检查。','', '## 生成统计','', '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','']
    for r in results:
        lines.extend([f'## {r["id"]}. {r["topic"]}','','**问题**','',r['question'],'','### Base 原始回答','',answer_block(r['base_answer']),'',f'字符数：{r["base_answer_characters"]}；生成 tokens（含停止 token）：{r["base_generated_tokens_including_stop"]}；标记：{r["base_flags"]}。','','### formal_v1 LoRA 原始回答','',answer_block(r['lora_answer']),'',f'字符数：{r["lora_answer_characters"]}；生成 tokens（含停止 token）：{r["lora_generated_tokens_including_stop"]}；标记：{r["lora_flags"]}。',''])
    with md_path.open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Saved:',json_path,md_path,flush=True)


if __name__ == '__main__': main()
