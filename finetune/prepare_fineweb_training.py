import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import tarfile

from datasets import Dataset
from transformers import AutoTokenizer

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
# Reject only affirmative unrelated evidence without computing evidence.
CS = re.compile(r'python|\bjava\b|c\+\+|\bsql\b|数据库|编程|计算机|算法|操作系统|linux|\bgit\b|javascript|typescript|\bhtml\b|\bcss\b|compiler|programming|computer|software|algorithm|data structure|database|neural|machine learning|deep learning|\bnetwork\b|\bprotocol\b|\bprocessor\b|\bmemory\b|\bcpu\b|\bgpu\b|\bcode\b|\bdata\b|\binformation\b|\bdigital\b|\belectronic\b|信息|网络|神经|数据|软件|电子|数字', re.I)
UNRELATED = re.compile(r'quantitative trading|trend.following|grid strateg|stock market|portfolio|financial market|投资|股票|量化交易|金融|profitability|macroeconomic|monetary policy|fiscal policy|市场营销|marketing strateg|medical diagnosis|clinical|patient|cancer|临床|病人|癌症|chemical reaction|thermodynamic|quantum mechanics|化学反应|热力学|量子力学|法律|司法|诉讼|literary|poetry|文学|诗歌', re.I)
HTML_TAG = re.compile(r'<(?:!DOCTYPE|/?(?:html|head|body|script|style|div|span|p|a|br|table|tr|td))\b[^>]*>', re.I)
COMPUTING_CONTEXT = re.compile(r'\b(?:python|java|sql|linux|database|computer|software|programming|neural|robot|robots|robotics|blockchain)\b|machine learning|deep learning|data mining|recommendation system|recommender|人工智能|机器学习|深度学习|推荐系统|机器人|编程|计算机|数据库|操作系统', re.I)


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def token_ids(tokenizer, messages):
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return result['input_ids'] if hasattr(result, 'keys') else result


def main():
    p = argparse.ArgumentParser(description='Preserve original CS train/validation; backup before replacement.')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    args = p.parse_args()
    data = args.data_dir
    data.mkdir(parents=True, exist_ok=True)
    run = data / ('fineweb_prepared_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir()
    raw = {'train': [], 'val': []}
    with tarfile.open(args.archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile(): continue
            split = 'train' if member.name.endswith('/train_alpaca.jsonl') else 'val' if member.name.endswith('/val_alpaca.jsonl') else None
            if split:
                for number, line in enumerate(tar.extractfile(member), 1):
                    if line.strip():
                        raw[split].append((json.loads(line), member.name, number))
    # Print and persist actual complete schema before deciding which fields to convert.
    for split in raw:
        type_counts = {field: dict(Counter(type(item[0].get(field)).__name__ for item in raw[split])) for field in ('instruction','input','output','metadata')}
        print(split, 'actual Python type distribution:', type_counts, flush=True)
        # Mixed malformed values cannot be coerced through Arrow; inspect a real
        # row's features, then validate every raw QA value explicitly below.
        ds = Dataset.from_list([raw[split][0][0]])
        print(split, 'column_names:', ds.column_names, 'features:', ds.features, flush=True)
        if not {'instruction','input','output','metadata'} <= set(ds.column_names):
            raise TypeError('Unexpected schema')
    kept, provenance, rejected, counts = {}, {}, [], {}
    seen_questions, seen_samples = set(), set()
    # Validation processed first: preserve its membership; remove any overlap from train.
    for split in ('val', 'train'):
        kept[split], provenance[split] = [], []
        counter = Counter()
        for row, source, number in raw[split]:
            q, answer = row.get('input'), row.get('output')
            reason = None
            if not isinstance(q, str) or not isinstance(answer, str): reason = 'abnormal_type'
            elif not q.strip() or not answer.strip(): reason = 'empty'
            else:
                q, answer = q.strip(), answer.strip()
                # Exact question equality after trimming only; no semantic deduplication.
                pair = (q, answer)
                text = q + '\n' + answer
                topic_text = re.sub(r'Algorithmic problem\.|\*\*Data / Model Specification\*\*', '', text, flags=re.I)
                # Generic words such as data/information/digital occur in finance
                # and in export templates; they are not computing evidence.
                topic_text = re.sub(r'\b(?:data|information|digital)\b|信息|数据|数字', '', topic_text, flags=re.I)
                if q in seen_questions or pair in seen_samples: reason = 'duplicate'
                elif (UNRELATED.search(topic_text) or re.search(r'\binvest(?:or|ors|ment|ments)\b|financial asset|security token offerings|\bdividends\b|\barbitrage\b|regulatory oversight', topic_text, re.I)) and not CS.search(topic_text.split('**The Questions**')[-1]) and not COMPUTING_CONTEXT.search(topic_text) and '```' not in q: reason = 'non_computing'
                elif text.count('\ufffd') >= 3 or '\x00' in text or sum(ord(c)<32 and c not in '\n\r\t' for c in text) / len(text) > 0.01: reason = 'corrupt_text'
                elif len(re.findall(r'[\w\u4e00-\u9fff]', text)) < 5: reason = 'broken_text'
                elif HTML_TAG.search(text) and len(re.sub(r'<[^>]*>|\s', '', text)) < 30: reason = 'pure_html_noise'
            if reason:
                counter[reason] += 1
                rejected.append({'split': split, 'reason': reason, 'archive_member': source, 'line': number, 'record': row})
            else:
                seen_questions.add(q)
                seen_samples.add((q,answer))
                kept[split].append({'messages':[{'role':'user','content':q},{'role':'assistant','content':answer}]})
                provenance[split].append({'output_index':len(kept[split])-1,'archive_member':source,'line':number,'metadata':row['metadata']})
        counts[split] = {'raw':len(raw[split]), 'kept':len(kept[split]), 'removed':dict(counter)}
        if not kept[split]: raise ValueError('Empty cleaned split')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lengths = [len(token_ids(tokenizer, row['messages'])) for row in kept['train']]
    ordered = sorted(lengths)
    stats = {'min':min(lengths),'mean':round(statistics.mean(lengths),2),'median':statistics.median(lengths),'P90':ordered[__import__('math').ceil(0.9*len(ordered))-1], 'P90_method':'nearest rank','max':max(lengths), 'over_512_count':sum(n>512 for n in lengths),'over_512_ratio':sum(n>512 for n in lengths)/len(lengths),'over_1024_count':sum(n>1024 for n in lengths),'over_1024_ratio':sum(n>1024 for n in lengths)/len(lengths)}
    for split in ('train','val'):
        write_jsonl(run / (split + '.jsonl'),kept[split])
        write_jsonl(run / (split + '_provenance.jsonl'),provenance[split])
    write_jsonl(run/'rejected.jsonl',rejected)
    backup = run/'backup'
    backup.mkdir()
    existing = {}
    for split in ('train','val'):
        target = data/(split+'.jsonl')
        if target.exists():
            content = target.read_bytes()
            existing[split] = {'rows':sum(bool(line.strip()) for line in content.splitlines()),'sha256':hashlib.sha256(content).hexdigest(),'backup':str(backup/target.name)}
            shutil.copy2(target,backup/target.name)
    report = {'source':'opencsg/Fineweb-Edu-Chinese-V3','revision':'e9fb36d4968663032ba5307e154924615b5c3420','package':args.archive.name,'archive_sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest(),'split_policy':'original membership; no re-split; validation wins exact cross-split duplicates','counts':counts,'train_token_lengths':stats,'first_messages':kept['train'][0],'previous_files':existing,'note':'No length filtering, no keyword whitelist; only affirmative non-CS evidence without computing evidence. Ambiguous content retained. Markdown, code, formulas unchanged except outer trim. Synthetic correctness not guaranteed.'}
    (run/'preparation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for split in ('train','val'):
        shutil.copy2(run/(split+'.jsonl'),data/(split+'.jsonl'))
    assert not {r['messages'][0]['content'] for r in kept['train']} & {r['messages'][0]['content'] for r in kept['val']}
    for split in ('train','val'):
        assert all(len(r['messages'])==2 and set(r)=={'messages'} for r in kept[split])
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('REPORT:',run/'preparation_report.json',flush=True)


if __name__ == '__main__': main()
