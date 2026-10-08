"""Audit a small real Fineweb V3 CS sample; never train or overwrite files.
Run with the project's .venv-qlora Python. No archive extraction is performed.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import tarfile

from datasets import Dataset
import requests

REPO = 'opencsg/Fineweb-Edu-Chinese-V3'
REVISION = 'e9fb36d4968663032ba5307e154924615b5c3420'
PACKAGE = 'disciplinary_knowledge_repository_CS_sft.tar.gz'
# These are transparent multi-label screening heuristics, not verified annotations.
TOPICS = {
    'Python': r'\bpython\b|\bpandas\b|\bnumpy\b',
    'Java/C++': r'\bjava\b|\bc\+\+|\bjvm\b',
    '前端': r'\bjavascript\b|\btypescript\b|\bhtml\b|\bcss\b|\breact\b|\bvue\b|前端',
    '数据库': r'数据库|\bsql\b|\bmysql\b|\bpostgresql\b|\bdatabase\b|\bredis\b',
    'Linux': r'\blinux\b|\bunix\b|\bbash\b|\bshell\b',
    'Git': r'\bgit\b|\bgithub\b|版本控制|version control',
    '网络': r'计算机网络|网络协议|\btcp\b|\bhttp\b|\bdns\b|\bsocket\b|computer network|network protocol',
    '操作系统': r'操作系统|operating system|死锁|\bdeadlock\b|虚拟内存|virtual memory|进程调度|process scheduling',
    '数据结构与算法': r'数据结构|算法|algorithm|data structure|二叉树|binary tree|链表|linked list|动态规划|dynamic programming|复杂度|complexity',
    'AI/机器学习/深度学习': r'机器学习|深度学习|人工智能|神经网络|machine learning|deep learning|neural network|\bpytorch\b|\btensorflow\b|\btransformer\b|\bllm\b',
    '其他计算机基础': r'编译器|编程|计算机|软件工程|compiler|programming|computer science|software engineering|面向对象|object.oriented',
}
HTML = re.compile(r'<(?:html|body|script|iframe|div|span|br|p)(?:\s[^>]*|/?)>', re.I)
MISSING_IMAGE = re.compile(r'如图|见图|下图|图中|shown in (?:the )?(?:figure|image)|refer to (?:the )?(?:figure|image)', re.I)


def dump(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def length_stats(values):
    return {'min': min(values), 'max': max(values), 'mean': round(statistics.mean(values), 2), 'median': statistics.median(values)} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 1000 <= args.sample_size <= 2000:
        parser.error('Use 1000 to 2000 rows for this initial audit.')
    out = args.output_dir or Path(__file__).resolve().parent / 'data' / ('fineweb_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    products = ['report.json', 'terminal_report.txt', 'raw_sample.jsonl', 'review_sample.jsonl', 'random_10.jsonl', 'train_candidate.jsonl']
    if any((out / name).exists() for name in products):
        raise FileExistsError('Audit outputs already exist; choose a new output directory.')
    archive = args.archive or out / PACKAGE
    if not archive.exists():
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PACKAGE}'
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with archive.open('xb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
    rng = random.Random(args.seed)
    sample, counts, package_meta = [], Counter(), None
    # Reservoir sampling covers all train documents instead of taking a biased prefix.
    # Only this 8 MB package is read. Other formats are alternative exports, not extra rows.
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name.endswith('/_dataset_meta.json'):
                package_meta = json.load(tar.extractfile(member))
            if not member.name.endswith(('/train_alpaca.jsonl', '/val_alpaca.jsonl')):
                continue
            split = 'train' if member.name.endswith('/train_alpaca.jsonl') else 'validation'
            for line in tar.extractfile(member):
                if not line.strip():
                    continue
                counts[split] += 1
                if split != 'train':
                    continue
                row = json.loads(line)
                if len(sample) < args.sample_size:
                    sample.append(row)
                else:
                    index = rng.randrange(counts['train'])
                    if index < args.sample_size:
                        sample[index] = row
    if not sample:
        raise ValueError('No real train Alpaca rows found.')
    dataset = Dataset.from_list(sample)
    # Inspect actual Arrow schema before any cleaning, including nested dict/list types.
    print('dataset.column_names =', dataset.column_names)
    print('dataset.features =', dataset.features)
    print('dataset[0] =', json.dumps(dataset[0], ensure_ascii=False))
    required = {'instruction', 'input', 'output', 'metadata'}
    if set(dataset.column_names) != required:
        raise TypeError(f'Unexpected columns: {dataset.column_names}')
    for row in sample:
        for field in ('instruction', 'input', 'output'):
            if not isinstance(row[field], str):
                raise TypeError(f'{field} is not string; stop rather than stringify.')
        meta = row['metadata']
        if not isinstance(meta, dict) or not isinstance(meta.get('images'), list):
            raise TypeError('Unexpected metadata/images type.')
        for image in meta['images']:
            if not isinstance(image, dict) or not isinstance(image.get('path'), str):
                raise TypeError('Unexpected image entry type.')
    distributions = {}
    for field in sorted({key for row in sample for key in row['metadata']}):
        counter = Counter()
        for row in sample:
            value = row['metadata'].get(field)
            # Preserve the real structured values; no guessed list flattening.
            label = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else str(value)
            counter[label] += 1
        distributions[field] = dict(counter.most_common())
    questions = [' '.join(row['input'].split()) for row in sample]
    lengths = [len(row['output']) for row in sample]
    topic_counts, flags, reviews, candidates = Counter(), Counter(), [], []
    seen = set()
    for index, row in enumerate(sample):
        q, answer = row['input'].strip(), row['output'].strip()
        topic_text = re.sub(r'Algorithmic problem\.', '', q, flags=re.I) + '\n' + answer
        topics = [name for name, pattern in TOPICS.items() if re.search(pattern, topic_text, re.I)]
        topic_counts.update(topics)
        reasons = []
        if not q: reasons.append('empty_question')
        if not answer: reasons.append('empty_answer')
        if questions[index] in seen: reasons.append('duplicate_question')
        seen.add(questions[index])
        if not topics: reasons.append('no_computing_keyword_manual_review')
        if len(answer) < 30: reasons.append('short_answer_review')
        if len(answer) > 12000 or len(q) > 16000: reasons.append('long_text_review')
        if HTML.search(q + '\n' + answer): reasons.append('html_review')
        if '\ufffd' in q + answer: reasons.append('replacement_character_review')
        if row['metadata']['images'] and MISSING_IMAGE.search(q): reasons.append('missing_image_dependency_review')
        flags.update(reasons)
        reviews.append({'sample_index': index, 'topics_heuristic': topics, 'review_flags': reasons, 'record': row})
        # Suspicious content is quarantined, never destructively stripped or truncated.
        if not reasons:
            candidates.append({'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': answer}], 'metadata': row['metadata']})
    ten_indices = random.Random(args.seed + 1).sample(range(len(sample)), min(10, len(sample)))
    ten = [{'sample_index': i, 'record': sample[i]} for i in ten_indices]
    token_report = {'status': 'not_measured', 'reason': 'No cached tokenizer; character lengths are not token lengths.'}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', local_files_only=True)
        token_lengths = []
        for r in sample:
            encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': r['input']}, {'role': 'assistant', 'content': r['output']}], tokenize=True)
            ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
            token_lengths.append(len(ids))
        if max(token_lengths) < 10:
            raise ValueError('Implausible tokenizer result; stop for inspection.')
        token_report = {'status': 'measured_with_cached_Qwen_tokenizer', 'lengths': length_stats(token_lengths), 'over_512': sum(n > 512 for n in token_lengths), 'over_2048': sum(n > 2048 for n in token_lengths), 'over_4096': sum(n > 4096 for n in token_lengths)}
    except (OSError, ImportError) as error:
        token_report['error_type'] = type(error).__name__
    report = {
        'source': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION,
        'package': PACKAGE, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'global_total_card_only_not_independently_counted': 188148,
        'package_metadata': package_meta, 'actual_package_split_counts': dict(counts),
        'sample_split': 'train', 'sampling': 'uniform row reservoir, seed=' + str(args.seed),
        'checked_rows': len(dataset), 'column_names': dataset.column_names, 'features': dataset.features.to_dict(),
        'first_real_sample': dataset[0],
        'empty_instruction': sum(not r['instruction'].strip() for r in sample),
        'empty_user_question_input': sum(not r['input'].strip() for r in sample),
        'empty_output_assistant_answer': sum(not r['output'].strip() for r in sample),
        'duplicate_questions_extra_rows': len(questions) - len(set(questions)),
        'duplicate_instruction_extra_rows_not_question_duplicates': len(sample) - len({r['instruction'] for r in sample}),
        'answer_lengths_characters': length_stats(lengths),
        'question_lengths_characters': length_stats([len(r['input']) for r in sample]),
        'metadata_distributions': distributions, 'topic_keyword_distribution_multilabel': dict(topic_counts),
        'english_background_prefix_rows': sum(r['input'].startswith('**Background**') for r in sample),
        'quality_flags': dict(flags), 'candidate_rows_pending_manual_review': len(candidates),
        'token_check': token_report, 'random_10': ten,
        'conclusion': 'Suitable as one SFT source after topic/manual correctness review; not ready for the existing 512-token training configuration.',
        'rules': ['Question is input, not boilerplate instruction.', 'Keep full questions, code, Markdown and LaTeX; never truncate answers.', 'Exclude empty QA and exact whitespace-normalized duplicate questions.', 'Quarantine heuristic topic misses, unusual lengths, apparent HTML, replacement characters and image-dependent questions; review before rejecting.', 'Do not filter quality_score=0.0 or reject every images list.', 'Use one export format only; keep original validation separate.', 'Future split by source_paper to avoid document leakage; check duplicates across splits.'],
        'risks': ['Computer domain labels include unrelated finance content; keyword hits can also be false positives.', 'Bilingual long textbook questions differ from short Chinese tutoring conversations.', 'Synthetic answers need factual/code review; automated flags do not establish correctness.', 'Image files are absent; references may be incomplete.', 'License is other; review OpenCSG agreement and commercial authorization.', 'Current model is 0.5B and max_length=512; measure tokens and choose complete shorter samples or adjust later.'],
        'next_step': 'Manually inspect random_10 and review_sample; check broad-topic coverage, then consider a bounded CSDN sample for practical programming. No training yet.'
    }
    dump(out / 'report.json', report)
    jsonl(out / 'raw_sample.jsonl', sample)
    jsonl(out / 'review_sample.jsonl', reviews)
    jsonl(out / 'random_10.jsonl', ten)
    jsonl(out / 'train_candidate.jsonl', candidates)
    lines = ['===== Fineweb V3 计算机数据小样本审计 =====', '来源: ' + report['source'], '固定 revision: ' + REVISION, '子集包: ' + PACKAGE,
             '全库总量（发布者声明，未全量读取）: 188148', '实际包内条数: ' + json.dumps(dict(counts)), '检查训练样本: ' + str(len(sample)),
             '字段: ' + str(dataset.column_names), '真实字段类型: ' + str(dataset.features),
             '空 instruction / 问题 input / output: ' + str((report['empty_instruction'], report['empty_user_question_input'], report['empty_output_assistant_answer'])),
             '重复实际问题: ' + str(report['duplicate_questions_extra_rows']), '回答字符长度: ' + str(report['answer_lengths_characters']),
             '领域分布: ' + str(distributions.get('domain')), '题型分布: ' + str(distributions.get('question_type')),
             '来源文档数: ' + str(len(distributions.get('source_paper', {}))), '完整来源/metadata 分布见 report.json',
             '计算机主题（关键词多标签，需人工确认）: ' + str(dict(topic_counts)), '质量标记: ' + str(dict(flags)),
             '建议保留待审候选: ' + str(len(candidates)) + '；不是已批准训练数据', '真实 token 检查: ' + str(token_report),
             '结论: ' + report['conclusion'], '清洗规则: ' + json.dumps(report['rules'], ensure_ascii=False),
             '风险: ' + json.dumps(report['risks'], ensure_ascii=False), '下一步: ' + report['next_step'],
             '\n===== 第一条真实抽样数据（完整） =====', json.dumps(dataset[0], ensure_ascii=False, indent=2),
             '\n===== 随机 10 条真实样本（完整，不截断） =====', json.dumps(ten, ensure_ascii=False, indent=2),
             '输出目录: ' + str(out.resolve())]
    terminal = '\n'.join(lines)
    with (out / 'terminal_report.txt').open('x', encoding='utf-8') as f:
        f.write(terminal)
    print(terminal)


if __name__ == '__main__':
    main()
