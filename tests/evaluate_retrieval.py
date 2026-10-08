"""Measure the unchanged raw retrieval pipeline; no LLM or indexing changes."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rag


def summarize(values):
    return {'scored_count':len(values), 'min':min(values) if values else None,
            'mean':statistics.mean(values) if values else None,
            'median':statistics.median(values) if values else None,
            'max':max(values) if values else None}


def evaluate():
    dataset = json.loads((ROOT/'evaluation'/'retrieval_eval_set.json').read_text(encoding='utf-8'))
    rows = []
    for case in dataset:
        with rag.KNOWLEDGE_LOCK:
            hits = rag._retrieve(case['question'])
        # Evidence is an explicit human-authored rubric, never another LLM judge.
        supported = [d for d in hits if d['source'] in case.get('expected_sources', [case.get('expected_source')])
                     and all(term in d['text'] for term in case.get('evidence_terms', []))]
        rows.append(dict(case, top1_score=hits[0]['score'] if hits else None,
                         top3_scores=[d['score'] for d in hits],
                         top1_source=hits[0]['source'] if hits else None,
                         top1_page=hits[0]['page'] if hits else None,
                         correct_evidence_in_top3=bool(supported), hits=hits))
        print(case['id'], case['label'], rows[-1]['top1_score'], flush=True)
    positive = [r['top1_score'] for r in rows if r['label']=='positive' and r['top1_score'] is not None]
    negative = [r['top1_score'] for r in rows if r['label']=='negative' and r['top1_score'] is not None]
    # Conservative calibration policy: strictly above every measured negative.
    threshold = math.ceil((max(negative)+0.000001)*10)/10
    lo, hi = max(min(positive),min(negative)), min(max(positive),max(negative))
    fp = [r['id'] for r in rows if r['label']=='negative' and r['top1_score'] is not None and r['top1_score']>=threshold]
    fn = [r['id'] for r in rows if r['label']=='positive' and (r['top1_score'] is None or r['top1_score']<threshold)]
    report = dict(reranker='BAAI/bge-reranker-base', vector_threshold=0.4,
                  score_activation=type(rag.reranker.activation_fn).__name__, knowledge_state=rag.load_state(), questions=len(rows), positive=summarize(positive), negative=summarize(negative),
                  unscored_ids=[r['id'] for r in rows if r['top1_score'] is None],
                  overlap=[lo,hi] if lo<=hi else None, recommended_threshold=threshold,
                  threshold_policy='Round up above highest scored negative; calibration only, not held-out accuracy.',
                  false_positive_ids=fp, false_negative_ids=fn,
                  accepted_without_answer_evidence_ids=[r['id'] for r in rows if r['label']=='positive' and r['top1_score'] is not None and r['top1_score']>=threshold and not r['correct_evidence_in_top3']],
                  positive_evidence_missing_ids=[r['id'] for r in rows if r['label']=='positive' and not r['correct_evidence_in_top3']], rows=rows)
    output=ROOT/'evaluation'/'retrieval_gate_baseline.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Retrieval gate calibration', '', 'No LLM evaluation. Evidence is checked using source and hand-authored answer terms.', '',
           'Missing candidates have null scores; they are not counted as zero.', '',
           'Statistics: '+json.dumps({k:report[k] for k in ['positive','negative','overlap','recommended_threshold','false_positive_ids','false_negative_ids']},ensure_ascii=False), '',
           '| ID | label | question | top1 | source | evidence in Top-3 |','|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['label']} | {r['question']} | {r['top1_score']} | {r['top1_source']} | {r['correct_evidence_in_top3']} |")
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        evaluate()
    finally:
        rag.client.close()
