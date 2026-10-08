"""Gate acceptance against real retrieval; no_hit must never call the LLM."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_knowledge_upload as helpers
from fastapi.testclient import TestClient

rag, main, ROOT = helpers.rag, helpers.main, helpers.ROOT


class RetrievalGateAcceptance(unittest.TestCase):
    def test_gate(self):
        baseline = json.loads((ROOT/'evaluation'/'retrieval_gate_baseline.json').read_text(encoding='utf-8'))
        self.assertEqual(rag.load_state(), baseline['knowledge_state'])
        actual = []
        for case in baseline['rows']:
            result = rag.retrieve_with_confidence(case['question'])
            expected = 'hit' if case['top1_score'] is not None and case['top1_score'] >= rag.RETRIEVAL_SCORE_THRESHOLD else 'no_hit'
            self.assertEqual(result['retrieval_status'], expected, case['id'])
            if expected == 'no_hit':
                self.assertEqual(result['hits'], [])
            actual.append({'id':case['id'], 'label':case['label'], 'status':result['retrieval_status'], 'top_score':result['top_score']})
        with TestClient(main.app) as api:
            hit = api.post('/ask',json={'question':'想在Python列表的末尾增加一个元素，要怎么做？'})
            self.assertEqual(hit.status_code,200,hit.text)
            self.assertEqual(hit.json()['retrieval_status'],'hit')
            self.assertIn('append',hit.json()['answer'])
            self.assertIn({'source':'python.txt','page':None},hit.json()['sources'])
            self.assertNotIn('top_score',hit.json())
            paraphrase=rag.retrieve_with_confidence('Python的list要在最后插入新元素，有什么方法？')
            self.assertEqual(paraphrase['retrieval_status'],'hit')
            self.assertTrue(any(d['source']=='python.txt' for d in paraphrase['hits']))

            rejected=[]
            with patch.object(rag,'chat_with_llm',side_effect=AssertionError('no_hit must not invoke LLM')) as llm:
                for question in ['量子纠缠实验怎样检验贝尔不等式？','怎么把Git仓库历史里的泄漏密钥彻底移除？']:
                    response=api.post('/ask',json={'question':question})
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.json()['retrieval_status'],'no_hit')
                    self.assertEqual(response.json()['sources'],[])
                    self.assertEqual(response.json()['retrieved_context'],'')
                    self.assertEqual(response.json()['answer'],'当前知识库中没有找到足够相关的资料。')
                    rejected.append(response.json())
                llm.assert_not_called()

            # Preserve list-returning tools; empty lists carry no irrelevant text.
            tool_call=SimpleNamespace(function=SimpleNamespace(name='search_knowledge',arguments=json.dumps({'question':'量子纠缠实验怎样检验贝尔不等式？'})))
            self.assertEqual(json.loads(helpers.agent.execute_tool(tool_call)),[])
            self.assertEqual(helpers.langgraph_agent.search_knowledge.invoke({'question':'量子纠缠实验怎样检验贝尔不等式？'}),[])
            self.assertTrue(helpers.langgraph_agent.search_knowledge.invoke({'question':'想在Python列表的末尾增加一个元素，要怎么做？'}))

            # Boundary behavior is deterministic and uses the observed score scale.
            document={'text':'boundary','source':'boundary.txt','page':None,'score':rag.RETRIEVAL_SCORE_THRESHOLD}
            with patch.object(rag,'_retrieve',return_value=[document]):
                self.assertEqual(rag.retrieve_with_confidence('boundary')['retrieval_status'],'hit')
            with patch.object(rag,'_retrieve',return_value=[dict(document,score=rag.RETRIEVAL_SCORE_THRESHOLD-0.00001)]):
                self.assertEqual(rag.retrieve_with_confidence('boundary')['retrieval_status'],'no_hit')
        self.assertEqual(helpers.finetune_manifest(),helpers.FINETUNE_BEFORE)
        report=dict(status='passed',threshold=rag.RETRIEVAL_SCORE_THRESHOLD,
                    activation_fn=type(rag.reranker.activation_fn).__name__,
                    calibrated_rows=actual, hit_example=hit.json(), no_hit_examples=rejected,
                    synonym_hit=True,no_hit_llm_calls=0,agent_tools_compatible=True,
                    finetune_unchanged=True)
        (ROOT/'docs'/'retrieval_gate_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        rag.client.close()
