"""Real PDF/TXT/MD retrieval citation acceptance; existing knowledge restored."""
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import test_knowledge_upload as helpers
from pdf_fixtures import pdf_bytes
from fastapi.testclient import TestClient

rag, main, ROOT = helpers.rag, helpers.main, helpers.ROOT


class RetrievalSourcesAcceptance(unittest.TestCase):
    assert_points_preserved = helpers.KnowledgeUploadAcceptance.assert_points_preserved

    def test_sources(self):
        original_state = rag.load_state()
        original_points = helpers.point_snapshot()
        tag = uuid.uuid4().hex[:12]
        names = [f'citation_{tag}.pdf', f'citation_{tag}.txt', f'citation_{tag}.md']
        paths = [rag.UPLOADS_DIR / name for name in names]
        self.assertTrue(all(not path.exists() for path in paths))
        report = {}
        try:
            with TestClient(main.app) as api:
                fact = '霜桥学习索引协议的恢复口令是霜桥-8259，重试间隔为29秒。'
                content = pdf_bytes([
                    {'text': '课程介绍：Python列表支持追加元素。'},
                    {'text': (fact+'\n')*18},
                    {'text': '课程结语：Linux命令用于管理文件。'}])
                response = api.post('/knowledge/upload', files={'file': (names[0], content, 'application/pdf')})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['pages'], 3)
                question = '霜桥学习索引协议的恢复口令和重试间隔分别是什么？'
                with patch.object(rag, 'retrieve_with_confidence', wraps=rag.retrieve_with_confidence) as retrieve:
                    response = api.post('/ask', json={'question': question})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(retrieve.call_count, 1)
                answer = response.json()
                self.assertIn('霜桥-8259', answer['answer'])
                self.assertIn('29', answer['answer'])
                self.assertEqual(answer['sources'], [{'source': 'uploads/'+names[0], 'page': 2}])
                documents = rag.retrieve(question)
                self.assertEqual(len(documents), 3)
                self.assertTrue(all(d['source'] == 'uploads/'+names[0] and d['page'] == 2 for d in documents))
                self.assertTrue(all(set(d) == {'text','source','page','score'} and isinstance(d['score'], float) for d in documents))
                report['pdf_upload'] = {'pages': 3, 'page_2_chunks': sum(p.payload.get('page') == 2 for p in helpers.points('uploads/'+names[0]))}
                report['pdf_ask'] = answer
                report['same_page_hits'] = len(documents)
                report['deduplicated_sources'] = len(answer['sources'])

                for name, label, code in [(names[1], '晴岚课程缓存', '晴岚-4618'), (names[2], '墨羽课程队列', '墨羽-7392')]:
                    text = f'{label}的恢复口令是{code}。这是独有的计算机学习资料。'
                    uploaded = api.post('/knowledge/upload', files={'file': (name, text.encode('utf-8'), 'text/plain')})
                    self.assertEqual(uploaded.status_code, 200, uploaded.text)
                    response = api.post('/ask', json={'question': f'{label}的恢复口令是什么？'})
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    self.assertIn(code, result['answer'])
                    self.assertIn({'source':'uploads/'+name, 'page': None}, result['sources'])
                    self.assertTrue(all(item['page'] is None for item in result['sources'] if item['source'].endswith(('.txt','.md'))))
                    report[name.rsplit('.',1)[1]+'_ask'] = result

                # Run both actual tool paths; metadata is carried in tool JSON.
                tool_call = SimpleNamespace(function=SimpleNamespace(name='search_knowledge', arguments=json.dumps({'question': question})))
                tool_documents = json.loads(helpers.agent.execute_tool(tool_call))
                self.assertTrue(any(d['source'] == 'uploads/'+names[0] and d['page'] == 2 for d in tool_documents))
                graph_documents = helpers.langgraph_agent.search_knowledge.invoke({'question': question})
                self.assertTrue(any(d['source'] == 'uploads/'+names[0] and d['page'] == 2 for d in graph_documents))
                report['agent_metadata_preserved'] = True

                # Deterministically exercise the existing no-candidate branch.
                with patch.object(rag.client, 'query_points', return_value=SimpleNamespace(points=[])), patch.object(rag.reranker, 'predict') as rerank, patch.object(rag, 'chat_with_llm', return_value='模型正文可能写出虚构.pdf 第99页'):
                    response = api.post('/ask', json={'question': '无可靠检索命中'})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['sources'], [])
                    self.assertEqual(response.json()['retrieved_context'], '')
                    rerank.assert_not_called()
                    report['no_candidate_branch_controlled'] = response.json()
                # An invented answer must never change sources from real retrieval.
                with patch.object(rag, 'chat_with_llm', return_value='引用不存在.pdf第99页'):
                    response = api.post('/ask', json={'question': question})
                    self.assertEqual(response.json()['sources'], [{'source':'uploads/'+names[0], 'page':2}])
                    report['invented_llm_citation_ignored'] = True
                report['status'] = 'passed'
        finally:
            for path in paths:
                self.assertEqual(path.resolve().parent, rag.UPLOADS_DIR.resolve())
                path.unlink(missing_ok=True)
            rag.sync_knowledge()
            self.assertEqual(rag.load_state(), original_state)
            self.assert_points_preserved(helpers.point_snapshot(), original_points)
            self.assertEqual(helpers.finetune_manifest(), helpers.FINETUNE_BEFORE)
            report.update(original_knowledge_preserved=True, finetune_unchanged=True, test_uploads_cleaned=True)
            (ROOT/'docs'/'rag_sources_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        rag.client.close()
