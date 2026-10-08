"""Real /agent local/web/none acceptance plus fail-closed controls."""
import json
import os
import unittest
from unittest.mock import patch
import uuid

import test_knowledge_upload as helpers
from pdf_fixtures import pdf_bytes
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

rag, main, ROOT = helpers.rag, helpers.main, helpers.ROOT
lg = helpers.langgraph_agent


class WebFallbackAcceptance(unittest.TestCase):
    assert_points_preserved = helpers.KnowledgeUploadAcceptance.assert_points_preserved

    def test_provider_contract(self):
        import web_search as ws
        # Controlled contract tests are separate from live B acceptance.
        payload={'results':[
            {'title':'Real API title','url':'https://example.org/page','content':'Document snippet'},
            {'title':'Duplicate','url':'https://example.org/page#section','content':'Duplicate'},
            {'title':'Bad scheme','url':'javascript:alert(1)','content':'Invalid'},
            {'title':'Empty','url':'https://example.org/empty','content':''}]}
        import httpx
        response=httpx.Response(200,json=payload,request=httpx.Request('POST','https://api.tavily.com/search'))
        with patch.dict(os.environ,{'WEB_SEARCH_PROVIDER':'tavily','TAVILY_API_KEY':'controlled-test-key'}), patch.object(ws.httpx,'post',return_value=response) as request:
            self.assertEqual(ws.web_search('test'),[{'title':'Real API title','url':'https://example.org/page','snippet':'Document snippet'}])
            self.assertEqual(request.call_args.kwargs['json']['max_results'],3)
            self.assertFalse(request.call_args.kwargs['json']['include_answer'])
        with patch.dict(os.environ,{'WEB_SEARCH_PROVIDER':'tavily','TAVILY_API_KEY':''}), patch.object(ws.httpx,'post') as request:
            self.assertEqual(ws.web_search('test'),[])
            request.assert_not_called()

    def test_live_paths(self):
        before_state = rag.load_state()
        before_points = helpers.point_snapshot()
        name = 'web_local_'+uuid.uuid4().hex[:12]+'.pdf'
        path = rag.UPLOADS_DIR/name
        self.assertFalse(path.exists())
        thread = 'web_acceptance_'+uuid.uuid4().hex
        report = {}
        searches = []
        real_search = lg.search_web

        def recorded_search(question):
            results = real_search(question)
            searches.append({'question':question, 'results':results})
            return results

        try:
            with TestClient(main.app) as api, patch.object(lg,'search_web',side_effect=recorded_search) as search:
                content = pdf_bytes([
                    {'text':'Python列表是有序可变的数据结构。'},
                    {'text':('青禾同步协议的恢复口令是青禾-6284，重试间隔为31秒。\n')*8},
                    {'text':'Git分支隔离不同功能的开发。'}])
                response = api.post('/knowledge/upload',files={'file':(name,content,'application/pdf')})
                self.assertEqual(response.status_code,200,response.text)
                local_question = '青禾同步协议的恢复口令和重试间隔是什么？'
                with patch.object(lg,'retrieve_with_confidence',wraps=lg.retrieve_with_confidence) as retrieve:
                    response = api.post('/agent',json={'question':local_question,'session_id':thread})
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(retrieve.call_count,1)
                local = response.json()
                self.assertEqual(local['knowledge_source'],'local')
                self.assertEqual(local['retrieval_status'],'hit')
                self.assertIn('青禾-6284',local['answer'])
                self.assertIn('31',local['answer'])
                self.assertIn({'type':'local','source':'uploads/'+name,'page':2},local['sources'])
                search.assert_not_called()
                report['local'] = local
                report['local_web_search_calls'] = 0

                with self.subTest('live_web_answer'):
                    question = 'In MySQL InnoDB, which SQL command shows the latest detected deadlock details?'
                    response = api.post('/agent',json={'question':question,'session_id':thread})
                    self.assertEqual(response.status_code,200,response.text)
                    web = response.json()
                    report['web']=web
                    self.assertEqual(web['retrieval_status'],'no_hit')
                    self.assertEqual(web['knowledge_source'],'web',json.dumps(web,ensure_ascii=False))
                    self.assertIn('SHOW ENGINE INNODB STATUS',web['answer'].replace('`','').replace('*','').upper())
                    self.assertTrue(web['sources'])
                    self.assertEqual(search.call_count,1)
                    actual_sources={(r['title'],r['url']) for r in searches[-1]['results']}
                    self.assertLessEqual(len(actual_sources),3)
                    for source in web['sources']:
                        self.assertEqual(source['type'],'web')
                        self.assertIn((source['title'],source['url']),actual_sources)
                    report['web']=web

                question = '尚未公开的私有设备 ZQX-'+uuid.uuid4().hex+' 的管理员恢复密钥是什么？'
                response = api.post('/agent',json={'question':question,'session_id':thread})
                self.assertEqual(response.status_code,200,response.text)
                none = response.json()
                self.assertEqual(none['retrieval_status'],'no_hit')
                self.assertEqual(none['knowledge_source'],'none')
                self.assertEqual(none['sources'],[])
                self.assertEqual(none['answer'],'当前没有找到足够可靠的资料。')
                self.assertEqual(search.call_count,2)
                report['none']=none
                report['real_searches']=searches

                # Reuse the same thread: a local hit must not inherit web sources.
                response=api.post('/agent',json={'question':local_question,'session_id':thread})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['knowledge_source'],'local')
                self.assertTrue(all(s['type']=='local' for s in response.json()['sources']))
                self.assertEqual(search.call_count,2)
                report['same_thread_sources_reset']=True

            self.assertEqual(rag.RETRIEVAL_SCORE_THRESHOLD,0.5)
            report['threshold']=0.5
            report['status']='passed' if report.get('web',{}).get('knowledge_source')=='web' else 'partial_web_unavailable'
        finally:
            self.assertEqual(path.resolve().parent,rag.UPLOADS_DIR.resolve())
            path.unlink(missing_ok=True)
            rag.sync_knowledge()
            self.assertEqual(rag.load_state(),before_state)
            self.assert_points_preserved(helpers.point_snapshot(),before_points)
            self.assertEqual(helpers.finetune_manifest(),helpers.FINETUNE_BEFORE)
            report.update(original_knowledge_preserved=True,finetune_unchanged=True,test_upload_cleaned=True)
            (ROOT/'docs'/'web_fallback_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))

    def test_controlled_paths(self):
        report = {}
        # Empty search results bypass the web LLM completely.
        with TestClient(main.app) as api, patch.object(lg,'search_web',return_value=[]) as search, patch.object(type(lg.model),'invoke',side_effect=AssertionError('empty search must not invoke LLM')) as invoke:
            response=api.post('/agent',json={'question':'量子纠缠实验怎样检验贝尔不等式？','session_id':uuid.uuid4().hex})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['knowledge_source'],'none')
            self.assertEqual(response.json()['sources'],[])
            search.assert_called_once()
            invoke.assert_not_called()
            report['empty_results_llm_calls']=0

        # Provider errors, malformed answers and invented URL/ids fail closed.
        import web_search as ws
        with patch.object(ws,'DDGS',side_effect=RuntimeError('intentional provider failure')):
            self.assertEqual(ws.web_search('test'),[])
        state={'question':'test','messages':[ToolMessage(content=json.dumps([{'title':'Real search result','url':'https://example.org/real','snippet':'Some evidence'}]),name='web_search',tool_call_id='test')]}
        for content in ['not JSON',json.dumps({'supported':False,'answer':'guess','used_result_ids':[]}),json.dumps({'supported':True,'answer':'https://invented.invalid/x','used_result_ids':[1]}),json.dumps({'supported':True,'answer':'guess','used_result_ids':[99]})]:
            with patch.object(type(lg.model),'invoke',return_value=AIMessage(content=content)):
                result=lg.chatbot(state)
                self.assertEqual(result['knowledge_source'],'none')
                self.assertEqual(result['sources'],[])
        report['malformed_and_invented_sources_rejected']=True
        with patch.object(type(lg.model),'invoke',return_value=AIMessage(content=json.dumps({'supported':True,'answer':'Some evidence.','used_result_ids':[1]}))):
            result=lg.chatbot(state)
            self.assertEqual(result['knowledge_source'],'web')
            self.assertEqual(result['sources'],[{'type':'web','title':'Real search result','url':'https://example.org/real'}])
        report['controlled_web_metadata_preserved']=True
        report['status']='passed'
        (ROOT/'docs'/'web_fallback_controlled_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        rag.client.close()
