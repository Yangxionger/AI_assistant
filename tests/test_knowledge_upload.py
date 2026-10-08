"""Live API/RAG acceptance test. Requires the project's existing DeepSeek key."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def finetune_manifest():
    return {str(p.relative_to(ROOT/'finetune')):(p.stat().st_size,p.stat().st_mtime_ns)
            for p in (ROOT/'finetune').rglob('*') if p.is_file()}


FINETUNE_BEFORE = finetune_manifest()
from fastapi.testclient import TestClient
import main
import rag
import agent
import langgraph_agent


def source_filter(source):
    return rag.Filter(must=[rag.FieldCondition(key='source', match=rag.MatchValue(value=source))])


def points(source=None):
    result = []
    offset = None
    while True:
        page, offset = rag.client.scroll(collection_name='ai_knowledge', limit=128, offset=offset,
                                        scroll_filter=source_filter(source) if source else None,
                                        with_payload=True, with_vectors=True)
        result.extend(page)
        if offset is None:
            return sorted(result, key=lambda p:str(p.id))


def point_snapshot(source=None):
    return {str(p.id):{'payload':p.payload,'vector':p.vector} for p in points(source)}


class KnowledgeUploadAcceptance(unittest.TestCase):
    def assert_points_preserved(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for point_id in expected:
            self.assertEqual(actual[point_id]['payload'], expected[point_id]['payload'])
            # Local Qdrant cosine queries normalize loaded vectors in place.
            current = np.asarray(actual[point_id]['vector'], dtype=np.float64)
            previous = np.asarray(expected[point_id]['vector'], dtype=np.float64)
            current /= np.linalg.norm(current) or 1
            previous /= np.linalg.norm(previous) or 1
            np.testing.assert_allclose(current, previous, rtol=1e-5, atol=1e-6)

    def test_upload_retrieve_repeat_update_and_validation(self):
        original_state = rag.load_state()
        original_points = point_snapshot()
        tag = uuid.uuid4().hex[:12]
        txt_name = f'upload_smoke_{tag}.txt'
        md_name = f'上传_smoke_{tag}.md'
        txt_path = rag.UPLOADS_DIR/txt_name
        md_path = rag.UPLOADS_DIR/md_name
        owned_files = [txt_path, md_path]
        self.assertTrue(all(not p.exists() for p in owned_files))
        text = ('玄鸟学习缓存协议是一份计算机学习资料。\n'
                '玄鸟学习缓存协议的刷新口令是“星河-7426”，缓存有效期为37分钟。\n'
                '缓存过期后，客户端先调用flush_learning_cache，再重新加载课程索引。\n'
                + '缓存过期后先清理课程索引，再重新读取学习资料。\n'*12)
        content = text.encode('utf-8')
        source = 'uploads/'+txt_name
        report = {'original_point_count':len(original_points), 'original_sources':sorted(original_state)}
        try:
            with TestClient(main.app) as api, patch.object(rag.embedding_model,'encode',wraps=rag.embedding_model.encode) as encode:
                uploaded = api.post('/knowledge/upload', files={'file':(txt_name,content,'text/plain')})
                self.assertEqual(uploaded.status_code, 200, uploaded.text)
                self.assertEqual(uploaded.json()['status'], 'indexed')
                self.assertEqual(uploaded.json()['source'], source)
                self.assertEqual(txt_path.read_bytes(), content)
                expected_chunks = len(rag.chunk_text(text))
                source_points = point_snapshot(source)
                self.assertEqual(len(source_points), expected_chunks)
                self.assertEqual(len(points()), len(original_points)+expected_chunks)
                self.assertEqual(set(source_points), {rag.make_point_id(d) for d in rag.load_file_documents(txt_path)})
                self.assertTrue(all({'text','source','chunk_id'} <= p['payload'].keys() for p in source_points.values()))
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assertIn(source, api.get('/knowledge/sources').json()['sources'])
                calls_before_sync = encode.call_count
                rag.sync_knowledge()
                self.assertEqual(encode.call_count, calls_before_sync)
                self.assert_points_preserved(point_snapshot(source), source_points)
                report['txt_upload'] = uploaded.json()
                report['points_after_txt'] = len(points())

                asked = api.post('/ask', json={'question':'玄鸟学习缓存协议的刷新口令是什么？缓存有效期多少分钟？','level':'beginner'})
                self.assertEqual(asked.status_code, 200, asked.text)
                self.assertIn('星河-7426', asked.json()['retrieved_context'])
                self.assertIn('37', asked.json()['retrieved_context'])
                self.assertIn('星河-7426', asked.json()['answer'])
                self.assertIn('37', asked.json()['answer'])
                report['ask'] = asked.json()
                report['ask_used_real_llm'] = True

                calls_before_repeat = encode.call_count
                repeated = api.post('/knowledge/upload', files={'file':(txt_name,content,'text/plain')})
                self.assertEqual(repeated.status_code, 200, repeated.text)
                self.assertEqual(repeated.json()['status'], 'unchanged')
                self.assertEqual(encode.call_count, calls_before_repeat)
                self.assert_points_preserved(point_snapshot(source), source_points)
                self.assertEqual(len(points()), len(original_points)+expected_chunks)
                report['repeat_upload'] = repeated.json()
                report['repeat_embedding_calls'] = encode.call_count-calls_before_repeat
                report['points_after_repeat'] = len(points())

                changed = '更新后的玄鸟学习缓存协议：刷新口令改为星河-9642，缓存有效期为19分钟。'.encode('utf-8')
                with patch.object(rag.embedding_model,'encode',side_effect=RuntimeError('intentional embedding failure')):
                    failed = api.post('/knowledge/upload', files={'file':(txt_name,changed,'text/plain')})
                self.assertEqual(failed.status_code, 500)
                self.assert_points_preserved(point_snapshot(source), source_points)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                updated = api.post('/knowledge/upload', files={'file':(txt_name,changed,'text/plain')})
                self.assertEqual(updated.status_code, 200, updated.text)
                self.assertEqual(updated.json()['status'], 'updated')
                self.assertEqual(len(points(source)), 1)
                self.assertEqual(len(points()), len(original_points)+1)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(changed).hexdigest())
                report['embedding_failure_preserved_old_points_and_state'] = True
                report['update_after_retry'] = updated.json()
                report['points_after_update'] = len(points())

                markdown = '# 上传测试\n\n```python\nprint("Markdown 已入库")\n```\n\n公式：$a^2+b^2=c^2$。\n'
                md_uploaded = api.post('/knowledge/upload', files={'file':(md_name,markdown.encode('utf-8'),'text/markdown')})
                self.assertEqual(md_uploaded.status_code, 200, md_uploaded.text)
                self.assertEqual(md_path.read_text(encoding='utf-8'), markdown)
                self.assertIn('uploads/'+md_name, rag.get_current_state())
                rag.sync_knowledge()
                sources = api.get('/knowledge/sources').json()['sources']
                self.assertIn('uploads/'+md_name, sources)
                self.assertEqual(langgraph_agent.list_knowledge_sources.invoke({}), sources)
                self.assertEqual(agent.list_knowledge_source(), sources)
                self.assertTrue(any('```python' in p.payload['text'] for p in points('uploads/'+md_name)))
                report['markdown_upload'] = md_uploaded.json()
                report['agent_source_lists_match_api'] = True

                invalid = [
                    ('bad.pdf',b'%PDF',400), ('../bad.txt',b'test',400), ('CON.txt',b'test',400),
                    ('empty.txt',b' \n',400), ('invalid.txt',b'\xff\xfe',400), ('nul.txt',b'a\x00b',400),
                    ('large.txt',b'x'*(main.MAX_UPLOAD_BYTES+1),413)
                ]
                before_invalid = point_snapshot()
                for filename, body, status in invalid:
                    response = api.post('/knowledge/upload', files={'file':(filename,body,'application/octet-stream')})
                    self.assertEqual(response.status_code, status, response.text)
                self.assert_points_preserved(point_snapshot(), before_invalid)
                report['invalid_upload_checks'] = [{'filename':filename,'status':status} for filename,_,status in invalid]
                report['status'] = 'passed'
        finally:
            for path in owned_files:
                self.assertEqual(path.resolve().parent, rag.UPLOADS_DIR.resolve())
                path.unlink(missing_ok=True)
            rag.sync_knowledge()
            self.assertEqual(rag.load_state(), original_state)
            self.assert_points_preserved(point_snapshot(), original_points)
            self.assertEqual(finetune_manifest(), FINETUNE_BEFORE)
            report['test_files_cleaned_up'] = True
            report['original_knowledge_preserved'] = True
            report['finetune_file_sizes_and_mtimes_unchanged'] = True
            (ROOT/'docs').mkdir(exist_ok=True)
            (ROOT/'docs'/'knowledge_upload_v1_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        rag.client.close()
