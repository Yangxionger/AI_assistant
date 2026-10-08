"""Live text-PDF acceptance and failure rollback checks; no OCR/mocked retrieval."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import test_knowledge_upload as helpers
from pdf_fixtures import pdf_bytes, TEXT_SPECS
from fastapi.testclient import TestClient

rag = helpers.rag
main = helpers.main
ROOT = helpers.ROOT


class PDFUploadAcceptance(unittest.TestCase):
    assert_points_preserved = helpers.KnowledgeUploadAcceptance.assert_points_preserved

    def test_text_pdf_repeat_errors_update_and_rollback(self):
        original_state = rag.load_state()
        original_points = helpers.point_snapshot()
        tag = uuid.uuid4().hex[:12]
        name = f'pdf_smoke_{tag}.pdf'
        path = rag.UPLOADS_DIR/name
        source = 'uploads/'+name
        self.assertFalse(path.exists())
        invalid_cases = [
            ('damaged', b'%PDF-1.7\nbroken', 'invalid_pdf'),
            ('zero_bytes', b'', 'empty_pdf'),
            ('zero_pages', pdf_bytes([]), 'empty_pdf'),
            ('blank', pdf_bytes([None]), 'no_extractable_text'),
            ('image_only', pdf_bytes([{'image':True}]), 'scanned_pdf'),
            ('mostly_images', pdf_bytes([TEXT_SPECS[0],{'image':True,'text':'2'},{'image':True,'text':'3'}]), 'scanned_pdf'),
            ('encrypted', pdf_bytes([TEXT_SPECS[0]], password='fixture-password'), 'encrypted_pdf')
        ]
        invalid_paths = [rag.UPLOADS_DIR/f'pdf_{tag}_{label}.pdf' for label,_,_ in invalid_cases]
        report = {'original_point_count':len(original_points), 'pdf_parser':'pypdf', 'original_sources':sorted(original_state)}
        content = pdf_bytes(TEXT_SPECS)
        fixtures = ROOT/'tests'/'fixtures'
        fixtures.mkdir(exist_ok=True)
        (fixtures/'text_learning.pdf').write_bytes(content)
        (fixtures/'image_only.pdf').write_bytes(dict((label,body) for label,body,_ in invalid_cases)['image_only'])
        try:
            with TestClient(main.app) as api, patch.object(rag.embedding_model,'encode',wraps=rag.embedding_model.encode) as encode, \
                    patch.object(rag,'extract_pdf_pages',wraps=rag.extract_pdf_pages) as extract:
                uploaded = api.post('/knowledge/upload', files={'file':(name,content,'application/pdf')})
                self.assertEqual(uploaded.status_code, 200, uploaded.text)
                self.assertEqual(uploaded.json()['status'], 'indexed')
                self.assertEqual(uploaded.json()['pages'], 3)
                self.assertEqual(uploaded.json()['text_pages'], 2)
                self.assertEqual(uploaded.json()['skipped_blank_pages'], [2])
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(extract.call_count, 1)
                source_points = helpers.point_snapshot(source)
                self.assertEqual(len(source_points), uploaded.json()['chunks'])
                self.assertEqual(len(helpers.points()), len(original_points)+uploaded.json()['chunks'])
                self.assertEqual({p['payload']['page'] for p in source_points.values()}, {1,3})
                self.assertEqual({p['payload']['chunk_id'] for p in source_points.values()}, set(range(len(source_points))))
                self.assertTrue(all({'text','source','chunk_id','page'} <= p['payload'].keys() for p in source_points.values()))
                self.assertTrue(all('page' not in p['payload'] for p in original_points.values()))
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assertIn(source, api.get('/knowledge/sources').json()['sources'])
                self.assertEqual(helpers.langgraph_agent.list_knowledge_sources.invoke({}), rag.list_knowledge_sources())
                before_sync_calls = encode.call_count
                rag.sync_knowledge()
                self.assertEqual(encode.call_count, before_sync_calls)
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['upload'] = uploaded.json()
                report['pdf_bytes'] = len(content)
                report['points_after_upload'] = len(helpers.points())
                report['payload_pages'] = sorted({p['payload']['page'] for p in source_points.values()})

                asked = api.post('/ask', json={'question':'星砂学习同步协议的刷新口令是什么？缓存有效期多少分钟？','level':'beginner'})
                self.assertEqual(asked.status_code, 200, asked.text)
                for value in ['星砂-5831','43',source,'第1页']:
                    self.assertIn(value, asked.json()['retrieved_context'])
                self.assertIn('星砂-5831', asked.json()['answer'])
                self.assertIn('43', asked.json()['answer'])
                report['ask'] = asked.json()
                report['ask_used_real_llm'] = True

                calls = encode.call_count
                parses = extract.call_count
                repeated = api.post('/knowledge/upload', files={'file':(name,content,'application/pdf')})
                self.assertEqual(repeated.status_code, 200, repeated.text)
                self.assertEqual(repeated.json()['status'], 'unchanged')
                self.assertEqual(encode.call_count, calls)
                self.assertEqual(extract.call_count, parses)
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['repeat'] = repeated.json()
                report['repeat_embedding_calls'] = encode.call_count-calls
                report['repeat_text_extraction_calls'] = extract.call_count-parses
                report['points_after_repeat'] = len(helpers.points())

                report['invalid_pdf_checks'] = []
                for (label,body,code), invalid_path in zip(invalid_cases, invalid_paths):
                    response = api.post('/knowledge/upload', files={'file':(invalid_path.name,body,'application/pdf')})
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(response.json()['detail']['code'], code)
                    self.assertFalse(invalid_path.exists())
                    self.assertNotIn('uploads/'+invalid_path.name, rag.load_state())
                    report['invalid_pdf_checks'].append({'case':label,'status':400,'detail':response.json()['detail']})
                bad_update = api.post('/knowledge/upload', files={'file':(name,invalid_cases[4][1],'application/pdf')})
                self.assertEqual(bad_update.status_code, 400)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['invalid_update_preserved_file_points_state'] = True

                changed = pdf_bytes([{'text':'星砂学习同步协议更新：刷新口令改为星砂-9642，缓存有效期为17分钟。'}])
                with patch.object(rag.embedding_model,'encode',side_effect=RuntimeError('intentional PDF embedding failure')):
                    failed = api.post('/knowledge/upload', files={'file':(name,changed,'application/pdf')})
                self.assertEqual(failed.status_code, 500)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['embedding_failure_rollback_passed'] = True

                real_upsert = rag.client.upsert
                upsert_calls = []
                def partial_fail_once(*args, **kwargs):
                    upsert_calls.append(True)
                    if len(upsert_calls) == 1:
                        real_upsert(*args, **{**kwargs,'points':kwargs['points'][:1]})
                        raise RuntimeError('intentional partial PDF upsert failure')
                    return real_upsert(*args, **kwargs)
                with patch.object(rag.client,'upsert',side_effect=partial_fail_once):
                    failed = api.post('/knowledge/upload', files={'file':(name,changed,'application/pdf')})
                self.assertEqual(failed.status_code, 500)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['partial_upsert_rollback_passed'] = True

                real_save_state = rag.save_state
                save_calls = []
                def save_fail_once(state):
                    save_calls.append(True)
                    if len(save_calls) == 1:
                        raise RuntimeError('intentional PDF state commit failure')
                    return real_save_state(state)
                with patch.object(rag,'save_state',side_effect=save_fail_once):
                    failed = api.post('/knowledge/upload', files={'file':(name,changed,'application/pdf')})
                self.assertEqual(failed.status_code, 500)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(rag.load_state()[source], hashlib.sha256(content).hexdigest())
                self.assert_points_preserved(helpers.point_snapshot(source), source_points)
                report['state_commit_rollback_passed'] = True

                updated = api.post('/knowledge/upload', files={'file':(name,changed,'application/pdf')})
                self.assertEqual(updated.status_code, 200, updated.text)
                self.assertEqual(updated.json()['status'], 'updated')
                self.assertEqual(updated.json()['pages'], 1)
                self.assertEqual(updated.json()['chunks'], 1)
                self.assertEqual(path.read_bytes(), changed)
                self.assertEqual(len(helpers.points(source)), 1)
                self.assertTrue(all('星砂-5831' not in p.payload['text'] for p in helpers.points(source)))
                self.assertFalse(set(source_points) & set(helpers.point_snapshot(source)))
                report['update'] = updated.json()
                report['points_after_update'] = len(helpers.points())

                # A bad manually placed PDF must not abort sync or advance its hash.
                invalid_paths[0].write_bytes(invalid_cases[0][1])
                rag.sync_knowledge()
                self.assertNotIn('uploads/'+invalid_paths[0].name, rag.load_state())
                self.assertEqual(len(helpers.points(source)), 1)
                report['invalid_manual_pdf_does_not_abort_sync'] = True
                report['status'] = 'passed'
        finally:
            for owned in [path]+invalid_paths:
                self.assertEqual(owned.resolve().parent, rag.UPLOADS_DIR.resolve())
                owned.unlink(missing_ok=True)
            rag.sync_knowledge()
            self.assertEqual(rag.load_state(), original_state)
            self.assert_points_preserved(helpers.point_snapshot(), original_points)
            self.assertEqual(helpers.finetune_manifest(), helpers.FINETUNE_BEFORE)
            self.assertFalse(list(rag.UPLOADS_DIR.glob('.upload-*.tmp')))
            self.assertFalse(list(rag.UPLOADS_DIR.glob('.previous-*.tmp')))
            report.update({'test_uploads_cleaned_up':True,'original_knowledge_preserved':True,
                           'finetune_file_sizes_and_mtimes_unchanged':True})
            (ROOT/'docs'/'pdf_upload_test_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    try:
        unittest.main(verbosity=2)
    finally:
        rag.client.close()
