"""Lightweight importer guard and diagnostic-feature regressions."""
import hashlib
import ast
import binascii
import io
import zipfile
import runpy
import types
from unittest import mock
from pathlib import Path
import tempfile
import unittest

from checkpoint_replay_assets import safe_target, sha, download_stored_member
from checkpoint_replay_worker import repeat_fraction, code_block
from checkpoint_replay_code import extract_code
from degen_checkpoint_assets import checkpoint_groups, REQUIRED


class ReplayTests(unittest.TestCase):
    def test_healthy_terminal_probe_requires_correct_completed_answer(self):
        import json
        fake_sg=types.ModuleType('sglang');fake_tf=types.ModuleType('transformers')
        fake_tf.AutoModelForCausalLM=mock.Mock();fake_tf.AutoTokenizer=mock.Mock()
        with mock.patch.dict('sys.modules',{'torch':types.ModuleType('torch'),'sglang':fake_sg,'transformers':fake_tf}):
            namespace=runpy.run_path(str(Path(__file__).parent/'checkpoint_replay_deep.py'),run_name='spawned_worker')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cases=[{'case':{'id':name},'prompt_ids':[2,3],'output_ids':ids}
                   for name,ids in [('math-apples',[8,9,107]),('math-speed',[8,107]),('math-age',[8,9])]]
            (root/'warmup-responses.json').write_text(json.dumps(cases))
            (root/'warmup-mathverify.json').write_text(json.dumps([{'id':'math-apples','passed':True},
                {'id':'math-speed','passed':False},{'id':'math-age','passed':True}]))
            self.assertEqual(namespace['healthy_terminal_prefixes'](root),[{'case_id':'math-apples','prefix_ids':[2,3,8,9]}])

    def test_cloud_overlay_forwards_importer_and_volume_selection(self):
        # Modal imports the module again inside its worker; client-only env is lost.
        path=Path(__file__).parent/'checkpoint_replay_modal.py'
        tree=ast.parse(path.read_text())
        env=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='env')
        keys={k.value for k in env.args[0].keys}
        self.assertEqual(keys,{'CHECKPOINT_REPLAY_RUN','CHECKPOINT_REPLAY_ASSETS','CHECKPOINT_REPLAY_PREPARE'})

    def test_flat_checkpoint_group_has_one_complete_run(self):
        base='simct-runs/qwen-gemma-mp_opd-trust_r-gpu7-run/checkpoint/step160/'
        infos=[zipfile.ZipInfo(base+name) for name in REQUIRED]
        group=checkpoint_groups(infos)['trust-r-step160']
        self.assertEqual(group['source_directory'],base.rstrip('/'))
        with self.assertRaises(ValueError):checkpoint_groups(infos[:-1])
        with self.assertRaises(ValueError):checkpoint_groups(infos+[infos[0]])
        with self.assertRaises(ValueError):checkpoint_groups(infos+[zipfile.ZipInfo('../secret.json')])

    def test_flat_checkpoint_alias_cannot_mix_two_runs(self):
        bases=['simct-runs/qwen-gemma-mp_opd-align-gpu4-run1/checkpoint/step80/',
               'simct-runs/qwen-gemma-mp_opd-align-gpu4-run2/checkpoint/step80/']
        with self.assertRaises(ValueError):checkpoint_groups([zipfile.ZipInfo(b+'config.json') for b in bases])

    def test_deep_worker_can_be_imported_by_spawn_without_starting_models(self):
        fake_sg=types.ModuleType('sglang');fake_sg.Engine=mock.Mock()
        fake_tf=types.ModuleType('transformers')
        fake_tf.AutoModelForCausalLM=mock.Mock();fake_tf.AutoTokenizer=mock.Mock()
        with mock.patch.dict('sys.modules',{'torch':types.ModuleType('torch'),'sglang':fake_sg,'transformers':fake_tf}):
            namespace=runpy.run_path(str(Path(__file__).parent/'checkpoint_replay_deep.py'),run_name='spawned_worker')
        self.assertTrue(callable(namespace['main']))
        fake_sg.Engine.assert_not_called();fake_tf.AutoModelForCausalLM.assert_not_called()

    def test_archive_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):safe_target(root,'../outside.safetensors')
            self.assertEqual(safe_target(root,'nested/config.json'),root/'nested/config.json')

    def test_streamed_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'data';p.write_bytes(b'checkpoint')
            self.assertEqual(sha(p),hashlib.sha256(b'checkpoint').hexdigest())

    def test_repeated_loop_vs_unique(self):
        self.assertGreater(repeat_fraction([1,2,3]*100),.98)
        self.assertEqual(repeat_fraction(list(range(300))),0)
        self.assertEqual(repeat_fraction([1,2]),0)

    def test_extract_fenced_python(self):
        self.assertEqual(code_block('Explanation\n```python\ndef f():\n    return 1\n```'),
                         'def f():\n    return 1\n')

    def test_author_extraction_matches_actual_evaluator(self):
        source=Path(__file__).resolve().parents[2]/'scripts/evaluation/evaluation.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='_extract_code_block')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        for tag in ('python','python3','python3.8','','py'):
            raw='```'+tag+'\ndef f():\n    return 1\n```'
            self.assertEqual(extract_code(raw,'author'),namespace['_extract_code_block'](raw))

    def test_python_version_fence_has_valid_body(self):
        raw='```python3.8\ndef f():\n    return 1\n```'
        with self.assertRaises(SyntaxError):ast.parse(extract_code(raw,'author'))
        self.assertIsInstance(ast.parse(extract_code(raw,'python-version-aware')),ast.Module)

    def test_resume_stored_zip_without_redownloading_prefix(self):
        payload=b'checkpoint data '*128
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('inner.zip',payload)
        blob=buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:info=z.infolist()[0]
        class Response:
            def __init__(self,start,end):self.content=blob[start:end+1]
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_content(self,size):yield self.content
        calls=[]
        def checked(start,end):calls.append((start,end));return Response(start,end)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inner.zip.range-partial').write_bytes(payload[:500])
            with mock.patch('checkpoint_replay_assets.access.checked_range',side_effect=checked):
                receipt=download_stored_member(info,root)
            self.assertEqual((root/'inner.zip').read_bytes(),payload)
            self.assertEqual(receipt['sha256'],hashlib.sha256(payload).hexdigest())
            self.assertTrue(receipt['zip_crc_pass'])
            self.assertEqual(calls[1][0],30+len('inner.zip')+500)

    def test_bad_partial_crc_cannot_be_promoted_to_final_zip(self):
        payload=b'checkpoint data '*128
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('inner.zip',payload)
        blob=buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:info=z.infolist()[0]
        class Response:
            def __init__(self,start,end):self.content=blob[start:end+1]
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_content(self,size):yield self.content
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'inner.zip.range-partial').write_bytes(b'wrong')
            with mock.patch('checkpoint_replay_assets.access.checked_range',side_effect=lambda a,b:Response(a,b)):
                with self.assertRaises(RuntimeError):download_stored_member(info,root)
            self.assertFalse((root/'inner.zip').exists())


if __name__=='__main__':unittest.main()
