import ast
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import subprocess
import sys
import runpy
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_source', ROOT/'prepare_source.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class NativePackagingTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_install_deadline_is_explicit_and_sync_only(self):
        script = ROOT/'run_phase.py'
        for phase, deadline in [('sync', '0'), ('canary', '5400')]:
            result = subprocess.run(
                [sys.executable, str(script), 'xtoken', phase,
                 '--root', '/tmp/unused-xtoken-test', '--sync-timeout-seconds', deadline],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
        self.assertIn("timeout=args.sync_timeout_seconds if args.phase=='sync' else 600",
                      script.read_text())

    def test_launcher_regressions(self):
        script = (ROOT/'xtoken_smoke.py').read_text()
        self.assertIn("'--project',str(HERE/'xtoken')", script)
        self.assertIn('cwd=source', script)
        for owner in ('policy', 'teachers.0'):
            for field in ('force_hf=true', 'attn_implementation=sdpa'):
                self.assertIn(f'+{owner}.dtensor_cfg.automodel_kwargs.{field}', script)
        self.assertIn("validated.get('config_digest') == config_digest", script)

    def test_http_timeout_is_explicit_positive_and_sync_only(self):
        script = ROOT/'run_phase.py'
        for phase, deadline in [('sync', '0'), ('sync', '-1'), ('canary', '300')]:
            result = subprocess.run(
                [sys.executable, str(script), 'xtoken', phase,
                 '--root', '/tmp/unused-xtoken-test', '--sync-http-timeout-seconds', deadline],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
        self.assertIn("f'UV_HTTP_TIMEOUT={args.sync_http_timeout_seconds}'", script.read_text())

    def test_sync_passes_and_records_authorized_http_timeout(self):
        sys.path.insert(0, str(ROOT.parent/'xtoken/scripts'))
        import run_logged
        with tempfile.TemporaryDirectory() as tmp:
            argv = [str(ROOT/'run_phase.py'), 'xtoken', 'sync', '--root', tmp,
                    '--sync-timeout-seconds', '5400', '--sync-http-timeout-seconds', '300']
            with patch.object(sys, 'argv', argv), \
                 patch('subprocess.run', return_value=subprocess.CompletedProcess(
                     ['uv', '--version'], 0, stdout='uv 0.12.7\n')), \
                 patch.object(run_logged, 'workload_environment', return_value={'UV_HTTP_TIMEOUT': '999'}), \
                 patch.object(run_logged, 'run_logged', return_value=(0, Path(tmp))) as logged:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(ROOT/'run_phase.py'), run_name='__main__')
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(logged.call_args.args[0],
                             ['env', 'UV_HTTP_TIMEOUT=300', 'uv', 'sync', '--locked'])
            self.assertEqual(logged.call_args.kwargs['timeout'], 5400)
            self.assertNotIn('UV_HTTP_TIMEOUT', logged.call_args.kwargs['env'])

    def test_cuda_workaround_is_opt_in_and_recorded(self):
        script = (ROOT/'xtoken_smoke.py').read_text()
        self.assertIn("'--nccl-cumem-host-fallback', action='store_true'", script)
        self.assertIn("env['NCCL_CUMEM_HOST_ENABLE'] = '0'", script)
        self.assertIn('cuda_environment=cuda_environment', script)
        self.assertIn('shutil.disk_usage(root).free < 2 * 1024**3', script)

    def test_diagnostic_regressions(self):
        probe = (ROOT/'xtoken_cuda_probe.py').read_text()
        self.assertIn("torch.autocast('cuda', dtype=torch.bfloat16)", probe)
        self.assertIn("stage='method_before_offload'", probe)
        self.assertIn("ACTOR_ENVIRONMENT_REGISTRY['cuda_probe_worker.BufferProbeWorker']", probe)
        runner = (ROOT/'run_cuda_probe.py').read_text()
        self.assertIn("marker.open('x')", runner)
        self.assertIn("'--locked', '--no-sync'", runner)
        self.assertIn('timeout=240', runner)

    def test_lock_identity(self):
        expected = {'xtoken': ('2.11.0+cu128', '5.12.1'), 'simct': ('2.9.1+cu128', '4.57.1')}
        for name, (torch, transformers) in expected.items():
            lock = tomllib.loads((ROOT/name/'uv.lock').read_text())
            packages = {p['name']: p for p in lock['package']}
            self.assertEqual(packages['torch']['version'], torch)
            self.assertEqual(packages['transformers']['version'], transformers)
            self.assertFalse(any(p.endswith('-cu13') for p in packages))

    def test_archive_pinned(self):
        lock = tomllib.loads((ROOT/'xtoken/uv.lock').read_text())
        package = next(p for p in lock['package'] if p['name'] == 'nemo-rl')
        self.assertIn('13a10647ebbf0f940d2b06ea41800b3f2fb46099', package['source']['url'])
        self.assertEqual(package['sdist']['hash'], 'sha256:4d8811064e04ccc90740eae8f2dd240ea501b0964f1eb559b9236a4491813fd0')

    def test_flash_attention_official_hash(self):
        lock = tomllib.loads((ROOT/'simct/uv.lock').read_text())
        package = next(p for p in lock['package'] if p['name'] == 'flash-attn')
        self.assertEqual(package['wheels'][0]['hash'], 'sha256:4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810')

    def test_source_verify_rejects_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            archive = temp/'source.tar.gz'
            payload = b'upstream source\n'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('source/module.py')
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            tree = temp/'tree'
            tree.mkdir()
            (tree/'module.py').write_bytes(payload)
            self.assertEqual(source.verify_tree(archive, tree)['files_verified'], 1)
            (tree/'module.py').write_text('changed')
            with self.assertRaises(ValueError):
                source.verify_tree(archive, tree)

    def test_projects_independent(self):
        simct = tomllib.loads((ROOT/'simct/pyproject.toml').read_text())
        xtoken = tomllib.loads((ROOT/'xtoken/pyproject.toml').read_text())
        self.assertEqual(simct['tool']['uv']['sources']['kdflow']['path'], '../../..')
        self.assertNotIn('kdflow', xtoken['tool']['uv']['sources'])
        self.assertFalse(xtoken['tool']['uv']['package'])
        self.assertEqual(len(xtoken['tool']['uv']['exclude-dependencies']), 5)


if __name__ == '__main__':
    unittest.main()
