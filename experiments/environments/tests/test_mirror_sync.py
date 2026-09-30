import importlib.util
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from urllib.parse import urlparse

SCRIPT = Path(__file__).resolve().parents[1]/'mirror_sync.py'
spec = importlib.util.spec_from_file_location('mirror_sync', SCRIPT)
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)
ORIGINAL = (SCRIPT.parent/'xtoken/uv.lock').read_text()
PACKAGES = {p['name']: p for p in tomllib.loads(ORIGINAL)['package']}


def index(name):
    wheel = PACKAGES[name]['wheels'][0]
    url = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web' + urlparse(wheel['url']).path
    return '<a href="' + url + '#' + wheel['hash'].replace(':', '=', 1) + '">wheel</a>'


class MirrorSync(unittest.TestCase):
    def test_only_two_urls_change_all_versions_hashes_and_torch_unchanged(self):
        rendered, mappings = mirror.transport_lock(ORIGINAL, fetch=index)
        self.assertEqual(len(mappings), 2)
        normalized = rendered
        for mapping in mappings:
            normalized = normalized.replace(mapping['mirror_url'], mapping['original_url'])
        self.assertEqual(normalized, ORIGINAL)
        packages = {p['name']: p for p in tomllib.loads(rendered)['package']}
        self.assertEqual(packages['torch'], PACKAGES['torch'])
        self.assertEqual(packages['torchvision'], PACKAGES['torchvision'])

    def test_wrong_hash_is_rejected(self):
        with self.assertRaises(AssertionError):
            mirror.transport_lock(ORIGINAL, fetch=lambda n: index(n).replace('sha256=', 'sha512='))

    def test_foreign_host_is_rejected(self):
        with self.assertRaises(AssertionError):
            mirror.transport_lock(ORIGINAL, fetch=lambda n: index(n).replace(
                'mirrors.tuna.tsinghua.edu.cn', 'untrusted.example'))

    def test_absent_wheel_is_rejected_without_fallback(self):
        with self.assertRaises(AssertionError):
            mirror.transport_lock(ORIGINAL, fetch=lambda n: '')

    def test_credentials_and_http_are_rejected(self):
        for prefix in ('http://', 'https://user:password@'):
            with self.subTest(prefix=prefix), self.assertRaises(AssertionError):
                mirror.transport_lock(ORIGINAL, fetch=lambda n: index(n).replace('https://', prefix))

    def test_no_source_or_package_selection_drift(self):
        altered = ORIGINAL.replace('name = "nvidia-cublas-cu12"\nversion',
                                   'name = "wrong-cublas"\nversion')
        with self.assertRaises(KeyError):
            mirror.transport_lock(altered, fetch=index)

    def test_private_view_sync_preserves_canonical_and_child_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root/'canonical'
            project.mkdir()
            original = {'uv.lock': ORIGINAL, 'pyproject.toml': '[project]\nname="fixture"\n',
                        '.python-version': '3.13.15\n'}
            for name, data in original.items():
                (project/name).write_text(data)
            rendered, mappings = mirror.transport_lock(ORIGINAL, fetch=index)
            with patch.object(mirror, 'transport_lock', return_value=(rendered, mappings)), \
                    patch.object(mirror.subprocess, 'run') as run, \
                    patch.dict(os.environ, {'UV_PROJECT_ENVIRONMENT': str(root/'xtoken/venv')}):
                run.return_value.returncode = 7
                self.assertEqual(mirror.run(root, project), 7)
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ['uv', 'sync', '--frozen', '--project'])
            self.assertEqual(run.call_count, 1)
            for name, data in original.items():
                self.assertEqual((project/name).read_text(), data)
            manifest = next((root/'mirror-installs').glob('*/result.json'))
            result = json.loads(manifest.read_text())
            self.assertTrue(result['dependency_graph_identical'])
            self.assertTrue(result['canonical_unchanged_after_sync'])
            self.assertFalse(result['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
