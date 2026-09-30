"""Verify byte-preserved archive payload without resolving or installing it."""
from pathlib import Path
import hashlib
import json


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'SHA256SUMS.json').read_text())
    assert manifest, 'Empty archive manifest'
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root), 'Manifest path escapes archive'
        assert path.is_file() and not (root / relative).is_symlink(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    provenance = json.loads((root / 'capture-provenance.json').read_text())
    for entry in provenance['files']:
        relative = entry['path']
        if relative in manifest:
            assert manifest[relative] == entry['sha256'], relative
    for name, count in [('xtoken-base', 255), ('xtoken-automodel-worker', 273)]:
        snapshot = json.loads((root / 'snapshots' / f'{name}.json').read_text())
        packages = {p['name'].lower(): p['version'] for p in snapshot['packages']}
        assert len(snapshot['packages']) == count
        assert snapshot['python'].startswith('3.13.14 ')
        assert packages['torch'] == '2.11.0+cu130'
    incomplete = json.loads((root / 'snapshots/simct-incomplete.json').read_text())
    assert incomplete['packages'] == []
    print(json.dumps({'verified_files': len(manifest), 'historical_identity': 'PASS',
                      'rebuilt_environment': False, 'real_model_training_complete': False}))


if __name__ == '__main__':
    verify(Path(__file__).parent)
