"""Package existing diagnostics only. Does not retry tests or use credentials."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/workspace/xtoken-native')
artifacts = root/'artifacts'
access_path = artifacts/'hf-access.json'
access = json.loads(access_path.read_text())
for row in access['models']:
    log = artifacts/('hf-'+row['model'].replace('/', '--')+'-access.log')
    if log.exists() and 'This repository requires approval' in log.read_text():
        row['gated_denied'] = True
        row['diagnosis'] = 'Hugging Face CLI: repository requires approval'
access['local_log_diagnosis_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
access_path.write_text(json.dumps(access, indent=2)+'\n')
backend = {'automodel_installed': importlib.util.find_spec('nemo_automodel') is not None,
           'scope': 'base environment only; optional Automodel extra not installed',
           'end_to_end_worker_validated': False,
           'reason_not_training': 'Llama gated access denied; no substitute model authorized'}
(artifacts/'backend-status.json').write_text(json.dumps(backend, indent=2)+'\n')
freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(root/'NeMo-RL/.venv/bin/python')], text=True)
(artifacts/'packages.txt').write_text(freeze)
result_paths = sorted((root/'NeMo-RL/tests/unit/unit_results').glob('*.json'))
unit_out = artifacts/'upstream-unit-results'
unit_out.mkdir(exist_ok=True)
for source in result_paths:
    (unit_out/source.name).write_bytes(source.read_bytes())
files = [p for p in artifacts.rglob('*') if p.is_file() and p.suffix != '.pt']
hashes = {str(p.relative_to(artifacts)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in files if p.name != 'sha256.json'}
(artifacts/'sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
with tarfile.open(root/'evidence-diagnostics.tar.gz', 'w:gz') as tar:
    for file in artifacts.rglob('*'):
        if file.is_file() and file.suffix != '.pt':
            tar.add(file, arcname=file.relative_to(root))
print(json.dumps({'archive': 'evidence-diagnostics.tar.gz',
                  'bytes': (root/'evidence-diagnostics.tar.gz').stat().st_size,
                  'included_heavy_models': False, 'backend': backend}, indent=2))
