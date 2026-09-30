"""Read-only availability check for exact locked worker wheel (no download)."""
import datetime as dt
import json
from pathlib import Path
import tomllib
import urllib.request
import urllib.error

root = Path('/workspace/xtoken-native')
data = tomllib.loads((root / 'NeMo-RL/pyproject.toml').read_text())
spec = next(s for s in data['project']['optional-dependencies']['automodel']
            if s.startswith('flash-attn @ ') and 'linux_x86_64.whl' in s)
url = spec.split(' @ ', 1)[1].split(' ; ', 1)[0]
report = {'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
          'scope': 'HEAD availability only, not ABI or CUDA execution',
          'source': 'exact automodel extra in upstream pyproject.toml',
          'wheel_name': url.rsplit('/', 1)[1], 'attempts': 1}
try:
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        report.update(status=response.status, available=response.status == 200)
except urllib.error.HTTPError as exc:
    report.update(status=exc.code, available=False, error_type=type(exc).__name__)
except Exception as exc:
    report.update(status=None, available=None, error_type=type(exc).__name__)
(root/'artifacts/worker-wheel-preflight.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
