"""Freeze 32 public Formal Logic rows; this is not a paper-size dataset."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
dataset = 'nvidia/Nemotron-Pretraining-Specialized-v1.1'
config = 'Nemotron-Pretraining-Formal-Logic'
url = 'https://datasets-server.huggingface.co/rows?' + urlencode(
    dict(dataset=dataset, config=config, split='train', offset=0, length=32))
with urlopen(url, timeout=60) as response:
    raw = response.read()
    revision = response.headers.get('X-Revision')
payload = json.loads(raw)
assert revision == '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af', 'Dataset Viewer revision changed'
assert len(payload['rows']) == 32
assert [r['row_idx'] for r in payload['rows']] == list(range(32))
assert all(not r.get('truncated_cells') for r in payload['rows'])
rows = [r['row'] for r in payload['rows']]
assert all(isinstance(r['text'], str) and r['text'] for r in rows)
(out/'viewer-response.json').write_bytes(raw)
pq.write_table(pa.Table.from_pylist(rows), out/'formal-logic-32.parquet')
report = dict(dataset=dataset, config=config, split='train', offset=0, rows=32,
              viewer_revision=revision, fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
              response_sha256=hashlib.sha256(raw).hexdigest(),
              parquet_sha256=hashlib.sha256((out/'formal-logic-32.parquet').read_bytes()).hexdigest(),
              scientific_scope='engineering smoke subset, not paper reproduction')
assert report['parquet_sha256'] == '19f84245ef610a9148092e7c6d91969e6fa8ae20e7ed4c477265c75f456ce771', 'Frozen dataset content changed'
(out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
