"""Fetch a bounded public dataset slice, freeze it, and build upstream projection."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, '/workspace/simct-xtoken-harness/experiments/xtoken/scripts')
from run_logged import run_logged, workload_environment

ROOT = Path('/workspace/xtoken-native')
OUT = ROOT/'smoke-20260903-r1'
OUT.mkdir(exist_ok=True)
env = workload_environment()
env.update(HF_HOME=str(ROOT/'hf-cache'), HF_HUB_OFFLINE='1', WANDB_MODE='disabled',
           TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(ROOT/'NeMo-RL'),
           UV_CACHE_DIR=str(ROOT/'uv-cache'), UV_NO_CACHE='false')
dataset = 'nvidia/Nemotron-Pretraining-Specialized-v1.1'
def get(endpoint, **params):
    url = 'https://datasets-server.huggingface.co/'+endpoint+'?'+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)
splits = get('splits', dataset=dataset)
(OUT/'dataset_splits.json').write_text(json.dumps(splits, indent=2)+'\n')
matching = [s for s in splits['splits'] if 'formal' in s['config'].lower() and s['split']=='train']
assert len(matching)==1, 'Expected unique formal-logic train config'
config = matching[0]['config']
rows = get('rows', dataset=dataset, config=config, split='train', offset=0, length=32)
(OUT/'dataset_rows.json').write_text(json.dumps(rows, ensure_ascii=False)+'\n')
import pyarrow as pa
import pyarrow.parquet as pq
texts = [r['row']['text'] for r in rows['rows']]
assert len(texts)==32 and all(isinstance(t,str) and t for t in texts)
data_path = OUT/'formal_logic_first32.parquet'
pq.write_table(pa.table({'text':texts}), data_path)
(OUT/'dataset_manifest.json').write_text(json.dumps(dict(
    dataset=dataset, config=config, split='train', offset=0, rows=32,
    parquet_path=str(data_path), sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
    provenance='Dataset Viewer current snapshot frozen by local content hash; not full paper dataset'),indent=2)+'\n')
print('DATASET_32_ROWS_FROZEN', flush=True)
student=ROOT/'hf-cache/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
teacher=ROOT/'hf-cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
cmd=[str(ROOT/'NeMo-RL/.venv/bin/python'), '-m', 'tools.x_token.minimal_projection_via_multitoken',
     '--student-model', str(student), '--teacher-model', str(teacher), '--top-k','4',
     '--enable-special-token-mapping','--enable-exact-match','--disable-reverse-pass',
     '--disable-scale-trick','--output-filename','xtoken_l1_smoke','--output-dir',str(OUT/'projection')]
rc, _=run_logged(cmd,cwd=ROOT/'NeMo-RL',root=OUT,name='projection-exact-pair',timeout=600,env=env)
sys.exit(rc)
