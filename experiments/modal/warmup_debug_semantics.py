"""Execute generated fixture code in CPU-only, network-blocked Modal sandboxes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time

import modal

from warmup_phi_qwen_debug import IMAGE, PY

app = modal.App('phi-qwen-warmup-code-tests-20261011')
image = modal.Image.from_registry(IMAGE).entrypoint([])
RUNNER = '''import json,resource,signal,sys
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
signal.alarm(5)
payload=json.loads(sys.argv[1])
scope={}
exec(compile(payload['code'],'generated.py','exec'),scope)
exec(compile(payload['tests'],'fixture_tests.py','exec'),scope)
print('FIXTURE_ASSERTIONS_PASSED')
'''


def evaluate(record):
    arm, sample = record
    blocks = re.findall(r'```(?:python|py|python3)?\s*\n(.*?)```', sample['text'], re.S)
    code = blocks[0] if blocks else sample['text']
    payload = json.dumps({'code': code, 'tests': sample['case']['tests']})
    start = time.monotonic()
    sandbox = modal.Sandbox.create(PY, '-I', '-c', RUNNER, payload,
        app=app, image=image, cpu=.25, memory=512, timeout=25,
        block_network=True, secrets=[], volumes={}, include_oidc_identity_token=False)
    try:
        sandbox.wait()
        stdout, stderr = sandbox.stdout.read(), sandbox.stderr.read()
        return {'arm': arm, 'case_id': sample['case']['id'], 'sandbox_id': sandbox.object_id,
            'exit_code': sandbox.returncode,
            'pass': sandbox.returncode == 0 and 'FIXTURE_ASSERTIONS_PASSED' in stdout,
            'stdout': stdout[-8192:], 'stderr': stderr[-8192:],
            'payload_sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'elapsed_seconds': time.monotonic() - start}
    finally:
        sandbox.terminate()


@app.local_entrypoint()
def main(input_root: str, arm: str = 'both'):
    root = Path(input_root)
    records = []
    baseline_hashes = {}
    arms = ('phi', 'qwen') if arm == 'both' else ('phi_teacher',) if arm == 'phi_teacher' else ()
    if not arms: raise ValueError(arm)
    for selected_arm in arms:
        path = root / (selected_arm + '-baseline.json')
        baseline_hashes[selected_arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.extend((selected_arm, sample) for sample in json.loads(path.read_text())['samples'] if sample['case']['kind'] == 'code')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate, records))
    result = {'scope': 'Eight shared diagnostic code fixtures, not benchmark accuracy',
        'image': IMAGE, 'baseline_sha256': baseline_hashes, 'network_blocked': True,
        'gpu': None, 'secrets': False, 'mounted_volumes': False, 'rows': rows}
    filename = 'semantic-code-tests.json' if arm == 'both' else 'semantic-teacher-code-tests.json'
    (root / filename).write_text(json.dumps(result, indent=2))
    print('SEMANTIC_CODE_TESTS_JSON=' + json.dumps({a: sum(row['pass'] for row in rows if row['arm'] == a) for a in arms}))
