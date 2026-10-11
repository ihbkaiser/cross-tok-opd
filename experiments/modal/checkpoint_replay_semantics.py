"""Execute replayed code only in credential-free, network-blocked CPU sandboxes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import modal
from checkpoint_replay_code import extract_code

from checkpoint_replay_modal import IMAGE, PY

app=modal.App('checkpoint-replay-code-tests-20261012')
image=modal.Image.from_registry(IMAGE).entrypoint([])
RUNNER='''import json,resource,signal,sys
resource.setrlimit(resource.RLIMIT_CPU,(3,3))
resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
signal.alarm(5)
payload=json.loads(sys.argv[1]);scope={}
exec(compile(payload['code'],'generated.py','exec'),scope)
exec(compile(payload['tests'],'fixture_tests.py','exec'),scope)
print('FIXTURE_ASSERTIONS_PASSED')
'''


def evaluate(record):
    arm,sample,variant=record
    code=extract_code(sample['text'],variant)
    payload=json.dumps({'code':code,'tests':sample['case']['tests']})
    start=time.monotonic()
    sandbox=modal.Sandbox.create(PY,'-I','-c',RUNNER,payload,app=app,image=image,
        cpu=.25,memory=512,timeout=25,block_network=True,secrets=[],volumes={},
        include_oidc_identity_token=False)
    try:
        sandbox.wait();stdout,stderr=sandbox.stdout.read(),sandbox.stderr.read()
        return {'arm':arm,'variant':variant,'case_id':sample['case']['id'],'sandbox_id':sandbox.object_id,
            'exit_code':sandbox.returncode,'pass':sandbox.returncode==0 and 'FIXTURE_ASSERTIONS_PASSED' in stdout,
            'stdout':stdout[-8192:],'stderr':stderr[-8192:],
            'payload_sha256':hashlib.sha256(payload.encode()).hexdigest(),
            'elapsed_seconds':time.monotonic()-start}
    finally:sandbox.terminate()


@app.local_entrypoint()
def main(input_root:str,arms:str=''):
    root=Path(input_root);records=[];hashes={}
    for source in sorted(root.glob('*-responses.json')):
        arm=source.name.removesuffix('-responses.json')
        if arms and arm not in arms.split(','):continue
        hashes[arm]=hashlib.sha256(source.read_bytes()).hexdigest()
        for s in json.loads(source.read_text()):
            if s['case']['kind']=='code':
                records.extend([(arm,s,'author'),(arm,s,'python-version-aware')])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,records))
    result={'scope':'Eight shared code fixtures per checkpoint, not benchmark accuracy',
            'source_sha256':hashes,'image':IMAGE,'network_blocked':True,
            'gpu':None,'secrets':False,'mounted_volumes':False,'rows':rows,
            'author_extractor':'Exact pattern from pinned author-code; other evaluator logic not modified'}
    (root/'semantic-code-tests.json').write_text(json.dumps(result,indent=2))
    print('REPLAY_SEMANTIC_TESTS='+json.dumps({a:{v:sum(r['pass'] for r in rows if r['arm']==a and r['variant']==v)
          for v in ('author','python-version-aware')} for a in hashes}))
