"""Bounded checkpoint replay using the existing immutable B200 runtime."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import modal

RUN = os.environ.get('CHECKPOINT_REPLAY_RUN', 'checkpoint-replay-20261012-r1')
ASSETS = 'checkpoint-replay-assets-20261012-v1'
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
PY = '/opt/venvs/simct-b200/bin/python'
HERE = Path(__file__).parent
app = modal.App(RUN)
image = modal.Image.from_registry(IMAGE).entrypoint([])
for name in ['checkpoint_replay_assets.py', 'warmup_zip_access.py']:
    image = image.add_local_file(str(HERE/name), '/opt/replay/'+name)
for name in ['checkpoint_replay_worker.py', 'checkpoint_replay_inputs.json', 'mp_opd_parity_worker.py', 'checkpoint_replay_deep.py']:
    if (HERE/name).exists():
        image = image.add_local_file(str(HERE/name), '/opt/replay/'+name)
assets = modal.Volume.from_name(ASSETS, create_if_missing=True)
results = modal.Volume.from_name(RUN, create_if_missing=True)
warmup = modal.Volume.from_name('phi-qwen-warmup-assets-20261011-v1')
scorers = modal.Volume.from_name('llama-qwen-screen-assets-20261011-v1')
volumes = {'/checkpoints':assets, '/results':results, '/warmup':warmup, '/assets':scorers}


def execute(stage, seconds, arm='', attempt=''):
    # Reused containers otherwise retain the Volume snapshot from their first call.
    assets.reload();results.reload()
    label = stage + ('-'+arm if arm else '') + ('-'+attempt if attempt else '')
    log = Path('/results')/(label+'.log')
    if log.exists():
        raise RuntimeError('Refuse to overwrite an earlier attempt; choose a new run ID')
    env = dict(os.environ)
    env.update(PYTHONPATH='/opt/replay', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PATH='/opt/venvs/simct-b200/bin:'+env.get('PATH',''), OMP_NUM_THREADS='4',
               HF_HUB_OFFLINE='0' if stage=='prepare' else '1',
               TRANSFORMERS_OFFLINE='0' if stage=='prepare' else '1')
    libs = ['/usr/local/cuda/lib64','/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    env['LD_LIBRARY_PATH'] = ':'.join(libs)
    script = 'checkpoint_replay_assets.py' if stage=='prepare' else 'checkpoint_replay_deep.py' if stage.startswith('deep') else 'checkpoint_replay_worker.py'
    command = [PY, '-u', '/opt/replay/'+script] + ([] if stage=='prepare' else [stage, arm])
    started = time.monotonic(); pos = 0
    with log.open('x') as writer:
        proc = subprocess.Popen(command, env=env, stdout=writer, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-started > seconds:
                    raise TimeoutError('Bounded replay timeout')
                with log.open() as reader:
                    reader.seek(pos); chunk=reader.read(); pos=reader.tell()
                if chunk: print(chunk, flush=True)
                assets.commit(); results.commit(); time.sleep(5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            assets.commit(); results.commit()
        with log.open() as reader:
            reader.seek(pos); print(reader.read(), flush=True)
    receipt={'stage':stage,'arm':arm,'exit_code':proc.returncode,
             'elapsed_seconds':time.monotonic()-started,'image':IMAGE}
    (Path('/results')/(label+'-receipt.json')).write_text(json.dumps(receipt)); results.commit()
    return receipt


@app.function(image=image,cpu=4,memory=16384,timeout=1860,retries=0,volumes=volumes)
def prepare(attempt=''): return execute('prepare',1800,attempt=attempt)


@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=660,retries=0,max_containers=1,volumes=volumes)
def replay(arm,attempt=''): return execute('replay',600,arm,attempt)


@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=660,retries=0,max_containers=1,volumes=volumes)
def deep(arm,stage='deep',attempt=''):return execute(stage,600,arm,attempt)


@app.function(image=image,cpu=4,memory=8192,timeout=660,retries=0,volumes=volumes)
def score(arm='',attempt=''): return execute('score',600,arm,attempt)


@app.function(image=image,cpu=.25,memory=1024,timeout=660,retries=0,volumes={'/checkpoints':assets})
def wait_ready(arm):
    started=time.monotonic()
    while time.monotonic()-started<600:
        assets.reload()
        matches=list(Path('/checkpoints').glob('*/'+arm+'-receipt.json')) if arm!='all' else [Path('/checkpoints/ready.json')]
        if any(p.exists() for p in matches):
            return {'arm':arm,'ready':True,'waited_seconds':time.monotonic()-started}
        time.sleep(10)
    raise TimeoutError('CPU preflight did not find a verified checkpoint: '+arm)


@app.local_entrypoint()
def main(stage:str='prepare', arm:str='', attempt:str=''):
    if stage=='readiness':
        print('CHECKPOINT_READY='+json.dumps(wait_ready.remote(arm or 'all')))
        return
    if stage in ('deep','deep-temporal','deep-all'):
        wait_ready.remote('all')
        groups=['deep','deep-temporal'] if stage=='deep-all' else [stage]
        for group in groups:
            selected_arms=['grass-rl3-s42-step312','grass-rl2-s42-step312','warmup'] if group=='deep' else ['temporal-rl2-s44-step160','native-rl2-s44-step160','warmup']
            for selected in selected_arms:
                receipt=deep.remote(selected,group,attempt)
                print('CHECKPOINT_REPLAY_RECEIPT='+json.dumps(receipt),flush=True)
                if receipt['exit_code']:raise RuntimeError('Read persisted worker failure before retrying')
        return
    if stage in ('remaining','resume'):
        arms=['grass-rl2-s42-step312','grass-rl3-s42-step312','simct-s44-step40','simct-s42-step40',
              'temporal-rl2-s44-step80','temporal-rl2-s44-step160','native-rl2-s44-step80','native-rl2-s44-step160']
        if stage=='resume':arms=arms[2:]
        for selected in arms:
            wait_ready.remote(selected)
            receipt=replay.remote(selected,attempt)
            print('CHECKPOINT_REPLAY_RECEIPT='+json.dumps(receipt),flush=True)
            if receipt['exit_code']:raise RuntimeError('Read persisted worker failure before retrying')
        return
    if stage=='replay' and arm!='warmup':wait_ready.remote(arm)
    receipt = prepare.remote(attempt) if stage=='prepare' else score.remote(arm,attempt) if stage=='score' else replay.remote(arm,attempt)
    print('CHECKPOINT_REPLAY_RECEIPT='+json.dumps(receipt))
    if receipt['exit_code']: raise RuntimeError('Read persisted worker failure before retrying')
