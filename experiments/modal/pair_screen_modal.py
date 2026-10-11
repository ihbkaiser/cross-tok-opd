"""Bounded Qwen3 student screen on the existing B200 runtime."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import modal

RUN = os.environ.get('PAIR_SCREEN_RUN', 'llama-qwen-screen-20261011-r3')
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/screen')
PY = '/opt/venvs/simct-b200/bin/python'
image = modal.Image.from_registry(IMAGE).entrypoint([])
image = image.add_local_file(str(ROOT/'experiments/modal/pair_screen_worker.py'), '/opt/screen/worker.py')
image = image.add_local_file(str(ROOT/'experiments/modal/pair_screen_rescore.py'), '/opt/screen/rescore.py')
image = image.add_local_file(str(ROOT/'experiments/modal/pair_screen_mathverify.py'), '/opt/screen/mathverify_worker.py')
for name in ('contract_eval.py', 'evaluation.py'):
    image = image.add_local_file(str(ROOT/'scripts/evaluation'/name), '/opt/screen/scripts/evaluation/'+name)
app = modal.App(RUN)
assets = modal.Volume.from_name('llama-qwen-screen-assets-20261011-v1', create_if_missing=True)
results = modal.Volume.from_name(RUN, create_if_missing=True)
volumes = {'/assets': assets, '/results': results}


def execute(stage, seconds):
    log = Path('/results')/(stage+'.log')
    if log.exists(): raise RuntimeError('Attempt exists; use a new run ID')
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               HF_HOME='/assets/hf-cache', SCREEN_IMAGE=IMAGE, OMP_NUM_THREADS='4')
    env['PATH'] = str(Path(PY).parent)+':'+env.get('PATH', '')
    if stage != 'prepare': env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    else:
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            env.pop(key, None)
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    env['LD_LIBRARY_PATH'] = ':'.join(libs)
    proc = None; error = None; pos = 0; start = time.monotonic()
    try:
        with log.open('x') as f:
            worker = '/opt/screen/rescore.py' if stage == 'rescore' else '/opt/screen/worker.py'
            proc = subprocess.Popen([PY, '-u', worker, stage], env=env,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic()-start > seconds: raise TimeoutError('Bounded screen timeout')
                with log.open() as f: f.seek(pos); chunk = f.read(); pos = f.tell()
                if chunk: print(chunk, flush=True)
                results.commit(); time.sleep(5)
    except Exception as exc: error = type(exc).__name__+': '+str(exc)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        receipt = {'stage': stage, 'exit_code': proc.returncode if proc else None,
                   'error': error, 'elapsed_seconds': time.monotonic()-start, 'image': IMAGE}
        (Path('/results')/(stage+'-receipt.json')).write_text(json.dumps(receipt))
        assets.commit(); results.commit()
    with log.open() as f: f.seek(pos); print(f.read(), flush=True)
    return receipt


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0, volumes=volumes)
def prepare(): return execute('prepare', 840)


@app.function(image=image, cpu=4, memory=16384, timeout=600, retries=0, volumes=volumes)
def rescore(): return execute('rescore', 540)


@app.function(image=image, gpu='B200', cpu=8, memory=65536, timeout=1080,
              retries=0, max_containers=1, volumes=volumes)
def screen(arm): return execute(arm, 1020)


@app.local_entrypoint()
def main(stage: str = 'prepare'):
    if stage not in {'prepare', 'rescore', 'qwen-nonthinking', 'qwen-thinking'}: raise ValueError(stage)
    receipt = prepare.remote() if stage == 'prepare' else rescore.remote() if stage == 'rescore' else screen.remote(stage)
    print('SCREEN_RECEIPT='+json.dumps(receipt))
    if receipt['exit_code'] != 0 or receipt['error']: raise RuntimeError('Screen failed; inspect persisted evidence')
