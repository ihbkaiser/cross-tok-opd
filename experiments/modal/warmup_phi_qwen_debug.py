"""Bounded actual-company-SFT diagnostics, pinned runtime and persistent results."""
import json
import os
from pathlib import Path
import subprocess
import time

import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/warmup_debug')
RUN=os.environ.get('WARMUP_DEBUG_RUN','phi-qwen-warmup-debug-20261011-r1')
IMAGE='docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
PY='/opt/venvs/simct-b200/bin/python'
app=modal.App(RUN)
image=modal.Image.from_registry(IMAGE).entrypoint([])
for name in ('warmup_zip_access.py','warmup_debug_prepare.py','warmup_debug_worker.py','warmup_credit_probe.py','warmup_teacher_baseline.py','warmup_teacher_score_readout.py','warmup_teacher_tokenization.py','warmup_tokenizer_audit.py','warmup_teacher_token_path_scores.py','warmup_tokenizer_regression.py'):
    image=image.add_local_file(str(ROOT/'experiments/modal'/name),'/opt/warmup_debug/'+name)
image=image.add_local_file(str(ROOT/'experiments/modal/warmup_debug_inputs.json'),'/opt/warmup_debug/inputs.json')
image=image.add_local_file(str(ROOT/'experiments/modal/mp_opd_parity_worker.py'),'/opt/warmup_debug/mp_opd_parity_worker.py')
image=image.add_local_file(str(ROOT/'scripts/sft/answer_verifier.py'),'/opt/warmup_debug/answer_verifier.py')
image=image.add_local_dir(str(ROOT/'kdflow'),'/opt/warmup_debug/kdflow',ignore=['**/__pycache__/**','**/*.pyc'])
assets=modal.Volume.from_name('phi-qwen-warmup-assets-20261011-v1',create_if_missing=True)
results=modal.Volume.from_name(RUN,create_if_missing=True)
teacher=modal.Volume.from_name('phi-gemma-debug-20261010-r1')

def environment():
    e=dict(os.environ)
    e.update(PATH='/opt/venvs/simct-b200/bin:'+e.get('PATH',''),PYTHONPATH='/opt/warmup_debug',
        PYTHONUNBUFFERED='1',HF_HOME='/warmup/hf',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT='1')
    libs=['/usr/local/cuda/lib64','/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    e['LD_LIBRARY_PATH']=':'.join(libs)
    return e

def execute(worker,seconds):
    log=Path('/results')/(worker+'.log')
    if log.exists():raise RuntimeError('Refuse duplicate attempt; use an inspected new run label')
    start=time.monotonic();pos=0
    with log.open('x') as f:
        process=subprocess.Popen([PY,'-u','/opt/warmup_debug/'+worker+'.py'],env=environment(),
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-start>seconds:raise TimeoutError('Bounded diagnostic timeout')
                with log.open() as reader:reader.seek(pos);chunk=reader.read();pos=reader.tell()
                if chunk:print(chunk,flush=True)
                results.commit();time.sleep(5)
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            assets.commit();results.commit()
        with log.open() as reader:reader.seek(pos);print(reader.read(),flush=True)
    receipt={'worker':worker,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-start,'image':IMAGE}
    (Path('/results')/(worker+'-receipt.json')).write_text(json.dumps(receipt));results.commit()
    return receipt

@app.function(image=image,cpu=4,memory=16384,timeout=1500,retries=0,volumes={'/warmup':assets,'/results':results})
def prepare():return execute('warmup_debug_prepare',1440)

@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=1260,retries=0,max_containers=1,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def diagnose():return execute('warmup_debug_worker',1200)

@app.function(image=image,gpu='B200',cpu=8,memory=32768,timeout=420,retries=0,max_containers=1,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def credit():return execute('warmup_credit_probe',360)

@app.function(image=image,gpu='B200',cpu=8,memory=32768,timeout=540,retries=0,max_containers=1,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def teacher_baseline():return execute('warmup_teacher_baseline',480)

@app.function(image=image,cpu=2,memory=4096,timeout=180,retries=0,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def teacher_readout():return execute('warmup_teacher_score_readout',120)

@app.function(image=image,cpu=2,memory=4096,timeout=180,retries=0,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def teacher_tokenization():return execute('warmup_teacher_tokenization',120)

@app.function(image=image,cpu=2,memory=4096,timeout=180,retries=0,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def tokenizer_audit():return execute('warmup_tokenizer_audit',120)

@app.function(image=image,gpu='B200',cpu=8,memory=32768,timeout=540,retries=0,max_containers=1,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def token_path_scores():return execute('warmup_teacher_token_path_scores',480)

@app.function(image=image,cpu=2,memory=4096,timeout=180,retries=0,
    volumes={'/warmup':assets,'/results':results,'/teacher_assets':teacher})
def tokenizer_regression():return execute('warmup_tokenizer_regression',120)

@app.local_entrypoint()
def main(stage:str='prepare'):
    stages={'prepare':prepare,'diagnose':diagnose,'credit':credit,'teacher-baseline':teacher_baseline,'teacher-readout':teacher_readout,'teacher-tokenization':teacher_tokenization,'tokenizer-audit':tokenizer_audit,'token-path-scores':token_path_scores,'tokenizer-regression':tokenizer_regression}
    receipt=stages[stage].remote() if stage in stages else None
    if receipt is None:raise ValueError(stage)
    print('WARMUP_DEBUG_RECEIPT_JSON='+json.dumps(receipt),flush=True)
    if receipt['exit_code']:raise RuntimeError('Worker failed; read saved evidence')
