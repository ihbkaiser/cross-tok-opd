"""Bounded pinned-runtime Qwen teacher audit, reusing actual company warmup."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import modal

RUN=os.environ.get('QWEN_DEBUG_RUN','qwen-pipeline-debug-20261011-r1')
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/warmup_debug')
PY='/opt/venvs/simct-b200/bin/python'
IMAGE='docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app=modal.App(RUN)
image=modal.Image.from_registry(IMAGE).entrypoint([])
for name in ['qwen_pipeline_debug_worker.py','warmup_debug_worker.py','mp_opd_parity_worker.py','qwen_scoring_isolation.py','qwen_scoring_reference.py']:
    image=image.add_local_file(str(ROOT/'experiments/modal'/name),'/opt/warmup_debug/'+name)
image=image.add_local_file(str(ROOT/'scripts/sft/answer_verifier.py'),'/opt/warmup_debug/answer_verifier.py')
corpus=os.environ.get('QWEN_DEBUG_CORPUS')
if corpus:image=image.add_local_file(corpus,'/opt/warmup_debug/prompts.parquet')
teacher=modal.Volume.from_name('qwen-pipeline-teacher-20261011-v1',create_if_missing=True)
warmup=modal.Volume.from_name('phi-qwen-warmup-assets-20261011-v1')
prior=modal.Volume.from_name('phi-qwen-warmup-debug-20261011-r1')
results=modal.Volume.from_name(RUN,create_if_missing=True)
volumes={'/teacher_assets':teacher,'/warmup':warmup,'/prior':prior,'/results':results}

def execute(stage,seconds):
    log=Path('/results')/(stage+'.log')
    if log.exists():raise RuntimeError('Refuse overwrite; inspect attempt and choose a new run ID')
    env=dict(os.environ);env.update(PYTHONPATH='/opt/warmup_debug',PYTHONUNBUFFERED='1',
        PATH='/opt/venvs/simct-b200/bin:'+env.get('PATH',''),HF_HOME='/teacher_assets/hf',
        HF_HUB_OFFLINE='0' if stage=='prepare' else '1',TRANSFORMERS_OFFLINE='0' if stage=='prepare' else '1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT='1')
    libs=['/usr/local/cuda/lib64','/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    env['LD_LIBRARY_PATH']=':'.join(libs)
    started=time.monotonic();pos=0
    with log.open('x') as f:
        special={'isolate':'qwen_scoring_isolation.py','reference':'qwen_scoring_reference.py'}
        command=[PY,'-u','/opt/warmup_debug/'+special[stage]] if stage in special else [PY,'-u','/opt/warmup_debug/qwen_pipeline_debug_worker.py',stage]
        proc=subprocess.Popen(command,
            env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-started>seconds:raise TimeoutError('Bounded diagnostic timeout')
                with log.open() as reader:reader.seek(pos);chunk=reader.read();pos=reader.tell()
                if chunk:print(chunk,flush=True)
                results.commit();time.sleep(5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            teacher.commit();results.commit()
        with log.open() as reader:reader.seek(pos);print(reader.read(),flush=True)
    receipt={'stage':stage,'exit_code':proc.returncode,'elapsed_seconds':time.monotonic()-started,'image':IMAGE}
    (Path('/results')/(stage+'-receipt.json')).write_text(json.dumps(receipt));results.commit();return receipt

@app.function(image=image,cpu=4,memory=8192,timeout=960,retries=0,volumes=volumes)
def prepare():return execute('prepare',900)

@app.function(image=image,cpu=4,memory=8192,timeout=360,retries=0,volumes=volumes)
def audit():return execute('audit',300)

@app.function(image=image,cpu=2,memory=4096,timeout=240,retries=0,volumes=volumes)
def mapping():return execute('mapping',180)

@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=600,retries=0,max_containers=1,volumes=volumes)
def probe():return execute('probe',540)

@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=900,retries=0,max_containers=1,volumes=volumes)
def isolate():return execute('isolate',840)

@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=480,retries=0,max_containers=1,volumes=volumes)
def reference():return execute('reference',420)

@app.local_entrypoint()
def main(stage:str='prepare'):
    receipt={'prepare':prepare,'audit':audit,'mapping':mapping,'probe':probe,'isolate':isolate,'reference':reference}[stage].remote();print('QWEN_DEBUG_RECEIPT='+json.dumps(receipt))
    if receipt['exit_code']:raise RuntimeError('Read persisted worker failure before retrying')
