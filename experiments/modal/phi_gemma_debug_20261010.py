"""Bounded Phi/Gemma diagnosis; cached pinned runtime, no benchmark training."""
import json
import os
import subprocess
import time
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/debug')
RUN = 'phi-gemma-debug-20261010-r1'
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
PY = '/opt/venvs/simct-b200/bin/python'
app = modal.App(RUN)
image = (modal.Image.from_registry(IMAGE).entrypoint([])
         .add_local_dir(str(ROOT/'kdflow'), '/opt/debug/kdflow', ignore=['**/__pycache__/**','**/*.pyc'])
         .add_local_file(str(ROOT/'experiments/modal/mp_opd_parity_worker.py'), '/opt/debug/mp_opd_parity_worker.py')
         .add_local_file(str(ROOT/'experiments/modal/phi_gemma_debug_worker.py'), '/opt/debug/phi_gemma_debug_worker.py'))
student = modal.Volume.from_name('simct-parity-longcap-assets-v1')
results = modal.Volume.from_name(RUN, create_if_missing=True)

def runtime_env(online=False):
    e=dict(os.environ)
    e.update(PATH='/opt/venvs/simct-b200/bin:'+e.get('PATH',''), PYTHONPATH='/opt/debug',
             HF_HOME='/runs/hf', HF_HUB_OFFLINE='0' if online else '1',
             TRANSFORMERS_OFFLINE='0' if online else '1', TOKENIZERS_PARALLELISM='false',
             PYTHONUNBUFFERED='1', RAY_USAGE_STATS_ENABLED='0', OMP_NUM_THREADS='4')
    libs=['/usr/local/cuda/lib64','/usr/local/nvidia/lib64']
    libs += [str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    e['LD_LIBRARY_PATH']=':'.join(libs)
    return e

@app.function(image=image,cpu=2,memory=8192,timeout=600,retries=0,volumes={'/runs':results,'/assets':student})
def prepare():
    code="""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig,AutoTokenizer
r=Path('/runs'); assert Path('/assets/student/config.json').exists()
revision='cfbefacb99257ffa30c83adab238a50856ac3083'
snapshot_download('microsoft/Phi-4-mini-instruct',revision=revision,local_dir='/runs/teacher',allow_patterns=['*.json','*.safetensors','*.model','merges.txt','vocab.json'],max_workers=4)
for p in ['/assets/student','/runs/teacher']:
 c=AutoConfig.from_pretrained(p,local_files_only=True);t=AutoTokenizer.from_pretrained(p,local_files_only=True)
 print('MODEL_READY',c.model_type,type(t).__name__,flush=True)
(r/'assets.json').write_text(json.dumps({'teacher_revision':revision,'student_ready':json.loads(Path('/assets/ready.json').read_text())}))
"""
    p=subprocess.run([PY,'-u','-c',code],env=runtime_env(True),timeout=540,check=True)
    results.commit()
    return {'status':'assets_ready'}

@app.function(image=image,gpu='B200',cpu=8,memory=65536,timeout=960,retries=0,max_containers=1,volumes={'/runs':results,'/assets':student})
def diagnose():
    root=Path('/runs'); log=root/'debug.log'
    if log.exists(): raise RuntimeError('Refuse duplicate paid attempt')
    with log.open('x') as f:
        p=subprocess.Popen([PY,'-u','/opt/debug/phi_gemma_debug_worker.py'],env=runtime_env(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic();pos=0
        try:
            while p.poll() is None:
                results.commit()
                with log.open() as reader:
                    reader.seek(pos);chunk=reader.read();pos=reader.tell()
                if chunk: print(chunk,flush=True)
                if time.monotonic()-start>900: raise TimeoutError('900-second GPU cap')
                time.sleep(8)
        finally:
            if p.poll() is None:
                import signal
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            results.commit()
        with log.open() as reader:reader.seek(pos);print(reader.read(),flush=True)
    receipt={'exit_code':p.returncode,'image':IMAGE,'elapsed_seconds':time.monotonic()-start}
    (root/'receipt.json').write_text(json.dumps(receipt));results.commit()
    return receipt

@app.local_entrypoint()
def main():
    print('PREPARE_RESULT',prepare.remote(),flush=True)
    receipt=diagnose.remote();print('DEBUG_RECEIPT_JSON='+json.dumps(receipt),flush=True)
    if receipt['exit_code']:raise RuntimeError('Diagnostic worker failed; inspect saved log')
