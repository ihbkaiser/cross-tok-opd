"""Bounded MP-OPD atomic diagnostic matched to SimCT r3, no task-specific SFT."""
import json,os,subprocess,time,hashlib
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
PYTHON='/opt/venvs/simct-b200/bin/python'
IMAGE='docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
RUN='mp-opd-phi-gemma-nosft-50-r1'
PROMPTS_REPO='codemaivanngu/simct-author-code-10k-prompts'
REVS={'student':'299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8','teacher':'cfbefacb99257ffa30c83adab238a50856ac3083'}
app=modal.App(RUN)
image=modal.Image.from_registry(IMAGE).entrypoint([]).env({'PYTHONPATH':'/opt/overlay/experiments/modal/vendor:/opt/overlay','PYTHONUNBUFFERED':'1'}).add_local_dir(str(ROOT/'kdflow'),'/opt/overlay/kdflow',ignore=['**/__pycache__/**','**/*.pyc']).add_local_dir(str(ROOT/'tests'),'/opt/overlay/tests',ignore=['**/__pycache__/**','**/*.pyc']).add_local_dir(str(ROOT/'experiments/modal'),'/opt/overlay/experiments/modal',ignore=['**/__pycache__/**','**/*.pyc'])
cache=modal.Volume.from_name('simct-phi-gemma-assets',create_if_missing=True)
outputs=modal.Volume.from_name(RUN,create_if_missing=True)
source=modal.Volume.from_name('simct-author-data-v1') if os.environ.get('SIMCT_STAGE')=='export' else cache

def env(online=False):
    e=dict(os.environ);e.update(PATH='/opt/venvs/simct-b200/bin:'+e.get('PATH',''),PYTHONPATH='/opt/overlay/experiments/modal/vendor:/opt/overlay',HF_HOME='/assets/hf',HF_HUB_OFFLINE='0' if online else '1',HF_DATASETS_OFFLINE='0' if online else '1',TRANSFORMERS_OFFLINE='0' if online else '1',TOKENIZERS_PARALLELISM='false',RAY_USAGE_STATS_ENABLED='0',NCCL_CUMEM_HOST_ENABLE='0',OMP_NUM_THREADS='4',WANDB_SILENT='true',KDFLOW_TRUST_REMOTE_CODE='0')
    libs=['/usr/local/cuda/lib64','/usr/local/nvidia/lib64']+[str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')]
    e['LD_LIBRARY_PATH']=':'.join(libs)
    return e

@app.function(image=image,cpu=2,memory=8192,timeout=900,retries=0,volumes={'/data':source})
def export_prompts():
    code=r"""
from pathlib import Path
import json,os,shutil,hashlib
from datasets import load_from_disk
from huggingface_hub import HfApi
root=Path('/data/author-v1');ready=json.loads((root/'ready.json').read_text());q=json.loads((root/'qualification.json').read_text())
assert q['status']=='author_input_audit_pass' and ready['rows_sha256']==q['rows_sha256']
ds=load_from_disk(str(root/'mixed_math_code_10k_with_source'));assert len(ds)==10000
stage=Path('/tmp/prompts-export');stage.mkdir(exist_ok=True)
ds.to_parquet(str(stage/'prompts.parquet'))
for name in ['ready.json','qualification.json','acquisition.json']:shutil.copyfile(root/name,stage/name)
(stage/'README.md').write_text('---\nlicense: other\n---\n# SimCT author-code prompts\n10000 prompts only; no teacher responses or SFT targets. See acquisition.json for source terms and qualification.json for audit.\n')
api=HfApi(token=os.environ['HF_TOKEN']);assert api.whoami()['name']=='codemaivanngu'
repo='codemaivanngu/simct-author-code-10k-prompts';api.create_repo(repo,repo_type='dataset',private=True,exist_ok=True)
c=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=stage,commit_message='Export qualified author-code prompts for OPD')
print('PROMPTS_EXPORTED',c.oid,hashlib.sha256((stage/'prompts.parquet').read_bytes()).hexdigest(),flush=True)
"""
    subprocess.run([PYTHON,'-c',code],env=env(True),check=True,timeout=800)

@app.function(image=image,cpu=4,memory=16384,timeout=1800,retries=0,volumes={'/assets':cache,'/runs':outputs})
def prepare(prompts_revision:str,prompts_sha:str):
    ready=json.loads(Path('/assets/ready.json').read_text())
    assert ready['rows']==10000 and ready['revisions']==REVS
    assert ready['prompts_revision']==prompts_revision and ready['prompt_sha256']==prompts_sha
    assert hashlib.sha256(Path('/assets/prompts.parquet').read_bytes()).hexdigest()==prompts_sha
    e=env();e['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT']='1'
    test=subprocess.run([PYTHON,'-m','pytest','-q','tests/mp_opd', 'tests/test_simct_paper_scores.py','tests/test_span_ctkd_metrics.py','tests/test_trajectory.py','tests/test_exact_trajectory_integration.py'],cwd='/opt/overlay',env=e,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=200)
    Path('/runs/preflight.log').write_text(test.stdout);outputs.commit();print(test.stdout[-3500:],flush=True)
    if test.returncode:raise RuntimeError('Preflight failed')
    Path('/runs/preflight-pass.json').write_text(json.dumps(dict(status='pass',source_scores_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('/opt/overlay/kdflow').rglob('*.py')))).hexdigest())));outputs.commit()

@app.function(image=image,gpu='B200' if os.environ.get('SIMCT_STAGE')=='train' else None,cpu=16,memory=98304,timeout=3300,retries=0,max_containers=1,volumes={'/assets':cache,'/runs':outputs})
def train(commit:str):
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('',None):
        import subprocess as sp
        sp.run(['nvidia-smi','-L'],check=True)
    root=Path('/runs');inv=root/'invocation.json'
    if inv.exists():raise RuntimeError('Existing training attempt requires inspection; no duplicate retry')
    ready=json.loads(Path('/assets/ready.json').read_text());preflight=json.loads(Path('/runs/preflight-pass.json').read_text())
    assert preflight['source_scores_sha256']==hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('/opt/overlay/kdflow').rglob('*.py')))).hexdigest()
    opts=dict(num_nodes=1,num_gpus_per_node=1,backend='fsdp2',student_name_or_path='/assets/student',teacher_name_or_path='/assets/teacher',attn_implementation='sdpa',num_epochs=2,train_batch_size=64,micro_train_batch_size=1,learning_rate=1e-6,lr_warmup_ratio=.05,lr_scheduler='cosine_with_min_lr',min_lr=0,weight_decay=0.,gradient_checkpointing=True,enable_sleep=True,bf16=True,seed=42,save_path='/runs/checkpoint',ckpt_path='/runs/checkpoints',train_dataset_path='/assets/prompts.parquet',input_key='messages',apply_chat_template=True,enable_thinking=False,max_samples=10000,prompt_max_len=0,max_len=4096,preprocess_num_workers=4,rollout_num_engines=1,rollout_disable_piecewise_cuda_graph=True,rollout_tp_size=1,rollout_mem_fraction_static=.25,rollout_batch_size=64,generate_max_len=4096,n_samples_per_prompt=1,temperature=.6,top_p=.95,teacher_tp_size=1,teacher_pp_size=1,teacher_ep_size=1,teacher_dp_size=1,teacher_mem_fraction_static=.3,teacher_context_length=16384,teacher_forward_n_batches=8,kd_algorithm='mp_opd',mp_opd_mode='atomic',mp_opd_max_span_length=1,kd_loss_fn='rkl',kd_ratio=1.,span_score_mode='mean_logprob',exact_token_trajectory=True,enforce_max_sequence_length=True,diagnostic_max_updates=50,diagnostic_collapse_gate=True,save_steps=10,logging_steps=1,use_wandb=True,wandb_org='kieusontung8-hanoi-university-of-science-and-technology',wandb_project='vdt-simct-tunix-reproduction',wandb_run_id=RUN,wandb_run_name=RUN,wandb_group='phi-gemma-nosft',wandb_job_type='implementation-validation',wandb_tags='method:mp-opd,regime:on-policy,objective:path-credit,variant:atomic-no-sft,platform:modal,accelerator:b200x1,budget:50-update',wandb_mode='online',wandb_dir='/runs/wandb')
    command=[PYTHON,'-m','kdflow.cli.train_kd_on_policy']
    for k,v in opts.items():command+=['--'+k,str(v)]
    inv.write_text(json.dumps(dict(source_commit=commit,image=IMAGE,assets=ready,command=command,no_task_specific_sft=True,score='detached scalar atom path credit times NLL',baseline_source='f72c8d7',baseline_run='simct-phi-gemma-nosft-100-r3',wall_cap_seconds=3000,scheduler_horizon='2*floor(10000/64)=312, warmup16',max_updates=50),indent=2));outputs.commit()
    process=None;started=time.monotonic()
    try:
        with (root/'train.log').open('x') as f:
            process=subprocess.Popen(command,cwd='/opt/overlay',env=env(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                outputs.commit()
                if time.monotonic()-started>3000:raise TimeoutError('Paid run time cap')
                time.sleep(15)
        if process.returncode:raise RuntimeError('Training process failed')
        result=json.loads((root/'checkpoint/run-summary.json').read_text());assert result['optimizer_updates']==50 and result['status']=='completed'
        (root/'result.json').write_text(json.dumps(dict(status='completed',training=result)));return result
    except Exception as e:
        (root/'result.json').write_text(json.dumps(dict(status='failed',error_type=type(e).__name__,error=str(e))))
        raise
    finally:
        if process and process.poll() is None:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        outputs.commit()

@app.local_entrypoint()
def main(stage:str,prompts_revision:str='',prompts_sha:str=''):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit first')
    import shlex
    secret_values={}
    for line in Path('/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env').read_text().splitlines():
        line=line.removeprefix('export ')
        for key in ('HF_TOKEN','WANDB_API_KEY'):
            if line.startswith(key+'='):secret_values[key]=shlex.split(line.split('=',1)[1])[0]
    secret=modal.Secret.from_dict(secret_values)
    if stage=='export':call=export_prompts.with_options(secrets=[secret]).spawn()
    elif stage=='prepare':call=prepare.with_options(secrets=[secret]).spawn(prompts_revision,prompts_sha)
    elif stage=='train':call=train.with_options(secrets=[secret]).spawn(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    else:raise ValueError('Unknown stage')
    print('SUBMITTED',stage,call.object_id,flush=True)
