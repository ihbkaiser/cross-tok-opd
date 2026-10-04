"""One-GPU full alternating campaign: qualification -> nine trains -> 72 checkpoint evals.

Plans and job IDs are immutable/idempotent. There is no adapter or fresh-SFT
fallback on resume failure. The full GPU qualification is a success dependency.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/runai'))
BASE=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
STEPS=(40,80,120,156,200,240,280,312)
# Execution order within each seed, cheapest and most memory-forgiving first: every4 pays
# the second-order energy step once per four updates, then lowLR, then main.
VARIANTS=(('every4',1e-3,4),('lowLR',1e-4,1),('main',1e-3,1))
# Training seeds, paired with the two host roles by the split queue: the one-GPU owner
# runs its seeds' variants in sequence, the many-GPU extras run theirs in parallel, one
# variant per GPU, one seed after another on the same GPU.
# Seeds this campaign trains. One B200 runs every one of them in sequence through the
# single-GPU full queue. The two-host split queue would place the first seed on the
# one-GPU owner and the rest on the many-GPU extras, so a single-seed campaign has no
# extras plan at all and must not be submitted there.
# Overridable so a new seed extends this campaign without invalidating an existing
# case: with MP_TRAIN_SEEDS unset the definition is exactly what the pinned s43 case
# was created against, so its contract check still passes.
TRAIN_SEEDS=tuple(int(part) for part in os.environ.get('MP_TRAIN_SEEDS','43').split(',') if part.strip())
OWNER_SEEDS=TRAIN_SEEDS[:1]
EXTRA_SEEDS=TRAIN_SEEDS[1:]
WRAPPER=Path('/workspace/storage-shared/nlp/tungks/SimCT/python-b200.sh')
SELF=Path(__file__).resolve()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    from borrowed_campaign import write as atomic
    atomic(Path(path),value)


def configurations():
    return [dict(id=f'ALT-{name}-s{seed}',train_seed=seed,student_updates=312,
        B=64,M=16,micro_B=1,micro_M=4,energy_lr=lr,energy_every=every,
        student='full',optimizer='AdamW',student_lr=1e-6,scheduler_horizon=312)
        for seed in TRAIN_SEEDS for name,lr,every in VARIANTS]


def expected_trains(owner):
    """Trains a host role must show in its receipt before the round counts as produced."""
    return len(VARIANTS)*(len(OWNER_SEEDS) if owner else len(EXTRA_SEEDS))


def eval_cells():
    """Checkpoint evaluations the campaign schedules: every run at every checkpoint step."""
    return len(configurations())*len(STEPS)


def specs(case,gpu):
    case=Path(case)
    campaign = read(case/'campaign.json') if (case/'campaign.json').exists() else {}
    offload_flag = offload_env(campaign.get('offload_adam_moments', False))
    prefix='alt312-'+hashlib.sha256(str(case.resolve()).encode()).hexdigest()[:10]
    jobs=[]
    def add(label,action,deps,gpu_job=True,run=None,step=None):
        argv=['bash',str(WRAPPER),'-u',str(SELF),action,'--case',str(case)]
        if run:argv+=['--run',run]
        if step is not None:argv+=['--step',str(step)]
        key=prefix+'-'+label
        jobs.append(dict(version=1,id=key,project='full-alternating-312',cwd=str(ROOT),
            argv=argv,env={'PYTHONPATH':str(ROOT/'experiments/modal/vendor')+':'+str(ROOT),
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                'MP_OFFLOAD_ADAM_MOMENTS':offload_flag},
            gpus=[gpu] if gpu_job else 0,cpu_slots=1,timeout_seconds=7*24*3600,
            dependencies=list(deps),dependency_policy='success'))
        return key
    audit=add('data-audit','audit',[],False)
    previous=add('qualify','qualify',[audit])
    for config in configurations():previous=add(config['id'],'train',[previous],run=config['id'])
    # On one GPU finish the agreed six-run round, then generate checkpoint evals.
    scores=[]
    for config in configurations():
        for step in STEPS:
            label=f"{config['id']}-step{step}"
            gen=add(label+'-gen','generate',[previous],run=config['id'],step=step)
            scores.append(add(label+'-score','score',[gen],False,config['id'],step))
            previous=gen
    add('report','report',scores,False)
    return jobs


def checked_config(case):
    c=read(case/'campaign.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if c['source']!=str(ROOT) or c['commit']!=commit or c['runs']!=configurations():
        raise ValueError('Immutable campaign/source contract changed')
    from borrowed_campaign import sha
    for path,expected in c['input_hashes'].items():
        if sha(Path(path))!=expected:raise ValueError('Input file changed: '+path)
    return c


def offload_env(value):
    """Serialize the case-level diagnostic flag for the launcher contract."""
    if not isinstance(value, bool):
        raise ValueError('campaign offload_adam_moments must be a boolean')
    return '1' if value else '0'


def audit(case):
    """Audit actual parquet inputs against pinned test prompts without changing B."""
    import re,unicodedata
    import pyarrow.parquet as pq
    import borrowed_campaign as B
    c=checked_config(case);E,D,_=B.eval_modules()
    def normalize(x):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',x)).strip().casefold()
    def text(row):
        messages=row['messages']
        if isinstance(messages,str):messages=json.loads(messages)
        return '\n'.join(m['content'] for m in messages if m['role']=='user')
    train=pq.read_table(c['dataset']).to_pylist(); meta=pq.read_table(c['meta']).to_pylist()
    train_texts={normalize(text(row)) for row in train}
    meta_texts={normalize(text(row)) for row in meta}
    test=[]
    for bench,data in read(case/'eval-template.json')['data'].items():
        # queue_data's company-internal-v1 schema is distinct from the older
        # contract_eval.load_prepared author-code/paper-spec schema.
        prepared=E.read_json(data['path'])
        if (E.file_hash(data['path'])!=data['sha256'] or prepared['profile']!=D.PROFILE
            or prepared['benchmark']!=bench or len(prepared['items'])!=data['count']):
            raise ValueError('Historical eval data contract mismatch')
        test.extend((bench,row['id'],normalize(row['prompt'])) for row in prepared['items'])
    overlaps=[]
    for bench,key,prompt in test:
        if prompt in train_texts or prompt in meta_texts:
            overlaps.append(dict(benchmark=bench,id=key,kind='normalized_exact'))
        elif len(prompt)>=100 and any(prompt in candidate or candidate in prompt
            for candidate in train_texts|meta_texts if len(candidate)>=100):
            overlaps.append(dict(benchmark=bench,id=key,kind='containment'))
    missing=sum(not isinstance(row.get('label'),str) or not row['label'].strip() for row in meta)
    outside=len(meta_texts-train_texts)
    report=dict(train_rows=len(train),meta_rows=len(meta),meta_unique=len(meta_texts),
        missing_references=missing,meta_outside_train=outside,test_overlap=overlaps,
        scope='normalized exact + containment; no semantic-independence claim; selected teacher references may have been used by SFT')
    report['pass']=len(train)==10000 and not(missing or outside or overlaps)
    write(case/'data-audit.json',report)
    if not report['pass']:raise ValueError('Data audit failed; see data-audit.json; B was not changed')
    if c.get('meta_policy'):
        # Use the production renderer/filter/sampler before admitting any GPU.
        from transformers import AutoTokenizer
        from kdflow.datasets.prompts_dataset import PromptDataset
        from kdflow.meta_data import MetaSampler,prepare_meta,POLICY
        if c['meta_policy']!=POLICY:raise ValueError('Meta policy drift')
        tokenizer=AutoTokenizer.from_pretrained(c['student'],local_files_only=True,trust_remote_code=False)
        renderer=PromptDataset.__new__(PromptDataset)
        renderer.apply_chat_template=True;renderer.image_key=None;renderer.enable_thinking=False
        render=lambda row:renderer._build_prompt(row,tokenizer,'messages')
        report['pass']=False;write(case/'data-audit.json',report)
        prepared,excluded=prepare_meta(meta,render,tokenizer,4096)
        sampler=MetaSampler(prepared,[render(r) for r in train],42,16)
        if len(sampler.rows)<80:raise ValueError('M pool cannot guarantee 16 distinct prompts after excluding B64')
        report.update(meta_sampling=sampler.audit,meta_excluded_length=excluded)
        report['pass']=True;write(case/'data-audit.json',report)


SERVING_ATTENTION_BACKENDS = frozenset({"triton", "flashinfer", "fa3"})


def serving_env(campaign):
    """Rollout serving contract for one campaign, pinned in campaign.json.

    Defaults reproduce the historical launch (deterministic inference off,
    backend auto, radix cache untouched). A deterministic contract must be
    complete: SGLang refuses an unsupported attention backend and the evidence
    in this campaign pinned radix cache off as part of the treatment.
    """
    serving = campaign.get("serving") or {}
    deterministic = bool(serving.get("deterministic", False))
    backend = str(serving.get("attention_backend", "") or "")
    disable_radix = bool(serving.get("disable_radix_cache", False))
    try:
        seed = int(serving.get("random_seed", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign serving.random_seed must be an integer") from exc
    if deterministic:
        if backend not in SERVING_ATTENTION_BACKENDS:
            raise ValueError(
                "deterministic rollout serving requires serving.attention_backend in "
                + ",".join(sorted(SERVING_ATTENTION_BACKENDS))
            )
        if not disable_radix:
            raise ValueError("deterministic rollout serving pins serving.disable_radix_cache=true")
        if seed < 0:
            raise ValueError("deterministic rollout serving requires a non-negative serving.random_seed")
    return {
        "MP_ROLLOUT_DETERMINISTIC": "1" if deterministic else "0",
        "MP_ROLLOUT_SEED": str(seed),
        "MP_ROLLOUT_ATTENTION_BACKEND": backend,
        "MP_ROLLOUT_DISABLE_RADIX_CACHE": "1" if disable_radix else "0",
    }


def run_command(case,config,out,limit=312,pause=0,resume=False):
    c=checked_config(case);out=Path(out)
    # An old shell's MP_* flags must not silently alter this immutable campaign.
    env={k:v for k,v in os.environ.items() if not k.startswith('MP_')}
    # The launcher imports a vendor-only module (the xtoken aligner) plus the repo root.
    # Queued jobs get these paths from the manager; a direct run_command call must not
    # depend on the caller's shell having exported them.
    vendor=str(ROOT/'experiments/modal/vendor')
    inherited=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep)
               if p and p not in (vendor,str(ROOT))]
    env['PYTHONPATH']=os.pathsep.join([vendor,str(ROOT)]+inherited)

    # Diagnostic-only passthrough: a shell variable may lower the training sequence cap
    # for a bounded diagnostic. It can never reach a full run, which stays at the recipe
    # recorded in launch-config.json.
    if 0 < limit <= 30 and os.environ.get('DIAG_MAX_LEN'):
        env['MP_MAX_LEN']=os.environ['DIAG_MAX_LEN']
    env.update(MP_STUDENT_PATH=c['student'],MP_TEACHER_PATH=c['teacher'],MP_DATASET_PATH=c['dataset'],
        MP_MICRO_TRAIN_BATCH_SIZE=str(config['micro_B']),MP_META_MICRO_BATCH_SIZE=str(config['micro_M']),
        MP_ENERGY_CHECKPOINT=c['energy'],MP_SEED=str(config['train_seed']),MP_PARTITION_SEED='43',
        MP_ALTERNATING='1',MP_META_PATH=c['meta'],MP_ENERGY_LR=str(config['energy_lr']),
        MP_ENERGY_EVERY=str(config['energy_every']),MP_RESUME=str(int(resume)),
        MP_OFFLOAD_ADAM_MOMENTS=offload_env(c.get('offload_adam_moments', False)),
        MP_PAUSE_AFTER_UPDATES=str(pause),MP_SOURCE_COMMIT=c['commit'],MP_SOURCE_DIRTY='',
        # Ray appends a timestamped session and socket names below this path.
        # Keep the node-local root short enough for AF_UNIX's 107-byte limit.
        MP_RAY_TMP=f'/tmp/ar{os.getpid()}-{time.time_ns()%1000000}',
        MP_CHECKPOINT_STEPS=','.join(map(str,STEPS)),
        **serving_env(c))
    record=out.parent/(out.name+'.qualification.json')
    if record.exists():
        evidence=read(record)
        env.update(MP_QUALIFICATION_POLICY=evidence['policy'],MP_QUALIFICATION_STATUS=evidence['status'])
    out.parent.mkdir(parents=True,exist_ok=True)
    log=out.parent/(out.name+f'.attempt-{time.time_ns()}.log')
    print('RUN_LOG',log,flush=True)
    with log.open('xb') as f:
        result=subprocess.run(['bash',str(WRAPPER),'-u',str(ROOT/'experiments/runai/run_single_gpu.py'),
            'soft',str(limit),str(out)],env=env,stdout=f,stderr=subprocess.STDOUT)
    write(out.parent/(out.name+'.last-exit.json'),dict(returncode=result.returncode,log=str(log)))
    if result.returncode:raise subprocess.CalledProcessError(result.returncode,result.args)


def compare_checkpoint(a,b, expected_rollout_files=4):
    import numpy as np
    import torch
    def equal(x,y,path):
        if torch.is_tensor(x):
            if not torch.equal(x,y):raise ValueError('Resume tensor differs: '+path)
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):raise ValueError('Resume array differs: '+path)
        elif isinstance(x,dict):
            if x.keys()!=y.keys():raise ValueError('Resume keys differ: '+path)
            for k in x:equal(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):raise ValueError('Resume sequence differs: '+path)
            for i,(u,v) in enumerate(zip(x,y)):equal(u,v,path+'/'+str(i))
        elif x!=y:raise ValueError('Resume value differs: '+path)
    def load(root):
        from kdflow.training_checkpoint import inspect
        pointer=read(root/'checkpoints/latest.json')
        folder=root/'checkpoints'/pointer['directory']
        manifest=read(folder/'manifest.json')
        folder,_=inspect(root/'checkpoints',manifest['contract'],1)
        return torch.load(folder/'rank0.pt',map_location='cpu',weights_only=False),torch.load(folder/'driver.pt',map_location='cpu',weights_only=False)
    x,cx=load(a);y,cy=load(b)
    equal(x,y,'actor')
    for c in (cx,cy):c.pop('resource_sample_index',None)
    equal(cx,cy,'driver')
    # Tokens, behavior logprobs and IDs must replay as well, not just weights.
    files=sorted((a/'checkpoint/rollout_data').glob('*.jsonl'))
    if len(files)!=expected_rollout_files or {p.name for p in files}!={p.name for p in (b/'checkpoint/rollout_data').glob('*.jsonl')}:
        raise ValueError('Incomplete qualification rollout evidence')
    def trajectory(path):
        result=[]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            # SGLang includes request IDs and latency/cache telemetry in
            # meta_info. Compare policy evidence, not process-specific timings.
            info=row.pop('meta_info',{})
            row['behavior_logprobs']={k:v for k,v in info.items() if 'logprob' in k}
            result.append(row)
        return result
    for p in files:
        other=b/'checkpoint/rollout_data'/p.name
        equal(trajectory(p),trajectory(other),'rollout/'+p.name)


def qualify(case, destination=None):
    c=checked_config(case)
    if not read(case/'data-audit.json')['pass']:raise ValueError('Data audit missing/failed')
    dest=destination or case/'qualification';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'PASS.json').exists():
        if read(dest/'PASS.json')['commit']!=c['commit']:raise ValueError('Qualification source changed')
        return
    # Actual pinned company runtime supplies transformers/Ray, which are not
    # installed in the lightweight local CPU test environment.
    with (dest/f'regression-{time.time_ns()}.log').open('xb') as log:
        subprocess.run([sys.executable,'-m','pytest','tests/mp_opd',
            'tests/test_training_checkpoint.py','tests/test_meta_data.py',
            'tests/test_trajectory.py','tests/test_exact_trajectory_integration.py',
            '-q','-p','no:cacheprovider'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for variant in ('main','every4'):
        config=next(r for r in c['runs'] if r['id']==f'ALT-{variant}-s{TRAIN_SEEDS[0]}')
        uninterrupted=dest/(variant+'-continuous');resumed=dest/(variant+'-resumed')
        # Unique qualification source/output; never delete a failed canary.
        if not uninterrupted.exists():run_command(case,config,uninterrupted,limit=4)
        if not resumed.exists():run_command(case,config,resumed,limit=4,pause=3)
        summary=read(resumed/'checkpoint/run-summary.json')
        if summary['optimizer_updates']==3 and summary['stop_reason']=='checkpoint_pause':
            run_command(case,config,resumed,limit=4,resume=True)
        for root in (uninterrupted,resumed):
            s=read(root/'checkpoint/run-summary.json')
            if (s['status']!='completed' or s['optimizer_updates']!=4
                or s['energy_updates']!=4//config['energy_every']):raise ValueError('Incomplete canary')
        compare_checkpoint(uninterrupted,resumed)
    write(dest/'PASS.json',dict(commit=c['commit'],status='EXACT_MATCH',
        scope='full B64/M16; main and every4; continuous4 vs pause3+resume1; same source/runtime/GPU'))


def train(case,run,qualification=None):
    c=checked_config(case)
    qdir=qualification or case/'qualification'
    proof=read(qdir/'PASS.json') if (qdir/'PASS.json').exists() else {}
    passed=proof.get('commit')==c['commit'] and proof.get('status')=='EXACT_MATCH'
    policy=c.get('qualification_policy','required')
    if policy not in ('required','advisory','skip'):raise ValueError('Unknown qualification policy')
    if policy=='skip':passed=False
    if not passed and policy=='required':raise ValueError('Qualification not passed')
    write(case/'train'/(run+'.qualification.json'),dict(
        policy=policy,status='skipped' if policy=='skip' else 'passed' if passed else 'unverified',
        qualification_directory=str(qdir),source_commit=c['commit'],
        evidence_label='qualified' if passed else 'diagnostic-unqualified',
        continuation_authorized=policy in ('advisory','skip')))
    config=next(r for r in c['runs'] if r['id']==run);dest=case/'train'/run
    summary=dest/'checkpoint/run-summary.json'
    if summary.exists():
        s=read(summary)
        if s['status']=='completed' and s['optimizer_updates']==312 and s['energy_updates']==312//config['energy_every']:return
    for attempt in range(3):
        try:
            run_command(case,config,dest,resume=dest.exists())
            s=read(summary)
            if (s['status']!='completed' or s['optimizer_updates']!=312
                or s['energy_updates']!=312//config['energy_every']):raise ValueError('Incomplete 312-update run')
            return
        except subprocess.CalledProcessError as e:
            retry_file=dest.parent/(run+'.retries.json')
            saved=read(retry_file) if retry_file.exists() else []
            if e.returncode not in (-9,-15) or len(saved)>=2 or not (dest/'checkpoints/latest.json').exists():raise
            saved.append(dict(returncode=e.returncode,time=time.time()))
            write(retry_file,saved)
    raise ValueError('Training recovery budget exhausted')


def eval_plan(case,run,step):
    import borrowed_campaign as B
    E,D,Q=B.eval_modules();path=case/'eval'/f'{run}-step{step}'/'plan.json'
    if path.exists():
        plan=read(path)
        if plan['source']!=D.script_hashes():raise ValueError('Eval source drift')
        for data in plan['data'].values():
            if E.file_hash(data['path'])!=data['sha256']:raise ValueError('Eval data drift')
        identity=B.checkpoint_stable(Path(plan['jobs'][0]['checkpoint']['path']))
        if identity!=plan['jobs'][0]['checkpoint']:raise ValueError('Eval checkpoint drift')
        return path,plan
    if read(case/'campaign.json').get('streaming_eval'):
        from kdflow.export_ready import validate
        validate(case/'train'/run/'checkpoint',step)
    else:
        s=read(case/'train'/run/'checkpoint/run-summary.json')
        if s['status']!='completed' or s['optimizer_updates']!=312:raise ValueError('Training not complete')
    plan=read(case/'eval-template.json');identity=B.checkpoint_stable(case/'train'/run/'checkpoint'/f'step{step}')
    plan.update(jobs=[dict(id=f'{run}-step{step}',mode='soft-alternating',step=step,tier=0,checkpoint=identity)],
                source=D.script_hashes(),hours=168,admit_hours=168)
    write(path,plan)
    write(path.parent/'state.json',dict(plan_sha256=E.file_hash(path),started=time.time(),
        deadline=time.time()+168*3600,admit_until=time.time()+168*3600,jobs={},durations=[],qualification=Q.preflight('/usr/bin/python3.12')))
    return path,plan


def evaluate(case,run,step,phase):
    import borrowed_campaign as B
    E,D,Q=B.eval_modules();path,plan=eval_plan(case,run,step);ph=E.file_hash(path);job=plan['jobs'][0]
    deadline=time.time()+7*24*3600
    if phase=='generate':
        Q.GENERATION_ONLY=True
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        matches=[int(line.split(',')[0]) for line in inventory.splitlines() if line.split(',')[1].strip()==visible]
        if len(matches)!=1:raise ValueError('GPU lease/index mismatch')
        args=argparse.Namespace(plan=path,gpu=matches[0],phase='generate',concurrency=256,score_workers=1,
            score_buffer=256,score_python='/usr/bin/python3.12',min_free_gib=20)
        Q.run_checkpoint(path.parent,plan,ph,job,args,deadline)
    else:
        import resilient_score as R
        from types import SimpleNamespace
        R.install(SimpleNamespace(B=B),Q)
        Q.GENERATION_ONLY=False
        for seed in plan['seeds']:
            for bench in plan['data']:
                cell=path.parent/'cells'/job['id']/bench/str(seed)
                with Q.locked(cell/'score.lock'):
                    marker=read(cell/'generation-complete.json');server=marker['contract']['server']
                    expected=Q.cell_contract(ph,job,plan['data'],bench,seed,server)
                    if marker['contract']!=expected or E.file_hash(cell/'responses.jsonl')!=marker['responses_sha256']:
                        raise ValueError('Generation spool contract mismatch')
                    args=argparse.Namespace(phase='score',concurrency=1,score_workers=4,score_buffer=256,score_python='/usr/bin/python3.12')
                    Q.run_cell(path.parent,plan,ph,job,bench,seed,None,server,args,deadline)
        subprocess.run(['/usr/bin/python3.12',str(ROOT/'scripts/evaluation/lcbfix.py'),'--plan',str(path),
            '--out',str(path.parent/'lcbfix'),'--workers','4','--internal-code-execution'],check=True)


def report(case):
    rows=[]
    for config in configurations():
        for step in STEPS:
            path=case/'eval'/f"{config['id']}-step{step}"
            plan=read(path/'plan.json')
            for bench in plan['data']:
                for seed in plan['seeds']:
                    metric=read(path/'cells'/plan['jobs'][0]['id']/bench/str(seed)/'metrics.json')
                    rows.append(dict(run=config['id'],group=config['id'].rsplit('-s',1)[0],
                        train_seed=config['train_seed'],step=step,benchmark=bench,eval_seed=seed,metrics=metric))
            if read(path/'lcbfix/summary.json')['status']!='completed':raise ValueError('LCBfix incomplete')
    write(case/'report.json',dict(status='completed',runs=len(configurations()),checkpoints=eval_cells(),cells=rows,
          eval_seeds=[42,43,44],note='Training seeds and evaluation seeds are separate axes; no imputed failed scores'))


def submit(args):
    import borrowed_campaign as B
    from queue_alternating_followup import choose_state
    case=args.case.resolve();case.mkdir(parents=True,exist_ok=True)
    gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    expected_pod=os.environ.get('MP_EXPECTED_POD','hieplh8-beyond-leakage-1-0-0')
    if socket.gethostname()!=expected_pod:
        raise ValueError('Unexpected pod '+socket.gethostname()
            +'; set MP_EXPECTED_POD to the verified allocation before submit')
    print('SUBMIT_POD='+socket.gethostname(),flush=True)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    E,D,Q=B.eval_modules()
    with Q.locked(case/'submission.lock'):
        config_path=case/'campaign.json'
        if not config_path.exists():
            template=read(BASE/'six-b200-v1/template.json')
            if template['seeds']!=[42,43,44] or set(template['data'])!=set(E.CAPS):raise ValueError('Eval protocol differs')
            paths={str(Path(d['path'])):d['sha256'] for d in template['data'].values()}
            shared=BASE.parent/'SimCT'
            c=dict(source=str(ROOT),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                runs=configurations(),steps=list(STEPS),student=str(shared/'runs/qwen-gemma-sft-paper-20260908-045828/checkpoint'),
                teacher='/workspace/storage-shared/models/Qwen2.5-7B-Instruct',
                dataset=str(shared/'data/qwen-author/data/prompts.parquet'),meta=str(shared/'data/qwen-author/data/selected.parquet'),
                energy=str(BASE/'owner-tungks-0-1/energy-select-4.pt'))
            for name in ('dataset','meta','energy'):paths[c[name]]=B.sha(Path(c[name]))
            if paths[c['energy']]!=B.ENERGY_SHA:raise ValueError('Energy identity differs')
            for raw,expected in paths.items():
                if B.sha(Path(raw))!=expected:raise ValueError('Input hash mismatch: '+raw)
            c['input_hashes']=paths
            write(case/'eval-template.json',template);write(config_path,c)
        checked_config(case)
        if getattr(args,'prepare_only',False):
            # Campaign creation and DAG construction are separate steps so the campaign-level
            # knobs (offload, serving) can be set on campaign.json before job envs are built.
            print('PREPARED '+str(config_path),flush=True)
            print(json.dumps(read(config_path),indent=2),flush=True)
            return
        jobs=specs(case,gpu);write(case/'queue-plan.json',jobs)
        args.gpu_uuid=gpu
        state=choose_state(args)
        sys.path.insert(0,str(args.manager))
        from job_manager.store import connect,rows,submit as put
        from job_manager.__main__ import snapshot
        db=connect(state)
        try:
            snap=snapshot(db)
            if not snap['manager']['running'] or snap['paused'] or snap['quarantined']:raise ValueError('Manager not ready')
            current={j['id']:j for j in rows(db)}
            for job in jobs:
                if job['id'] in current and current[job['id']]['spec']!=job:raise ValueError('Existing job spec changed')
            pending=[j for j in jobs if j['id'] not in current]
            if pending:put(db,pending)
            write(case/'queue-receipt.json',dict(state=str(state),host=socket.gethostname(),gpu=gpu,jobs=[j['id'] for j in jobs]))
            print(json.dumps(snapshot(db),indent=2))
        finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','audit','qualify','train','generate','score','report','status'])
    p.add_argument('--case',type=Path,required=True);p.add_argument('--run');p.add_argument('--step',type=int,choices=STEPS)
    p.add_argument('--manager',type=Path,default=BASE/'job-manager');p.add_argument('--state',type=Path);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();case=a.case.resolve()
    if a.action=='submit':return submit(a)
    if a.action=='status':
        receipt=read(case/'queue-receipt.json');sys.path.insert(0,str(a.manager))
        from job_manager.store import connect
        from job_manager.__main__ import snapshot
        db=connect(Path(receipt['state']))
        try:print(json.dumps(snapshot(db),indent=2))
        finally:db.close()
        return
    checked_config(case)
    if a.action=='audit':audit(case)
    elif a.action=='qualify':qualify(case)
    elif a.action=='train':train(case,a.run)
    elif a.action in ('generate','score'):evaluate(case,a.run,a.step,a.action)
    elif a.action=='report':report(case)


if __name__=='__main__':main()
