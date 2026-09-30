"""Eight-hour campaign on the separately borrowed host, with generic queue v1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/workspace/storage-shared/nlp/tungks')
ENERGY_SHA = 'c7fe0fff9d9f46bb64962d4d0baccd119c6f4fd60a232f05aa5ea026113817b6'
OLD_WORK = BASE/'campaign20-A7hEqD4H/work'
OLD_RANDOM = BASE/'random43-train-txaDvRVj/runs/qwen-gemma-mp_opd-random-gpu0-limit0-20260911-042532-895995'
OLD_FIXED = OLD_WORK/'fixed/qwen-gemma-mp_opd-fixed-gpu0-limit0-20260910-121704-708660'
OLD_ATOMIC = OLD_WORK/'atomic/qwen-gemma-mp_opd-atomic-gpu1-limit0-20260910-180146-803721'
HOST_WRAPPER = ROOT/'experiments/runai/python-b200-host.sh'
SELF = Path(__file__).resolve()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def execute(argv, **kw):
    subprocess.run([str(x) for x in argv], check=True, **kw)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def eval_modules():
    sys.path.insert(0, str(ROOT/'scripts/evaluation'))
    import contract_eval as E
    import queue_data as D
    import eval_queue as Q
    return E, D, Q


def checkpoint_stable(path):
    E, _, _ = eval_modules()
    files = list(path.glob('*'))
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file()}
    identity = E.checkpoint_identity(path)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in path.glob('*') if p.is_file()}
    if before != after:
        raise ValueError('Checkpoint changed during hashing: '+str(path))
    return identity


def build_eval(work, name, checkpoints, deadline):
    E, D, Q = eval_modules()
    if time.time()+600 >= deadline:
        raise ValueError('Less than ten minutes left for evaluation admission')
    plan = E.read_json(BASE/'endpoint-recovery-y8cQZzRW/work/eval-endpoints/plan.json')
    for data in plan['data'].values():
        if E.file_hash(data['path']) != data['sha256']:
            raise ValueError('Evaluation data drift')
    jobs = [dict(id=key, mode=mode, step=step, tier=0, checkpoint=checkpoint_stable(cp))
            for key, mode, step, cp in checkpoints]
    target = work/name
    target.mkdir(exist_ok=False)
    plan.update(jobs=jobs, source=D.script_hashes(), hours=(deadline-time.time())/3600,
                admit_hours=(deadline-time.time())/3600)
    plan['protocol']['scope'] = 'Exploratory matched-update comparison; 3 evaluation seeds, not 3 training seeds'
    E.write_new(target/'plan.json', plan)
    qualification = Q.preflight('/usr/bin/python3.12')
    Q.atomic_json(target/'state.json', dict(plan_sha256=E.file_hash(target/'plan.json'),
        started=time.time(), deadline=deadline, admit_until=deadline-300,
        jobs={}, durations=[], qualification=qualification))
    return target/'plan.json'


def numeric_gate(work):
    os.environ['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT'] = '1'
    sys.path.insert(0, str(ROOT))
    import torch
    from kdflow.algorithms._mp_opd_energy import MPAtomEnergy, load_energy_checkpoint
    from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition
    energy = work/'energy-select-4.pt'
    if sha(energy) != ENERGY_SHA:
        raise ValueError('Energy SHA mismatch')
    net = MPAtomEnergy(10,32,2)
    opt = torch.optim.AdamW(net.parameters())
    step = load_energy_checkpoint(energy, net, opt, expected_extra_config={'max_span_length':2})
    if step != 128 or not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Energy checkpoint contract failed')
    rows = []
    with torch.no_grad():
        for length in (128, 1031, 2048, 4096):
            for scale in (0., 3., 10.):
                e = torch.full((length,2),scale,device='cuda')
                out = semi_markov_partition(e)
                error = float(out.coverage_max_error)
                if not torch.isfinite(out.entropy) or error > 1e-6:
                    raise ValueError('Long-chain partition invariant failed')
                rows.append(dict(atoms=length,energy=scale,coverage_max_error=error))
    write(work/'numeric-qualification.json', dict(technical_pass=True, device=torch.cuda.get_device_name(),
          torch=torch.__version__, energy_sha256=sha(energy), energy_updates=step, cases=rows,
          scope='Synthetic long-chain numerical qualification, not model efficacy'))
    print('NUMERIC_QUALIFIED', json.dumps(rows), flush=True)


def validate_canary(work):
    summaries = list((work/'soft-canary').glob('*/checkpoint/run-summary.json'))
    if len(summaries) != 1:
        raise ValueError('Expected one canary summary')
    summary = json.loads(summaries[0].read_text())
    if summary['status'] != 'completed' or summary['optimizer_updates'] != 5:
        raise ValueError('Canary did not complete five updates')
    run = summaries[0].parents[1]
    if Path(str(run)+'.exitcode').read_text().strip() != '0':
        raise ValueError('Canary exit was not zero')
    launch = json.loads((run/'launch-config.json').read_text())
    if launch['energy_sha256'] != ENERGY_SHA or launch['partition_dp_dtype'] != 'float64':
        raise ValueError('Canary source/energy contract drift')
    checkpoint_stable(run/'checkpoint/step5')
    write(work/'canary-qualification.json', dict(technical_pass=True, updates=5, checkpoint=str(run/'checkpoint/step5')))


def trained_run(work, kind):
    summaries = list((work/kind).glob('*/checkpoint/run-summary.json'))
    if len(summaries) != 1:
        raise ValueError('Missing final/partial summary for '+kind)
    run = summaries[0].parents[1]
    summary = json.loads(summaries[0].read_text())
    if summary['status'] not in ('completed','stopped') or summary['optimizer_updates'] < 20:
        raise ValueError('Insufficient saved training updates')
    if summary['status'] == 'stopped' and summary['stop_reason'] != 'deadline_checkpoint_reserve':
        raise ValueError('Non-deadline training stop')
    if Path(str(run)+'.exitcode').read_text().strip() != '0':
        raise ValueError('Training did not exit cleanly')
    return run


def prepare_new_eval(work):
    cfg = json.loads((work/'campaign.json').read_text())
    soft, fixed = trained_run(work,'soft-train'), trained_run(work,'fixed-train')
    left, right = [json.loads((p/'launch-config.json').read_text()) for p in (soft,fixed)]
    keys = ('seed','train_batch_size','micro_train_batch_size','lr_scheduler_horizon_steps',
            'temperature','learning_rate','student_name_or_path','teacher_name_or_path')
    if any(left['options'][k] != right['options'][k] for k in keys):
        raise ValueError('Training control mismatch')
    if left['dataset_sha256'] != right['dataset_sha256'] or left['models_sha256'] != right['models_sha256'] or left['runtime_versions'] != right['runtime_versions']:
        raise ValueError('Input/runtime mismatch')
    def steps(run):
        return {int(p.name[4:]) for p in (run/'checkpoint').glob('step*') if p.is_dir() and p.name[4:].isdigit()}
    common = sorted(steps(soft) & steps(fixed) - {0,5})
    if not common:
        raise ValueError('No common saved update count')
    # Predeclared early + final common checkpoint; never choose using scores.
    selected = sorted({40 if 40 in common else common[0], common[-1]})
    checkpoints = [(f'{mode}-seed42-{step}',mode,step,run/'checkpoint'/f'step{step}')
                   for step in selected for mode,run in [('soft',soft),('fixed',fixed)]]
    build_eval(work,'eval-new',checkpoints,cfg['deadline']-600)
    write(work/'matched-evaluation.json',dict(steps=selected,selection='early40-or-first and last common; not score-selected'))


def prepare(work, manager, started, expected_host):
    if socket.gethostname() != expected_host or expected_host == 'tungks-0-0':
        raise ValueError('Wrong host; do not deploy on old campaign host')
    deadline = started+8*3600
    if started > time.time()+5 or time.time()-started > 3600:
        raise ValueError('Check original new-allocation clock; more than one hour spent in setup')
    if (work/'campaign.json').exists():
        raise ValueError('Campaign already prepared; inspect existing queue, never create duplicate')
    work.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Dirty source')
    listing = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus = {}
    for line in listing.splitlines():
        index,uid,name,used = [v.strip() for v in line.split(',')]
        if index in ('0','1'):
            if 'B200' not in name or float(used)>1024:
                raise ValueError('Expected two idle B200 GPUs; no stopping unrelated workloads')
            gpus[index] = uid
    if len(gpus)!=2:
        raise ValueError('Two B200 GPUs required')
    execute(['bash',HOST_WRAPPER,'-c',"import torch,transformers,sglang,ray; assert torch.cuda.is_available(); print('RUNTIME_IMPORT_PASS',torch.__version__,transformers.__version__,flush=True)"],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus['1']))
    cfg=json.loads((OLD_WORK/'config.json').read_text())
    for key in ('student','teacher','dataset'):
        if not Path(cfg[key]).exists():
            raise ValueError('Missing input '+key)
    energy=BASE/'partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt'
    if sha(energy)!=ENERGY_SHA:
        raise ValueError('Unexpected learned energy')
    shutil.copy2(energy,work/'energy-select-4.pt')
    # Existing immutable step200: random must have advanced beyond it.
    import re
    with Path(str(OLD_RANDOM)+'.log').open('rb') as stream:
        stream.seek(0,2); stream.seek(max(0,stream.tell()-262144))
        found=re.findall(r'completed_optimizer_updates:\s*([0-9.]+)',stream.read().decode(errors='replace'))
    if not found or float(found[-1])<=200:
        raise ValueError('Random checkpoint200 not proven past writer stage')
    old_checkpoints=[(f'{mode}-seed43-200',mode,200,run/'checkpoint/step200')
                     for mode,run in [('random',OLD_RANDOM),('fixed',OLD_FIXED),('atomic',OLD_ATOMIC)]]
    # Add the old soft pilot only if a safe, same-training-seed baseline exists.
    # Missing optional artifacts are recorded, not turned into negative evidence.
    optional={'old_soft':'not available with a matched saved baseline'}
    try:
        E,_,_=eval_modules()
        template=E.read_json(Path(cfg['eval_template']))
        soft_parent=BASE/'partition-qualify-fix-IcuIufCf/work/soft-pilot'
        logs=list(soft_parent.glob('*.log'))
        if len(logs)==1:
            old_run=Path(str(logs[0])[:-4])
            raw=logs[0].read_text(errors='replace')
            latest=float(re.findall(r'completed_optimizer_updates:\s*([0-9.]+)',raw)[-1])
            bases=[j for j in template['jobs'] if j['mode']=='fixed']
            if bases:
                baseline_root=Path(bases[0]['checkpoint']['path']).parent
                steps=[int(cp.name[4:]) for cp in (old_run/'checkpoint').glob('step*')
                       if cp.name[4:].isdigit() and 0<int(cp.name[4:])<latest
                       and (baseline_root/cp.name).is_dir()]
                if steps:
                    step=max(steps)
                    old_launch=json.loads((old_run/'launch-config.json').read_text())
                    baseline_run=baseline_root.parent
                    base_launch=json.loads((baseline_run/'launch-config.json').read_text())
                    if old_launch['options']['seed']!=base_launch['options']['seed']:
                        raise ValueError('Old soft baseline seed mismatch')
                    seed=old_launch['options']['seed']
                    old_checkpoints += [(f'oldsoft-seed{seed}-{step}','soft',step,old_run/'checkpoint'/f'step{step}'),
                                        (f'oldfixed-seed{seed}-{step}','fixed',step,baseline_root/f'step{step}')]
                    optional={'old_soft':'included','step':step,'training_seed':seed}
    except (OSError,ValueError,KeyError,IndexError) as exc:
        optional={'old_soft':'skipped','reason':str(exc)}
    write(work/'optional-artifacts.json',optional)
    build_eval(work,'eval-old',old_checkpoints,min(deadline-1800,started+2*3600))
    sys.path.insert(0,str(manager))
    from job_manager.store import initialize,connect,submit
    import tempfile
    state=Path(tempfile.mkdtemp(prefix='simct-borrow8-',dir='/var/tmp'))
    config=dict(gpus=list(gpus.values()),cpu_slots=16,deadline=deadline,sample_seconds=5,poll_seconds=1,
                kill_grace_seconds=10,low_util_percent=5,max_external_memory_mib=1024,
                max_admission_util_percent=5,lease_dir=f'/tmp/job-manager-{os.getuid()}-gpu-leases')
    initialize(state,config)
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(work/'campaign.json',dict(started=started,deadline=deadline,training_stop_at=started+5.5*3600,
          host=expected_host,state=str(state),manager=str(manager),source_commit=source_commit,
          energy_sha256=ENERGY_SHA,scope='Borrowed 2xB200 eight-hour allocation; old campaign unchanged'))
    common=dict(MP_STUDENT_PATH=cfg['student'],MP_TEACHER_PATH=cfg['teacher'],MP_DATASET_PATH=cfg['dataset'],
                MP_ENERGY_CHECKPOINT=str(work/'energy-select-4.pt'),MP_SEED='42',MP_PARTITION_SEED='43',
                MP_MAX_SPAN_LENGTH='2',MP_FIXED_SPAN_LENGTH='2',MP_TRAIN_STOP_AT=str(started+5.5*3600),
                MP_CHECKPOINT_RESERVE_SECONDS='300',MP_ALGORITHM='mp_opd',MP_PREFLIGHT_ONLY='0')
    jobs=[]
    def job(key,argv,gpu=None,deps=(),policy='success',end=None,env=None):
        jobs.append(dict(version=1,id=key,project='simct-borrow8',argv=[str(x) for x in argv],cwd=str(ROOT),
            gpus=[gpus[str(gpu)]] if gpu is not None else 0,cpu_slots=4 if gpu is not None else 1,
            timeout_seconds=max(1,(end or deadline-300)-time.time()),deadline=end or deadline-300,
            dependencies=list(deps),dependency_policy=policy,env=env or {},
            metadata=dict(source_commit=source_commit,scope='technical/exploratory; not exact reproduction')))
    adapter=lambda action:[sys.executable,SELF,action,'--work',work]
    job('numeric',['bash',HOST_WRAPPER,SELF,'numeric','--work',work],1,end=started+1800)
    job('soft-canary',['bash',ROOT/'experiments/runai/run_single_gpu.sh','1','soft','5'],1,['numeric'],
        end=started+3600,env=common|dict(MP_RUN_ROOT=str(work/'soft-canary')))
    job('validate-canary',adapter('validate-canary'),deps=['soft-canary'],end=started+3700)
    job('soft-train',['bash',ROOT/'experiments/runai/run_single_gpu.sh','1','soft','156'],1,['validate-canary'],
        end=started+5.5*3600,env=common|dict(MP_RUN_ROOT=str(work/'soft-train')))
    job('old-eval',adapter('eval')+['--name','eval-old','--gpu','0'],0,end=started+2*3600)
    job('old-score',adapter('score')+['--name','eval-old'],end=deadline-600)
    job('fixed-train',['bash',ROOT/'experiments/runai/run_single_gpu.sh','0','fixed','156'],0,['old-eval'],policy='terminal',
        end=started+5.5*3600,env=common|dict(MP_RUN_ROOT=str(work/'fixed-train')))
    job('new-eval-plan',adapter('new-eval-plan'),deps=['soft-train','fixed-train'],end=deadline-1200)
    for slot in (0,1):
        job(f'new-eval-{slot}',adapter('eval')+['--name','eval-new','--gpu',str(slot)],slot,['new-eval-plan'],end=deadline-600)
    job('new-score',adapter('score')+['--name','eval-new'],deps=['new-eval-plan'],end=deadline-600)
    for entry in jobs:
        if entry['id'] in ('old-score','new-score'):
            entry['cpu_slots']=4
    job('collect',adapter('collect'),deps=[j['id'] for j in jobs],policy='terminal',end=deadline)
    db=connect(state)
    try: submit(db,jobs)
    finally: db.close()
    write(work/'jobs.json',jobs)
    print('CAMPAIGN_READY',str(work/'campaign.json'),flush=True)


def collect(work):
    E,D,Q=eval_modules()
    for path in work.glob('eval-*/plan.json'):
        try: Q.summarize(argparse.Namespace(plan=path))
        except Exception as exc: write(path.parent/'summary-error.json',dict(error=type(exc).__name__,detail=str(exc)))
    cfg=json.loads((work/'campaign.json').read_text())
    state=Path(cfg['state'])
    with sqlite3.connect(state/'queue.sqlite3') as src, sqlite3.connect(work/'queue-snapshot.sqlite3') as dst:
        src.backup(dst)
    shutil.copytree(state/'logs',work/'manager-job-logs',dirs_exist_ok=True)
    print('ARTIFACTS_COLLECTED',work,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','numeric','validate-canary','new-eval-plan','eval','score','collect'])
    p.add_argument('--work',type=Path,required=True);p.add_argument('--manager',type=Path)
    p.add_argument('--started',type=float);p.add_argument('--expected-host',default='tungdd11-sparse-vllm-core-0-0')
    p.add_argument('--name');p.add_argument('--gpu',type=int)
    a=p.parse_args();work=a.work.resolve()
    if a.action=='prepare':prepare(work,a.manager.resolve(),a.started,a.expected_host)
    elif a.action=='numeric':numeric_gate(work)
    elif a.action=='validate-canary':validate_canary(work)
    elif a.action=='new-eval-plan':prepare_new_eval(work)
    elif a.action=='collect':collect(work)
    elif a.action=='score':
        execute([sys.executable,ROOT/'scripts/evaluation/eval_queue.py','score-spool','--plan',work/a.name/'plan.json',
                 '--score-workers','4','--internal-code-execution'])
    else:
        execute([sys.executable,ROOT/'scripts/evaluation/eval_queue.py','worker','--plan',work/a.name/'plan.json',
                 '--gpu',a.gpu,'--phase','generate','--concurrency','256',
                 '--score-buffer','256','--internal-code-execution'])
