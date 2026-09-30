"""Six-GPU continuation: immutable per-checkpoint plans and a shared work pool.

Run init on the borrowed node, then submit on each node. No existing queue is modified.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
WORK = BASE/'six-b200-v1'
OWNER = 'tungks-0-1'
BORROWED = 'tungdd11-sparse-vllm-core-0-0'
STEPS = (40, 80, 120, 156, 200, 240, 280, 312)
TRAIN = {'soft42': ('soft', 42), 'soft43': ('soft', 43), 'random43': ('random', 43)}
sys.path.insert(0, str(ROOT/'experiments/runai'))
import borrowed_campaign as B


def config():
    value = json.loads((WORK/'campaign.json').read_text())
    if value['source'] != str(ROOT):
        raise ValueError('Use the original immutable campaign source')
    return value


def now_deadline():
    return min(float(os.environ.get('JM_DEADLINE', time.time()+72*3600)),
               config()['borrowed_until'] if socket.gethostname() == BORROWED else float('inf')) - 45


def init(hours):
    if socket.gethostname() != BORROWED:
        raise ValueError('Anchor allocation clock on '+BORROWED)
    if hours != 12:
        raise ValueError('This campaign is authorized for 12 borrowed hours')
    WORK.mkdir(exist_ok=True)
    E, D, Q = B.eval_modules()
    with Q.locked(WORK/'init.lock'):
        if (WORK/'campaign.json').exists():
            c = config()
            print('CAMPAIGN_EXISTS_DEADLINE_UNCHANGED', c['borrowed_until']); return
        # Capture node time BEFORE setup. Retrying initialization never resets it.
        clock = WORK/'allocation-clock.json'
        if not clock.exists():
            started=time.time()
            B.write(clock,dict(host=socket.gethostname(),started=started,until=started+hours*3600))
        allocation=E.read_json(clock);deadline=allocation['until']
        if deadline-time.time()<1800:raise ValueError('Allocation already expired or almost over')
        # Reuse the successful pilot launch contract, not the old eight-hour spec.
        launches = list((BASE/'owner-tungks-0-1/soft50-followup/runs').glob('*/launch-config.json'))
        if len(launches) != 1:
            raise ValueError('Expected one soft50 launch')
        launch = E.read_json(launches[0]); run = launches[0].parent
        summary = E.read_json(run/'checkpoint/run-summary.json')
        if summary['status'] != 'completed' or summary['optimizer_updates'] != 50:
            raise ValueError('Soft50 qualification incomplete')
        if Path(str(run)+'.exitcode').read_text().strip() != '0':
            raise ValueError('Soft50 exit failed')
        energy = BASE/'owner-tungks-0-1/energy-select-4.pt'
        if B.sha(energy) != launch['energy_sha256'] or launch['energy_sha256'] != B.ENERGY_SHA:
            raise ValueError('Energy drift')
        template = E.read_json(B.BASE/'endpoint-recovery-y8cQZzRW/work/eval-endpoints/plan.json')
        if template['seeds'] != [42, 43, 44] or set(template['data']) != set(E.CAPS):
            raise ValueError('Evaluation seed/benchmark mismatch')
        for data in template['data'].values():
            if E.file_hash(data['path']) != data['sha256']:
                raise ValueError('Dataset drift')
        qualification = Q.preflight('/usr/bin/python3.12')
        B.write(WORK/'template.json', template)
        B.write(WORK/'campaign.json', dict(source=str(ROOT),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            borrowed_until=deadline, owner_deadline=None, created=time.time(),allocation=allocation,
            pilot=str(launches[0]), energy=str(energy), qualification=qualification,
            eval_seeds=[42,43,44], steps=list(STEPS), concurrency=256,
            scope='Two training seeds; frozen pilot energy; no oracle efficacy claim'))
        from datetime import timezone, timedelta
        print('CAMPAIGN_READY', WORK)
        for key in ('started','until'):
            stamp=datetime.fromtimestamp(allocation[key],timezone.utc)
            print(key,'UTC',stamp.isoformat(),'VIETNAM',stamp.astimezone(timezone(timedelta(hours=7))).isoformat())


def train_env(key, slot, c):
    launch = json.loads(Path(c['pilot']).read_text()); opts = launch['options']
    mode, seed = TRAIN[key]
    env = dict(MP_STUDENT_PATH=opts['student_name_or_path'],
        MP_TEACHER_PATH=opts['teacher_name_or_path'], MP_DATASET_PATH=opts['train_dataset_path'],
        MP_SEED=str(seed), MP_PARTITION_SEED=str(opts['mp_opd_random_seed']),
        MP_MAX_SPAN_LENGTH=str(opts['mp_opd_max_span_length']),
        MP_FIXED_SPAN_LENGTH=str(opts['mp_opd_fixed_span_length']),
        MP_ALGORITHM='mp_opd', MP_ATTN_IMPLEMENTATION='eager', MP_PREFLIGHT_ONLY='0',
        MP_RUN_ROOT=str(WORK/'train'/key), MP_PARITY_CAPTURE_DIR=str(WORK/'captures'/key))
    if mode == 'soft': env['MP_ENERGY_CHECKPOINT'] = c['energy']
    if key == 'random43':
        env.update(MP_TRAIN_STOP_AT=str(c['borrowed_until']-900), MP_CHECKPOINT_RESERVE_SECONDS='300')
    return env


def train(key, slot):
    target = WORK/'train'/key; target.mkdir(parents=True, exist_ok=True)
    if key=='random43' and config()['borrowed_until']-time.time()<7*3600:
        B.write(target/'finished.json',dict(status='skipped',reason='less than seven hours for full replay'))
        print('TRAIN_SKIPPED random43: insufficient admission budget',flush=True)
        return
    # Refuse implicit restart from SFT if a previous attempt has artifacts.
    if list(target.glob('*/launch-config.json')):
        raise ValueError('Run already exists; inspect before retry')
    c = config(); env = dict(os.environ, **train_env(key,slot,c))
    # Prevent inherited deadline/preflight knobs from an old interactive shell.
    if key != 'random43':
        env.pop('MP_TRAIN_STOP_AT', None); env.pop('MP_CHECKPOINT_RESERVE_SECONDS', None)
    result = subprocess.run(['bash',str(ROOT/'experiments/runai/run_single_gpu.sh'),
                             str(slot),TRAIN[key][0],'312'],env=env)
    B.write(target/'finished.json',dict(returncode=result.returncode,time=time.time()))
    if result.returncode: raise SystemExit(result.returncode)


def complete_run(run):
    summary = run/'checkpoint/run-summary.json'; rc = Path(str(run)+'.exitcode')
    return (summary.exists() and rc.exists() and rc.read_text().strip() == '0'
            and json.loads(summary.read_text()).get('status') == 'completed')


def safe_steps(run):
    log = Path(str(run)+'.log')
    latest = 0
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(0,2); stream.seek(max(0,stream.tell()-2*1024*1024))
            updates = re.findall(r'completed_optimizer_updates:\s*([0-9.]+)',stream.read().decode(errors='replace'))
        if updates: latest = int(float(updates[-1]))
    finished = complete_run(run)
    # Strict > also protects epoch-end step156 which is saved after step logging.
    return [s for s in STEPS if (run/'checkpoint'/f'step{s}').is_dir() and (finished or latest>s)]


def sources():
    result = []
    for label, parent, mode, seed in [
        ('random42-complete',BASE/'work/random42-full312','random',42),
        ('simct43-complete',BASE/'simct43-followup/runs','simct',43),
        *[(k,WORK/'train'/k,m,s) for k,(m,s) in TRAIN.items()]]:
        runs = list(parent.glob('*/launch-config.json'))
        if len(runs)>1: raise ValueError('Ambiguous run '+label)
        if runs:
            run = runs[0].parent; launch = json.loads(runs[0].read_text())
            opts = launch['options']
            algorithm = 'span_ctkd' if mode == 'simct' else 'mp_opd'
            if opts['seed'] != seed or opts['kd_algorithm'] != algorithm:
                raise ValueError('Training identity mismatch '+label)
            if mode != 'simct' and opts['mp_opd_mode'] != mode:
                raise ValueError('Training mode mismatch '+label)
            result.append((label,run,mode,seed))
    return result


def publish_available():
    E,D,Q = B.eval_modules(); c=config()
    pool=WORK/'pool';pool.mkdir(exist_ok=True)
    with Q.locked(WORK/'publisher.lock',blocking=False) as lock:
        if lock is None:return
        for label,run,mode,seed in sources():
            for step in safe_steps(run):
                key=f'{label}-step{step}';target=pool/key
                if (target/'ready.json').exists():continue
                target.mkdir(exist_ok=True)
                identity=B.checkpoint_stable(run/'checkpoint'/f'step{step}')
                plan=E.read_json(WORK/'template.json')
                plan.update(jobs=[dict(id=key,mode=mode,step=step,tier=0,checkpoint=identity)],
                            source=D.script_hashes(),hours=168,admit_hours=168)
                # A manifest per checkpoint stays immutable as new checkpoints arrive.
                if (target/'plan.json').exists() and E.read_json(target/'plan.json') != plan:
                    raise ValueError('Interrupted publication identity drift')
                Q.atomic_json(target/'plan.json',plan)
                Q.atomic_json(target/'state.json',dict(plan_sha256=E.file_hash(target/'plan.json'),
                    started=time.time(),deadline=time.time()+168*3600,admit_until=time.time()+168*3600,
                    jobs={},durations=[],qualification=c['qualification']))
                Q.atomic_json(target/'ready.json',dict(training_seed=seed,mode=mode,step=step,
                    launch=str(run/'launch-config.json'),checkpoint_sha256=identity['sha256'],
                    plan_sha256=E.file_hash(target/'plan.json')))
                print('CHECKPOINT_PUBLISHED',key,flush=True)


def training_done():
    for key in TRAIN:
        if (WORK/'train'/key/'finished.json').exists():continue
        # Use the borrowed node's own clock marker, not an owner's wall clock
        # compared with a timestamp produced on another host.
        if key=='random43' and (WORK/'borrowed-window-ended.json').exists():continue
        return False
    return True


def producers_done():
    return (WORK/'publisher-done.json').exists()


def plans():
    return sorted((WORK/'pool').glob('*/ready.json'))


def cells(path, plan):
    return [path.parent/'cells'/plan['jobs'][0]['id']/b/str(s) for s in plan['seeds'] for b in plan['data']]


def generate(slot):
    E,D,Q=B.eval_modules(); Q.GENERATION_ONLY=True
    # eval_queue uses a physical index; prove it equals the job-manager UUID lease.
    uid=subprocess.check_output(['nvidia-smi','-i',str(slot),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uid:raise ValueError('Physical slot/lease mismatch')
    end=now_deadline()
    while time.time()+600<end:
        pending=False
        for ready in plans():
            path=ready.parent/'plan.json';plan=E.read_json(path);root=path.parent
            if E.file_hash(path)!=E.read_json(ready)['plan_sha256']:raise ValueError('Published plan drift')
            if all((p/'generation-complete.json').exists() for p in cells(path,plan)):continue
            if (root/'generation-error.json').exists():continue
            pending=True
            with Q.locked(root/'generation-dispatch.lock',blocking=False) as claim:
                if claim is None:continue
                if all((p/'generation-complete.json').exists() for p in cells(path,plan)):continue
                if time.time()+600>=end:return
                if plan['source']!=D.script_hashes():raise ValueError('Eval source drift')
                for data in plan['data'].values():
                    if E.file_hash(data['path'])!=data['sha256']:raise ValueError('Evaluation dataset drift')
                args=argparse.Namespace(plan=path,gpu=slot,phase='generate',concurrency=256,
                    score_workers=1,score_buffer=256,score_python='/usr/bin/python3.12',min_free_gib=20)
                try:
                    print('GEN_CLAIM',root.name,socket.gethostname(),slot,flush=True)
                    Q.run_checkpoint(root,plan,E.file_hash(path),plan['jobs'][0],args,end)
                except Q.Deadline:return
                except Exception as exc:
                    B.write(root/'generation-error.json',dict(error=str(exc),host=socket.gethostname()))
                    print('GEN_ERROR',root.name,str(exc),flush=True)
        if not pending and producers_done():return
        time.sleep(15)


def score():
    E,D,Q=B.eval_modules();Q.GENERATION_ONLY=False
    qualification=Q.preflight('/usr/bin/python3.12')
    if qualification!=config()['qualification']:raise ValueError('Scorer qualification drift')
    end=now_deadline()
    while time.time()+150<end:
        pending=False
        for ready in plans():
            path=ready.parent/'plan.json';plan=E.read_json(path);ph=E.file_hash(path)
            if ph!=E.read_json(ready)['plan_sha256']:raise ValueError('Published plan drift')
            if plan['source']!=D.script_hashes():raise ValueError('Eval source drift')
            for seed in plan['seeds']:
                for bench in plan['data']:
                    job=plan['jobs'][0];cell=path.parent/'cells'/job['id']/bench/str(seed)
                    if (cell/'metrics.json').exists() or (cell/'score-error.json').exists():continue
                    if not (cell/'generation-complete.json').exists():
                        if not (path.parent/'generation-error.json').exists():pending=True
                        continue
                    pending=True
                    with Q.locked(cell/'score.lock',blocking=False) as claim:
                        if claim is None:continue
                        if (cell/'metrics.json').exists() or (cell/'score-error.json').exists():continue
                        if time.time()+150>=end:return
                        try:
                            marker=E.read_json(cell/'generation-complete.json');server=marker['contract']['server']
                            expected=Q.cell_contract(ph,job,plan['data'],bench,seed,server)
                            if marker['contract']!=expected or E.file_hash(cell/'responses.jsonl')!=marker['responses_sha256']:
                                raise ValueError('Spool contract drift')
                            if E.file_hash(plan['data'][bench]['path'])!=plan['data'][bench]['sha256']:
                                raise ValueError('Evaluation dataset drift')
                            args=argparse.Namespace(phase='score',concurrency=1,score_workers=4,
                                score_buffer=256,score_python='/usr/bin/python3.12')
                            print('SCORE_CLAIM',job['id'],bench,seed,flush=True)
                            Q.run_cell(path.parent,plan,ph,job,bench,seed,None,server,args,end)
                        except Q.Deadline:return
                        except Exception as exc:
                            B.write(cell/'score-error.json',dict(error=str(exc),host=socket.gethostname(),time=time.time()))
                            print('SCORE_ERROR',job['id'],bench,seed,str(exc),flush=True)
            # Run the secondary scorer on the unlimited node so an allocation
            # cutoff does not interrupt its separate subprocess tree.
            if socket.gethostname()==OWNER and all((p/'metrics.json').exists() for p in cells(path,plan)):
                with Q.locked(path.parent/'lcbfix-dispatch.lock',blocking=False) as claim:
                    output=path.parent/'lcbfix'
                    if claim is not None and not (output/'summary.json').exists() and not (path.parent/'lcbfix-error.json').exists():
                        if time.time()+300>=end:return
                        try:
                            subprocess.run([sys.executable,str(ROOT/'scripts/evaluation/lcbfix.py'),
                                '--plan',str(path),'--out',str(output),'--workers','4','--internal-code-execution'],
                                check=True,timeout=min(7200,end-time.time()))
                        except Exception as exc:B.write(path.parent/'lcbfix-error.json',dict(error=str(exc)))
        if not pending and producers_done():return
        time.sleep(15)


def specs(host,gpus,c):
    owner=host==OWNER; jobs=[]
    def add(key,action,slot=None,deps=(),cpu=1,policy='success'):
        value=dict(version=1,id=key,project='six-b200-v1',cwd=str(ROOT),
            argv=[sys.executable,str(Path(__file__).resolve()),action],env={},
            gpus=0 if slot is None else [gpus[slot]],cpu_slots=cpu,timeout_seconds=72*3600,
            dependencies=list(deps),dependency_policy=policy)
        if not owner:value['deadline']=c['borrowed_until']
        jobs.append(value);return value
    if owner:
        add('publisher','publish')
        for slot,key in enumerate(('soft42','soft43')):
            j=add(key,'train',slot,cpu=8);j['argv']+=['--key',key,'--slot',str(slot)]
            j=add('gen-'+str(slot),'gen',slot,[key],4,'terminal');j['argv']+=['--slot',str(slot)]
    else:
        add('allocation-clock','clock')
        j=add('random43','train',0,cpu=8);j['argv']+=['--key','random43','--slot','0']
        # The payload checks a seven-hour admission budget. Once admitted it
        # retains the entire allocation, with cooperative checkpoint reserve.
        j.update(timeout_seconds=12*3600)
        for slot in range(4):
            j=add('gen-'+str(slot),'gen',slot,['random43'] if slot==0 else [],4,'terminal')
            j['argv']+=['--slot',str(slot)]
    for i in range(2 if owner else 4):add('score-'+str(i),'score',cpu=4)
    return jobs


def submit():
    host=socket.gethostname()
    if host not in (OWNER,BORROWED):raise ValueError('Wrong host')
    c=config();E,D,Q=B.eval_modules();manager=BASE/'job-manager'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=c['commit']:
        raise ValueError('Source commit changed')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Dirty source')
    if not Path('/opt/venvs/simct-b200/bin/python').is_file():
        raise ValueError('Restore the already-qualified runtime on this node first')
    subprocess.run(['bash',str(B.HOST_WRAPPER),'-c',
        'import torch,transformers,sglang,ray; print("RUNTIME_IMPORT_PASS",torch.__version__,transformers.__version__)'],
        check=True,timeout=120)
    sys.path.insert(0,str(manager))
    from job_manager.store import initialize,connect,submit as put,rows
    from job_manager.__main__ import snapshot
    receipt=WORK/(host+'.json')
    with Q.locked(WORK/(host+'.submit.lock')):
        if receipt.exists():
            rec=E.read_json(receipt);state=Path(rec['state'])
        else:
            if host==BORROWED and time.time()>=c['borrowed_until']-1800:raise ValueError('Allocation expired')
            count=2 if host==OWNER else 4
            listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
            inventory={int(a.strip()):(b.strip(),name.strip()) for a,b,name in (line.split(',') for line in listing.splitlines())}
            if any(i not in inventory or 'B200' not in inventory[i][1] for i in range(count)):
                raise ValueError('Expected B200 inventory changed')
            gpus={i:inventory[i][0] for i in range(count)}
            # Occupied GPUs are left to the scheduler's idle+lease admission; never killed.
            state=Path(tempfile.mkdtemp(prefix='simct-six-',dir='/var/tmp'))
            cfg=dict(gpus=list(gpus.values()),cpu_slots=40 if host==OWNER else 48,
                deadline=None if host==OWNER else c['borrowed_until'],sample_seconds=5,poll_seconds=1,
                kill_grace_seconds=15,low_util_percent=5,max_external_memory_mib=1024,
                max_admission_util_percent=5,lease_dir=f'/tmp/job-manager-{os.getuid()}-gpu-leases')
            initialize(state,cfg)
            rec=dict(state=str(state),source=str(ROOT),host=host,gpus=gpus,specs=specs(host,gpus,c))
            B.write(receipt,rec)
        db=connect(state)
        try:
            existing={j['id'] for j in rows(db)}
            pending=[s for s in rec['specs'] if s['id'] not in existing]
            if pending:put(db,pending)
            if not snapshot(db)['manager']['running']:
                with (WORK/(host+'.manager.log')).open('ab') as out:
                    proc=subprocess.Popen([sys.executable,'-m','job_manager','--state',str(state),'run'],
                        cwd=manager,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(2)
                if proc.poll() is not None:raise RuntimeError('Manager exited; inspect manager log')
            print(json.dumps(snapshot(db),indent=2));print('RECEIPT',receipt)
        finally:db.close()


def status():
    E,D,Q=B.eval_modules();counts={'plans':0,'gen':0,'score':0,'errors':0,'lcbfix':0}
    for ready in plans():
        path=ready.parent/'plan.json';plan=E.read_json(path);paths=cells(path,plan)
        g=sum((p/'generation-complete.json').exists() for p in paths)
        s=sum((p/'metrics.json').exists() for p in paths)
        errors=sum((p/'score-error.json').exists() for p in paths)+int((path.parent/'generation-error.json').exists())
        counts['plans']+=1;counts['gen']+=g;counts['score']+=s;counts['errors']+=errors
        counts['lcbfix']+=int((path.parent/'lcbfix/summary.json').exists())
        print(path.parent.name,'gen',g,'score',s,'errors',errors)
    print('POOL',json.dumps(counts));print('PRODUCERS_DONE',producers_done())
    print('EXPECTED_FULL_CURVES',40,'EXPECTED_FULL_CELLS',480)
    for label,run,mode,seed in sources():
        published=[s for s in STEPS if (WORK/'pool'/f'{label}-step{s}'/'ready.json').exists()]
        print('RUN',label,'train_seed',seed,'published',published,
              'missing',[s for s in STEPS if s not in published],'completed',complete_run(run))
    print('NODE_UTC',datetime.now(timezone.utc).isoformat())
    heartbeat=WORK/'borrowed-clock.json'
    if heartbeat.exists():
        value=E.read_json(heartbeat)
        print('BORROWED_CLOCK',value,'OWNER_OR_LOCAL_MINUS_HEARTBEAT_SECONDS',time.time()-value['now'])
    sys.path.insert(0,str(BASE/'job-manager'))
    from job_manager.store import connect
    from job_manager.__main__ import snapshot
    receipt=WORK/(socket.gethostname()+'.json')
    if receipt.exists():
        db=connect(Path(E.read_json(receipt)['state']))
        try:print(json.dumps(snapshot(db),indent=2))
        finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','submit','publish','train','gen','score','status','clock'])
    p.add_argument('--hours',type=int,default=12);p.add_argument('--key',choices=TRAIN);p.add_argument('--slot',type=int)
    a=p.parse_args()
    if a.action=='init':init(a.hours)
    elif a.action=='submit':submit()
    elif a.action=='train':train(a.key,a.slot)
    elif a.action=='gen':generate(a.slot)
    elif a.action=='score':score()
    elif a.action=='status':status()
    elif a.action=='clock':
        end=config()['borrowed_until']
        while time.time()<end-90:
            B.write(WORK/'borrowed-clock.json',dict(host=socket.gethostname(),now=time.time(),until=end))
            time.sleep(5)
        B.write(WORK/'borrowed-window-ended.json',dict(host=socket.gethostname(),time=time.time(),until=end))
    else:
        while True:
            publish_available()
            if training_done():
                # Final scan above precedes the completion marker, avoiding early
                # dispatcher exit while the final checkpoint is being published.
                B.write(WORK/'publisher-done.json',dict(time=time.time()))
                break
            time.sleep(20)


if __name__=='__main__':main()
