"""Two-host full alternating campaign with shared, committed export consumption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/runai'))
import queue_full_alternating as F

OWNER='hieplh8-beyond-leakage-1-0-0'
EXTRA='nlp-core-team-0-0'
EXTRA_GPUS=['GPU-f6156669-ead8-55b2-53b3-15cfd97aaaa7',
 'GPU-e8c8d5ba-f298-c8fd-f775-cacee381e0e5','GPU-f0e8ff06-ff72-5da4-5df7-ac29566a0a07',
 'GPU-660da293-fe99-a434-9b50-88c8c88e0546','GPU-c9f4c8b4-a5b4-e7f8-a7b2-0f0645b6a89f']
SELF=Path(__file__).resolve()
TERMINAL={'completed','failed','blocked','cancelled','timeout','lost'}


def specs(case,host,gpus,qualification_policy='required',qualification_timeout=3600):
    if host not in (OWNER,EXTRA):raise ValueError('Unknown host')
    if len(gpus) not in ((1,) if host==OWNER else (4,5)):raise ValueError('Wrong GPU pool')
    if qualification_policy not in ('required','advisory','skip'):raise ValueError('Invalid qualification policy')
    prefix='altsplit-'+hashlib.sha256((str(case)+host).encode()).hexdigest()[:10]
    jobs=[]
    def add(label,action,deps=(),gpu=None,run=None):
        env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
             'PYTHONPATH':str(ROOT/'experiments/modal/vendor')+':'+str(ROOT)}
        if gpu is not None:
            env.update(KDFLOW_ROLLOUT_PORT_BASE=str(18000+1000*gpu),
                       KDFLOW_ROUTER_PORT_BASE=str(23000+1000*gpu),
                       KDFLOW_ROUTER_PROMETHEUS_PORT=str(28000+gpu))
        argv=['bash',str(F.WRAPPER),'-u',str(SELF),action,'--case',str(case)]
        if run:argv+=['--run',run]
        job=dict(version=1,id=prefix+'-'+label,project='full-alternating-split',cwd=str(ROOT),
                 argv=argv,env=env,gpus=0 if gpu is None else [gpus[gpu]],cpu_slots=1,
                 timeout_seconds=qualification_timeout if action=='qualify' else 7*24*3600,
                 dependencies=list(deps),dependency_policy='terminal' if qualification_policy in ('advisory','skip') else 'success')
        jobs.append(job);return job['id']
    audit=add('audit','audit')
    qualified=audit if qualification_policy=='skip' else add('qualify','qualify',[audit],0)
    if host==OWNER:
        # One GPU: the owner walks its seeds' variants in sequence, then generates.
        previous=qualified
        for seed in F.OWNER_SEEDS:
            for name,_,_ in F.VARIANTS:
                previous=add(f'ALT-{name}-s{seed}','train',[previous],0,f'ALT-{name}-s{seed}')
        add('gen-0','generate',[previous],0)
    else:
        # Many GPUs: one variant per GPU, its seeds one after another, then generate on it.
        for i,(name,_,_) in enumerate(F.VARIANTS):
            previous=qualified
            for seed in F.EXTRA_SEEDS:
                run=f'ALT-{name}-s{seed}'
                previous=add(run,'train',[previous],i,run)
            add(f'gen-{i}','generate',[previous],i)
        for i in range(len(F.VARIANTS),len(gpus)):add(f'gen-{i}','generate',[audit],i)
    for i in range(2 if host==OWNER else 4):add(f'score-{i}','score',[audit])
    add('state-export','export')
    return jobs


def initialize_case(case,qualification_policy='required',extra_gpu_count=5,qualification_timeout=3600):
    import borrowed_campaign as B
    E,D,Q=B.eval_modules()
    case.mkdir(parents=True,exist_ok=True)
    with Q.locked(case/'submission.lock'):
        if not (case/'campaign.json').exists():
            template=F.read(F.BASE/'six-b200-v1/template.json')
            if template['seeds']!=[42,43,44] or set(template['data'])!=set(E.CAPS):raise ValueError('Eval protocol differs')
            shared=F.BASE.parent/'SimCT'
            c=dict(source=str(ROOT),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                runs=F.configurations(),steps=list(F.STEPS),streaming_eval=True,
                meta_policy='uniform-normalized-group-original-pair-v3',
                qualification_policy=qualification_policy,extra_gpu_count=extra_gpu_count,
                qualification_timeout_seconds=qualification_timeout,
                student=str(shared/'runs/qwen-gemma-sft-paper-20260908-045828/checkpoint'),
                teacher='/workspace/storage-shared/models/Qwen2.5-7B-Instruct',
                dataset=str(shared/'data/qwen-author/data/prompts.parquet'),
                meta=str(shared/'data/qwen-author/data/selected.parquet'),
                energy=str(F.BASE/'owner-tungks-0-1/energy-select-4.pt'))
            paths={d['path']:d['sha256'] for d in template['data'].values()}
            for name in ('dataset','meta','energy'):paths[c[name]]=B.sha(Path(c[name]))
            if paths[c['energy']]!=B.ENERGY_SHA:raise ValueError('Wrong energy initialization')
            c['input_hashes']=paths
            F.write(case/'eval-template.json',template);F.write(case/'campaign.json',c)
        c=F.checked_config(case)
        if not c.get('streaming_eval'):raise ValueError('Use a fresh split campaign directory')
        if (c.get('qualification_policy','required')!=qualification_policy or
            c.get('extra_gpu_count',5)!=extra_gpu_count or
            c.get('qualification_timeout_seconds',3600)!=qualification_timeout):
            raise ValueError('Immutable scheduling policy differs; use a fresh case')


def manager_state(args,gpus):
    from queue_alternating_followup import active_states
    from job_manager.store import initialize,connect,setting
    from job_manager.__main__ import snapshot
    host=socket.gethostname();folder=args.case/'nodes'/host
    folder.mkdir(parents=True,exist_ok=True)
    states=active_states()
    if args.state:state=args.state.resolve()
    elif len(states)>1:raise ValueError('Multiple managers; select --state explicitly')
    elif states:state=Path(states[0])
    else:state=Path('/var/tmp')/('alt-split-'+hashlib.sha256((str(args.case)+host).encode()).hexdigest()[:16])
    if not str(state).startswith('/var/tmp/'):raise ValueError('Manager must use local /var/tmp state')
    if not (state/'queue.sqlite3').exists():
        initialize(state,dict(gpus=gpus,cpu_slots=16,deadline=None,sample_seconds=5,poll_seconds=1,
            kill_grace_seconds=30,low_util_percent=5,max_external_memory_mib=1024,
            max_admission_util_percent=5,lease_dir=f'/tmp/job-manager-{os.getuid()}-gpu-leases'))
    db=connect(state)
    try:
        config=setting(db,'config')
        if not set(gpus)<=set(config['gpus']) or config['cpu_slots']<(4 if host==OWNER else 10) or config.get('deadline') is not None:
            raise ValueError('Existing manager capacity/deadline incompatible; no settings changed')
        if not snapshot(db)['manager']['running']:
            with (folder/'manager.log').open('ab') as log:
                child=subprocess.Popen(['/usr/bin/python3.12','-m','job_manager','--state',str(state),'run'],
                    cwd=args.manager,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(40):
                if child.poll() is not None:raise ValueError('Manager exited; inspect manager.log')
                if snapshot(db)['manager']['running']:break
                time.sleep(.5)
            else:raise ValueError('Manager startup not confirmed')
    finally:db.close()
    return state


def submit(args):
    import borrowed_campaign as B
    host=socket.gethostname()
    if host not in (OWNER,EXTRA):raise ValueError('Unexpected host')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    cards={int(p[0]):(p[1].strip(),p[2].strip()) for line in inventory.splitlines() if (p:=line.split(','))}
    indices=range(1 if host==OWNER else args.extra_gpu_count)
    if any(i not in cards or cards[i][1]!='NVIDIA B200' for i in indices):raise ValueError('Unexpected GPU inventory')
    gpus=[cards[i][0] for i in indices]
    if host==EXTRA and gpus!=EXTRA_GPUS[:args.extra_gpu_count]:raise ValueError('New-host GPU UUID mismatch')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    initialize_case(args.case,args.qualification_policy,args.extra_gpu_count,args.qualification_timeout_seconds)
    sys.path.insert(0,str(args.manager))
    from job_manager.store import connect,rows,submit as put
    from job_manager.__main__ import snapshot
    _,_,Q=B.eval_modules()
    (args.case/'nodes'/host).mkdir(parents=True,exist_ok=True)
    with Q.locked(args.case/'nodes'/host/'submit.lock'):
        state=manager_state(args,gpus);db=connect(state)
        try:
            snap=snapshot(db)
            if not snap['manager']['running'] or snap['paused'] or snap['quarantined']:raise ValueError('Manager not ready')
            jobs=specs(args.case,host,gpus,args.qualification_policy,args.qualification_timeout_seconds)
            current={j['id']:j for j in rows(db)}
            for j in jobs:
                if j['id'] in current and current[j['id']]['spec']!=j:raise ValueError('Existing spec differs')
            receipt=dict(state=str(state),host=host,manager=str(args.manager),jobs=jobs)
            F.write(args.case/'nodes'/host/'receipt.json',receipt)
            pending=[j for j in jobs if j['id'] not in current]
            if pending:put(db,pending)
            print(json.dumps(snapshot(db),indent=2))
        finally:db.close()


def export_state(case,once=False):
    host=socket.gethostname();dest=case/'nodes'/host
    receipt=F.read(dest/'receipt.json');sys.path.insert(0,receipt['manager'])
    from job_manager.store import connect,rows
    db=connect(Path(receipt['state']))
    wanted={j['id'] for j in receipt['jobs'] if j['argv'][4]!='export'}
    try:
        while True:
            states={j['id']:j['status'] for j in rows(db) if j['id'] in wanted}
            F.write(dest/'status.json',dict(time=time.time(),jobs=states))
            if once or (set(states)==wanted and all(s in TERMINAL for s in states.values())):return
            time.sleep(10)
    finally:db.close()


def producers_done(case):
    for host in (OWNER,EXTRA):
        dest=case/'nodes'/host
        if not (dest/'status.json').exists():return False
        status=F.read(dest/'status.json');receipt=F.read(dest/'receipt.json')
        trains=[j for j in receipt['jobs'] if j['argv'][4]=='train']
        if len(trains)!=F.expected_trains(host==OWNER) or any(status['jobs'].get(j['id']) not in TERMINAL for j in trains):return False
    return True


def retire_unstarted(case, disconnected_meta=False):
    """Retire only this host's pre-training failed campaign, preserving evidence."""
    host=socket.gethostname()
    if host not in (OWNER,EXTRA):raise ValueError('Unexpected host')
    receipt=F.read(case/'nodes'/host/'receipt.json')
    if receipt['host']!=host:raise ValueError('Host mismatch')
    if disconnected_meta:
        c=F.read(case/'campaign.json')
        if c['commit']!='f5d5114e1d7a3c3a29a1901f167968acb5e8b747':
            raise ValueError('Recovery only applies to the verified disconnected-meta source')
    sys.path.insert(0,receipt['manager'])
    from job_manager.store import connect,rows,cancel
    db=connect(Path(receipt['state']))
    try:
        current={j['id']:j for j in rows(db)}
        for spec in receipt['jobs']:
            old=current[spec['id']]
            if old['spec']!=spec:raise ValueError('Existing spec changed')
            if not disconnected_meta and spec['argv'][4]=='train' and old['started'] is not None:
                raise ValueError('Training already started: do not retire this campaign')
            if not disconnected_meta and spec['argv'][4]=='qualify' and old['status'] not in ('failed','blocked','cancelled'):
                raise ValueError('Qualification is not failed/blocked; inspect first')
        for spec in receipt['jobs']:cancel(db,spec['id'])
        wanted={j['id'] for j in receipt['jobs']}
        for _ in range(60):
            active=[j['id'] for j in rows(db) if j['id'] in wanted and j['status'] in ('running','starting','queued')]
            if not active:break
            time.sleep(1)
        else:raise ValueError('Old workers have not stopped; new campaign not submitted: '+str(active))
        record='retired-disconnected-meta.json' if disconnected_meta else 'retired-before-training.json'
        F.write(case/'nodes'/host/record,dict(host=host,time=time.time(),jobs=sorted(wanted),
            invalid_for_alternating_efficacy=disconnected_meta,
            reason='Disconnected FSDP meta gradients; preserve outputs, do not resume them' if disconnected_meta else 'Retired before training'))
        print('OLD_CAMPAIGN_RETIRED',host,flush=True)
    finally:db.close()


def worker(case,phase):
    import fcntl
    from kdflow.export_ready import marker
    # Locks are held throughout a checkpoint operation and released by the OS
    # on process exit. A second host never clears another worker's claim.
    while True:
        pending=False
        for config in F.configurations():
            run=config['id']
            for step in F.STEPS:
                if not marker(case/'train'/run/'checkpoint',step).exists():continue
                folder=case/'dispatch'/f'{run}-step{step}';folder.mkdir(parents=True,exist_ok=True)
                if phase=='score' and not (folder/'generate.done.json').exists():continue
                if (folder/(phase+'.done.json')).exists() or (folder/(phase+'.error.json')).exists():continue
                pending=True
                with (folder/(phase+'.lock')).open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if (folder/(phase+'.done.json')).exists() or (folder/(phase+'.error.json')).exists():continue
                    try:
                        F.evaluate(case,run,step,phase)
                        F.write(folder/(phase+'.done.json'),dict(host=socket.gethostname(),time=time.time()))
                    except Exception as e:
                        traceback.print_exc()
                        F.write(folder/(phase+'.error.json'),dict(host=socket.gethostname(),time=time.time(),error=repr(e)))
        if producers_done(case) and not pending:
            # Score workers must remain available while any generation is in flight.
            ready=[(r['id'],s) for r in F.configurations() for s in F.STEPS
                   if marker(case/'train'/r['id']/'checkpoint',s).exists()]
            if phase=='score' and any(not any((case/'dispatch'/f'{r}-step{s}'/n).exists()
                 for n in ('generate.done.json','generate.error.json')) for r,s in ready):
                time.sleep(10);continue
            if phase=='score':
                done=list((case/'dispatch').glob('*/score.done.json'))
                if len(done)==F.eval_cells():F.report(case)
            if list((case/'dispatch').glob('*/'+phase+'.error.json')):
                raise RuntimeError('Work drained with retained '+phase+' errors; see dispatch markers')
            return
        time.sleep(10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','status','audit','qualify','train','generate','score','export','retire-unstarted','retire-disconnected-meta'])
    p.add_argument('--case',type=Path,required=True);p.add_argument('--run')
    p.add_argument('--manager',type=Path,default=F.BASE/'job-manager');p.add_argument('--state',type=Path)
    p.add_argument('--qualification-policy',choices=['required','advisory','skip'],default='required')
    p.add_argument('--extra-gpu-count',type=int,choices=[4,5],default=5)
    p.add_argument('--qualification-timeout-seconds',type=int,default=3600)
    a=p.parse_args();a.case=a.case.resolve();host=socket.gethostname()
    if a.qualification_timeout_seconds<=0:raise ValueError('Qualification timeout must be positive')
    if a.action=='retire-unstarted':return retire_unstarted(a.case)
    if a.action=='retire-disconnected-meta':return retire_unstarted(a.case,disconnected_meta=True)
    if a.action=='submit':return submit(a)
    if a.action=='status':
        export_state(a.case,once=True)
        for node in (OWNER,EXTRA):
            path=a.case/'nodes'/node/'status.json'
            print(node,json.dumps(F.read(path)) if path.exists() else 'NOT_SUBMITTED')
        print('EVAL_DONE',len(list((a.case/'dispatch').glob('*/score.done.json'))),'/'+str(F.eval_cells()))
        for path in sorted((a.case/'dispatch').glob('*/*.error.json')):print('ERROR',path,F.read(path))
        return
    F.checked_config(a.case)
    qualification=a.case/'qualification'/host
    if a.action=='audit':
        import borrowed_campaign as B
        _,_,Q=B.eval_modules()
        with Q.locked(a.case/'audit.lock'):F.audit(a.case)
    elif a.action=='qualify':F.qualify(a.case,qualification)
    elif a.action=='train':F.train(a.case,a.run,qualification)
    elif a.action=='export':export_state(a.case)
    else:worker(a.case,a.action)


if __name__=='__main__':main()
