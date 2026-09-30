#!/usr/bin/env python3
"""Bounded Linux DAG runner. Commands are argv arrays, never shell interpolation."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, time
from pathlib import Path

def write(p, v):
    t=p.with_suffix(".tmp"); t.write_text(json.dumps(v,indent=2)); t.replace(p)
def proc(pid):
    try:
        fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return int(fields[1]),fields[19]
    except (OSError,ValueError): return None
def descendants(known):
    rows={int(p.name):proc(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()}
    changed=True
    while changed:
        changed=False
        for pid,v in rows.items():
            if v and v[0] in known and pid not in known:
                known[pid]=v[1];changed=True
    return known
def terminate(known):
    descendants(known)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,start in reversed(list(known.items())):
            v=proc(pid)
            if v and v[1]==start:
                try: os.kill(pid,sig)
                except ProcessLookupError: pass
        if sig==signal.SIGTERM: time.sleep(3)
def eligible(job, state):
    required=[state['jobs'].get(k,{}).get('status') for k in job.get('require_success',[])]
    if any(x in ('failed','timeout','blocked') for x in required):return 'blocked'
    if any(x!='completed' for x in required):return 'waiting'
    deps=[state['jobs'].get(k,{}).get('status') for k in job.get('after',[])]
    if job.get('after_terminal') and all(x in ('completed','failed','timeout','blocked') for x in deps): return 'ready'
    if any(x in ('failed','timeout','blocked') for x in deps): return 'blocked'
    return 'ready' if all(x=='completed' for x in deps) else 'waiting'
def external_ready(root, key):
    path=root/'admission.json'
    if not path.exists():return True
    rule=json.loads(path.read_text()).get(key)
    if not rule:return True
    other=Path(rule['state'])
    if not other.exists():return False
    state=json.loads(other.read_text())
    return state.get('jobs',{}).get(rule['job'],{}).get('status') in ('completed','failed','blocked','timeout')
def run(path):
    plan=json.loads(path.read_text()); root=path.parent
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    lock=(root/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=root/'campaign-state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(plan_sha256=digest,started=time.time(),jobs={})
    if state['plan_sha256']!=digest: raise ValueError('immutable plan changed')
    # Restart never duplicates a live worker or silently restarts partial training.
    for key,row in state['jobs'].items():
        if row['status']=='running':
            v=proc(row['pid'])
            if v and v[1]==row['start_ticks']: raise RuntimeError('worker still live: '+key)
            row.update(status='failed',reason='manager interrupted; inspect checkpoint before retry')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['source'],text=True).strip()
    if current!=plan['source_commit']:raise ValueError('source commit changed')
    if plan.get('config_sha256') and hashlib.sha256((root/'config.json').read_bytes()).hexdigest()!=plan['config_sha256']:raise ValueError('config changed')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=plan['source'],check=True)
    deadline=state['started']+plan['hours']*3600
    active={}; stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping and time.time()<deadline:
            for key,item in list(active.items()):
                job,p,log,lease,known,end=item;descendants(known)
                rc=p.poll(); timed=time.time()>=end
                if rc is not None or timed:
                    terminate(known);p.wait();log.close()
                    if lease: lease.close()
                    state['jobs'][key].update(status='timeout' if timed else ('completed' if rc==0 else 'failed'),returncode=p.returncode,ended=time.time())
                    del active[key];write(statepath,state)
            for job in plan['jobs']:
                key=job['id']
                if key in state['jobs']:continue
                ready=eligible(job,state)
                if ready=='blocked':state['jobs'][key]={'status':'blocked','reason':'dependency failed'};continue
                if ready!='ready':continue
                if not external_ready(root,key):continue
                gpu=job.get('gpu')
                if any(x[0].get('gpu')==gpu for x in active.values()):continue
                if time.time()+job['budget_seconds']>deadline:
                    state['jobs'][key]={'status':'blocked','reason':'insufficient remaining wall budget'};continue
                lease=None
                if gpu is not None:
                    lease=open(f'/tmp/simct-eval-gpu{gpu}.lock','a')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    if used>=1024:lease.close();continue
                    # Eval worker holds this lock itself. Release immediately before it launches;
                    # its own lock and GPU check arbitrate other queues.
                    if job.get('self_locks'):lease.close();lease=None
                env=dict(os.environ,**job.get('env',{}))
                if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env['CAMPAIGN_DEADLINE']=str(deadline)
                log=(root/(key+'.log')).open('ab')
                p=subprocess.Popen(job['argv'],cwd=plan['source'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=proc(p.pid)[1]
                state['jobs'][key]=dict(status='running',pid=p.pid,start_ticks=start,started=time.time(),gpu=gpu)
                active[key]=(job,p,log,lease,{p.pid:start},time.time()+job['budget_seconds'])
                write(statepath,state)
            write(statepath,state)
            if len(state['jobs'])==len(plan['jobs']) and not active:break
            time.sleep(2)
    finally:
        for key,(job,p,log,lease,known,end) in active.items():
            terminate(known);p.wait();log.close()
            if lease:lease.close()
            state['jobs'][key].update(status='timeout',reason='campaign stopped or deadline',ended=time.time())
        for job in plan['jobs']:
            state['jobs'].setdefault(job['id'],dict(status='blocked',reason='campaign ended before admission'))
        state['ended']=time.time();write(statepath,state)
        for ep in root.glob('eval-*/plan.json'):
            with ep.with_name('summary.json').open('w') as f:
                subprocess.run(['/usr/bin/python3.12',str(Path(plan['source'])/'scripts/evaluation/eval_queue.py'),'summarize','--plan',str(ep)],stdout=f,timeout=60)
        (root/'REPORT.md').write_text('# Campaign result\n\n'+ '\n'.join(f"- {k}: {v['status']} ({v.get('reason','see job log')})" for k,v in state['jobs'].items()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('plan',type=Path);run(a.parse_args().plan.resolve())
