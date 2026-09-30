"""Bounded gen handoff, matched seed44 training, then return GPUs to the pool."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import tempfile

import repair_six_gen as R

EXTRA={'fixed44':('fixed',44),'atomic44':('atomic',44),'simct44':('simct',44),
       'random44':('random',44),'soft44':('soft',44)}
BORROW_KEYS=('fixed44','atomic44','simct44')
OWN_KEYS=('random44','soft44')


def modules():
    M=R.load_original();M.TRAIN.update(EXTRA)
    E,D,Q=M.B.eval_modules()
    # New workers use their operational watchdog, never the old allocation cap.
    M.now_deadline=lambda: float(os.environ.get('JM_DEADLINE',time.time()+72*3600))-45
    M.producers_done=lambda: extended_done(M.WORK/'publisher-done.json',M.TRAIN)
    sys.path.insert(0,str(M.BASE/'job-manager'))
    return M,E,D,Q


def extended_done(path,keys):
    return path.exists() and set(json.loads(path.read_text()).get('training_keys',[]))==set(keys)


def terminal_exports(M,db):
    from job_manager.store import rows,TERMINAL
    for job in rows(db):
        if job['id'] in M.TRAIN and job['status'] in TERMINAL:
            path=M.WORK/'train'/job['id']/'finished.json'
            if not path.exists():
                M.B.write(path,dict(queue_status=job['status'],returncode=job['returncode'],
                                   reason=job['reason'],time=time.time()))


def local_db(M,E,old=False):
    from job_manager.store import connect
    receipt=E.read_json(M.WORK/(socket.gethostname()+('.json' if old else '.seed44.json')))
    return connect(Path(receipt['state']))


def cancel_and_wait(db,ids):
    from job_manager.store import rows,cancel
    for j in rows(db):
        if j['id'] in ids and j['status'] in ('queued','starting','running'):cancel(db,j['id'])
    end=time.monotonic()+50
    while any(j['id'] in ids and j['status'] in ('starting','running') for j in rows(db)):
        if time.monotonic()>end:raise RuntimeError('Shutdown pending: rerun after checking supervisor logs')
        time.sleep(1)


def initial_gen_done(M,E):
    for label in ('random42-complete','simct43-complete'):
        for step in M.STEPS:
            path=M.WORK/'pool'/f'{label}-step{step}'/'plan.json'
            if not path.exists():return False
            if not all((p/'generation-complete.json').exists() for p in M.cells(path,E.read_json(path))):return False
    return True


def handoff():
    M,E,D,Q=modules();cfg=E.read_json(M.WORK/'seed44-extension.json')
    if socket.gethostname()!=M.BORROWED:raise ValueError('Handoff runs on borrowed node')
    while time.time()<cfg['switch_by'] and not initial_gen_done(M,E):time.sleep(15)
    db=local_db(M,E,old=True)
    try:cancel_and_wait(db,{f'gen-minfree-{i}' for i in (1,2,3)})
    finally:db.close()
    M.B.write(M.WORK/'seed44-handoff.json',dict(time=time.time(),initial_gen_complete=initial_gen_done(M,E)))
    print('HANDOFF_READY: GPU1 fixed44, GPU2 atomic44, GPU3 simct44',flush=True)


def environment(M,key,slot):
    c=M.config();mode,_=EXTRA[key]
    # Same launcher and frozen pilot contract; SimCT uses its historical backend.
    env=M.train_env(key,slot,c)
    if mode=='simct':env.update(MP_ALGORITHM='span_ctkd',MP_ATTN_IMPLEMENTATION='sdpa')
    env.pop('MP_TRAIN_STOP_AT',None)
    env.pop('MP_CHECKPOINT_RESERVE_SECONDS',None)
    return env


def train(key,slot):
    M,E,D,Q=modules();target=M.WORK/'train'/key;target.mkdir(parents=True,exist_ok=True)
    if list(target.glob('*/launch-config.json')):raise ValueError('Existing run; no implicit restart')
    env=dict(os.environ)
    for k in ('MP_TRAIN_STOP_AT','MP_CHECKPOINT_RESERVE_SECONDS','MP_ENERGY_CHECKPOINT'):env.pop(k,None)
    env.update(environment(M,key,slot))
    mode=EXTRA[key][0];slotmode='atomic' if mode=='simct' else mode
    try:
        result=subprocess.run(['bash',str(M.ROOT/'experiments/runai/run_single_gpu.sh'),str(slot),slotmode,'312'],env=env)
        M.B.write(target/'finished.json',dict(returncode=result.returncode,time=time.time()))
        if result.returncode:raise SystemExit(result.returncode)
    except OSError as exc:
        M.B.write(target/'finished.json',dict(error=str(exc),time=time.time()));raise


def publisher():
    M,E,D,Q=modules();db=local_db(M,E);old=local_db(M,E,old=True)
    try:
        while True:
            terminal_exports(M,db)
            terminal_exports(M,old)
            M.publish_available()
            done=all((M.WORK/'train'/k/'finished.json').exists() for k in M.TRAIN)
            if done:
                M.B.write(M.WORK/'publisher-done.json',dict(time=time.time(),training_keys=list(M.TRAIN)))
                return
            time.sleep(20)
    finally:db.close();old.close()


def export():
    M,E,D,Q=modules();db=local_db(M,E);old=local_db(M,E,old=True)
    try:
        while True:
            terminal_exports(M,db)
            terminal_exports(M,old)
            if all((M.WORK/'train'/k/'finished.json').exists() for k in (*BORROW_KEYS,'random43')):return
            time.sleep(10)
    finally:db.close();old.close()


def bridge(key):
    M,E,D,Q=modules();db=local_db(M,E,old=True)
    from job_manager.store import rows,TERMINAL
    try:
        while True:
            job=next(j for j in rows(db) if j['id']==key)
            if job['status'] in TERMINAL:
                terminal_exports(M,db);return
            time.sleep(10)
    finally:db.close()


def build_specs(M,host,original):
    jobs=[];owner=host==M.OWNER
    base=next(j['spec'] for j in original if j['id']==('soft42' if owner else 'random43'))
    def make(key,action,slot=None,deps=(),cpu=1,policy='success'):
        s=copy.deepcopy(base);s.update(id=key,argv=[sys.executable,str(Path(__file__).resolve()),action],
            env={},dependencies=list(deps),dependency_policy=policy,cpu_slots=cpu,gpus=0,
            timeout_seconds=72*3600)
        if slot is not None:
            gen=next(j for j in original if j['id']==f'gen-minfree-{slot}')
            s['gpus']=gen['spec']['gpus']
        s.pop('deadline',None)
        jobs.append(s);return s
    if owner:
        make('publisher-seed44','publish')
        for key in ('soft42','soft43'):
            make('wait-'+key,'bridge')['argv']+=['--key',key]
    else:
        make('seed44-handoff','handoff');make('seed44-state-export','export')
        make('wait-random43','bridge')['argv']+=['--key','random43']
        make('gen-after44-0','gen',0,['wait-random43'],4)['argv']+=['--slot','0']
    for i in range(2 if owner else 4):make(f'score-seed44-{i}','score',cpu=4)
    for slot,key in (enumerate(OWN_KEYS) if owner else enumerate(BORROW_KEYS,1)):
        deps=[('wait-soft42' if slot==0 else 'wait-soft43')] if owner else ['seed44-handoff']
        s=make(key,'train',slot,deps,8,'terminal' if owner else 'success')
        s['argv']+=['--key',key,'--slot',str(slot)]
        s=make(f'gen-after44-{slot}','gen',slot,[key],4,'terminal');s['argv']+=['--slot',str(slot)]
    return jobs


def submit():
    M,E,D,Q=modules();host=socket.gethostname()
    if host not in (M.OWNER,M.BORROWED):raise ValueError('Wrong host')
    from job_manager.store import rows,submit as put,initialize,connect,setting
    from job_manager.__main__ import snapshot
    db=local_db(M,E,old=True)
    try:
        with Q.locked(M.WORK/(host+'.seed44-submit.lock')):
            current=rows(db)
            receipt=M.WORK/(host+'.seed44.json')
            if receipt.exists():
                retire_dispatchers(M,Q,db,current)
                new=connect(Path(E.read_json(receipt)['state']))
                try:start_manager(M,E,new,receipt);print(json.dumps(snapshot(new),indent=2))
                finally:new.close()
                return
            if not snapshot(db)['manager']['running']:raise ValueError('Original manager not running; inspect active training first')
            if host==M.BORROWED:
                with Q.locked(M.WORK/'seed44-extension.lock'):
                    path=M.WORK/'seed44-extension.json'
                    if not path.exists():M.B.write(path,dict(created=time.time(),switch_by=time.time()+45*60,
                        borrowed_until=M.config()['borrowed_until'],source=str(Path(__file__).resolve()),
                        training_seeds=[42,43,44],extra=EXTRA,scope='Frozen-energy replication, not online energy learning'))
            elif not (M.WORK/'seed44-extension.json').exists():raise ValueError('Submit borrowed extension first')
            planned=build_specs(M,host,current)
            if host==M.OWNER and any(j['id'] in ('gen-minfree-0','gen-minfree-1') and j['status'] in ('starting','running') for j in current):
                raise ValueError('Owner gen already active; inspect before changing schedule')
            cfg=setting(db,'config');cfg.update(deadline=None,cpu_slots=40 if host==M.OWNER else 64)
            state=Path(tempfile.mkdtemp(prefix='simct-seed44-',dir='/var/tmp'))
            initialize(state,cfg)
            new=connect(state)
            try:put(new,planned)
            finally:new.close()
            # Persist the new queue before stopping old dispatchers; retry can resume it.
            M.B.write(receipt,dict(state=str(state),source=str(Path(__file__).resolve()),specs=planned,time=time.time()))
            retire_dispatchers(M,Q,db,current)
            new=connect(state)
            try:start_manager(M,E,new,receipt);print(json.dumps(snapshot(new),indent=2))
            finally:new.close()
    finally:db.close()


def retire_dispatchers(M,Q,db,current):
    if socket.gethostname()==M.OWNER:
        cancel_and_wait(db,{'publisher','gen-minfree-0','gen-minfree-1'})
        with Q.locked(M.WORK/'publisher.lock'):
            done=M.WORK/'publisher-done.json'
            # Only archive the old marker, not a later completed extension.
            if done.exists() and 'training_keys' not in json.loads(done.read_text()):
                done.rename(done.with_name(f'publisher-done-before44-{time.time_ns()}.json'))
    else:cancel_and_wait(db,{'gen-minfree-0'})
    cancel_and_wait(db,{j['id'] for j in current if j['id'].startswith('score-')})


def start_manager(M,E,db,receipt):
    from job_manager.__main__ import snapshot
    if snapshot(db)['manager']['running']:return
    state=E.read_json(receipt)['state']
    with (M.WORK/(socket.gethostname()+'.seed44-manager.log')).open('ab') as out:
        p=subprocess.Popen([sys.executable,'-m','job_manager','--state',state,'run'],
            cwd=M.BASE/'job-manager',stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    if p.poll() is not None:raise RuntimeError('New manager failed; inspect seed44-manager.log')


def status():
    M,E,D,Q=modules();counts=dict(plans=0,gen=0,score=0,errors=0)
    from datetime import datetime,timezone
    print('NODE_UTC',datetime.now(timezone.utc).isoformat())
    print('SOFT_REFERENCE_UTC',datetime.fromtimestamp(M.config()['borrowed_until'],timezone.utc).isoformat(),
          'HOURS_TO_REFERENCE',(M.config()['borrowed_until']-time.time())/3600,'ALLOCATION_CUTOFF=False')
    for ready in M.plans():
        path=ready.parent/'plan.json';plan=E.read_json(path);cs=M.cells(path,plan)
        counts['plans']+=1;counts['gen']+=sum((p/'generation-complete.json').exists() for p in cs)
        counts['score']+=sum((p/'metrics.json').exists() for p in cs)
        counts['errors']+=sum((p/'score-error.json').exists() for p in cs)+int((path.parent/'generation-error.json').exists())
    print('POOL',json.dumps(counts),'EXPECTED_FULL_PLANS=80 EXPECTED_FULL_CELLS=960')
    print('PUBLISHER_DONE',M.producers_done())
    from job_manager.__main__ import snapshot
    db=local_db(M,E)
    try:print(json.dumps(snapshot(db),indent=2))
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['submit','handoff','train','publish','export','gen','score','bridge','status'])
    p.add_argument('--key',choices=[*EXTRA,'soft42','soft43','random43']);p.add_argument('--slot',type=int,choices=range(4))
    a=p.parse_args()
    if a.action=='train':train(a.key,a.slot)
    elif a.action=='gen':
        M,E,D,Q=modules();R.install_adapter(Q);M.generate(a.slot)
    elif a.action=='score':
        M,E,D,Q=modules();M.score()
    elif a.action=='bridge':bridge(a.key)
    else:{'submit':submit,'handoff':handoff,'publish':publisher,'export':export,'status':status}[a.action]()
