"""Bounded, journal-preserving retries around the pinned six-B200 scorer."""
import argparse
import copy
import json
from pathlib import Path
import re
import socket
import sys
import time

import queue_seed44 as S


def signal_code(message):
    if 'scorer runtime error' not in message:return None
    match=re.search(r'"returncode"\s*:\s*(-\d+)',message)
    return int(match[1]) if match and int(match[1]) in (-15,-9,-11) else None


def reserve(cell,message,write):
    """Caller owns score.lock. Persist before retry so restarts cannot reset budget."""
    code=signal_code(message)
    if code is None:return False
    path=cell/'score-recovery.json'
    state=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
    maximum=3 if code==-15 else 1
    used=sum(a['signal']==code for a in state['attempts'])
    if used>=maximum or len(state['attempts'])>=4:
        state['exhausted']=True;state['last_error']=message;write(path,state);return False
    state['attempts'].append(dict(signal=code,error=message,time=time.time(),host=socket.gethostname()))
    write(path,state);return True


def recover_markers(M,E,Q,readies):
    for ready in readies:
        path=ready.parent/'plan.json'
        for cell in M.cells(path,E.read_json(path)):
            if not (cell/'score-error.json').exists():continue
            with Q.locked(cell/'score.lock',blocking=False) as lock:
                if lock is None:continue
                error=cell/'score-error.json'
                if (cell/'metrics.json').exists() or not error.exists():continue
                value=E.read_json(error)
                if reserve(cell,value.get('error',''),M.B.write):
                    error.rename(cell/f'score-error-recovered-{time.time_ns()}.json')
                    print('AUTO_RETRY_READY',cell,flush=True)


def install(M,Q):
    original=Q.run_cell
    def wrapped(root,plan,ph,job,bench,seed,base,server,args,deadline):
        cell=root/'cells'/job['id']/bench/str(seed)
        attempt_args=copy.copy(args)
        # Existing crash retries start serially too.
        if (cell/'score-recovery.json').exists():attempt_args.score_workers=1
        while True:
            try:return original(root,plan,ph,job,bench,seed,base,server,attempt_args,deadline)
            except Q.Deadline:raise
            except Exception as exc:
                if time.time()+180>=deadline or not reserve(cell,str(exc),M.B.write):raise
                print('AUTO_RETRY_SERIAL',cell,str(exc),flush=True)
                attempt_args.score_workers=1
                time.sleep(10)
    Q.run_cell=wrapped


def worker():
    M,E,D,Q=S.modules()
    original_plans=M.plans
    def recovering_plans():
        readies=list(original_plans())
        recover_markers(M,E,Q,readies)
        return readies
    M.plans=recovering_plans;install(M,Q)
    M.score()
    unresolved=[]
    for ready in original_plans():
        unresolved += [str(p) for p in ready.parent.rglob('score-error.json')]
        unresolved += [str(p) for p in ready.parent.glob('lcbfix-error.json')]
    M.B.write(M.WORK/(socket.gethostname()+'.scoring-attention.json'),dict(time=time.time(),unresolved=unresolved))
    if unresolved:raise SystemExit('Retry budget exhausted or non-retryable errors; see scoring-attention.json')


def submit():
    M,E,D,Q=S.modules()
    from job_manager.store import rows,submit as put
    from job_manager.__main__ import snapshot
    db=S.local_db(M,E)
    try:
        with Q.locked(M.WORK/(socket.gethostname()+'.resilient-score.lock')):
            current=rows(db)
            if not snapshot(db)['manager']['running']:raise ValueError('Seed44 manager is not running')
            source=[j for j in current if j['id'].startswith('score-seed44-')]
            expected=2 if socket.gethostname()==M.OWNER else 4
            if len(source)!=expected:raise ValueError('Unexpected scorer inventory')
            planned=[]
            for j in source:
                spec=copy.deepcopy(j['spec']);spec['id']=j['id'].replace('score-seed44-','score-resilient-')
                spec['argv']=[sys.executable,str(Path(__file__).resolve()),'worker'];planned.append(spec)
            # Stop old coordinators first; their SIGTERM errors are recovered under cell locks.
            S.cancel_and_wait(db,{j['id'] for j in source})
            existing={j['id'] for j in rows(db)}
            pending=[j for j in planned if j['id'] not in existing]
            if pending:put(db,pending)
            print(json.dumps(snapshot(db),indent=2))
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['submit','worker'])
    {'submit':submit,'worker':worker}[p.parse_args().action]()
