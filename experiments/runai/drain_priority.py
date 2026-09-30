"""Drain legacy manager without killing active training, then replace scheduler."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import campaign as C


def inspect(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'state':fields[0], 'start':fields[19], 'exit':int(fields[49])}
    except FileNotFoundError:
        return None


def matching_manager(work):
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:args=path.joinpath('cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(x.endswith(b'/campaign.py') for x in args) and str(work/'plan.json').encode() in args:
            matches.append(int(path.name))
    if len(matches)!=1:raise RuntimeError(f'Expected one original manager, found {matches}')
    return matches[0]


def run(work,recovery):
    lock=(work/'priority-controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(*_):raise KeyboardInterrupt('priority controller interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    # All identities and paths are checked before signalling anything.
    plan=json.loads((work/'plan.json').read_text())
    json.loads((recovery/'plan.json').read_text())
    pid=matching_manager(work); identity=inspect(pid)['start']
    stopped=False; replaced=False
    try:
        os.kill(pid,signal.SIGSTOP);stopped=True
        for _ in range(100):
            if inspect(pid)['state'] in ('T','t'):break
            time.sleep(.01)
        else:raise RuntimeError('Manager did not stop')
        state=json.loads((work/'campaign-state.json').read_text())
        active={k:r for k,r in state['jobs'].items() if r['status']=='running'}
        children={int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()
                  and (C.proc(p.name) or (None,))[0]==pid}
        if children!={r['pid'] for r in active.values()}:
            raise RuntimeError('Manager was between launches; resumed unchanged, retry controller')
        known={}
        for key,row in active.items():
            info=inspect(row['pid'])
            if not info or info['start']!=row['start_ticks']:raise RuntimeError('Cannot adopt '+key)
            known[key]=C.descendants({row['pid']:row['start_ticks']})
        deadline=state['started']+plan['hours']*3600
        budgets={j['id']:j['budget_seconds'] for j in plan['jobs']}
        print('DRAINING',list(active),'original deadline',deadline,flush=True)
        while active:
            for key,row in list(active.items()):
                C.descendants(known[key]);info=inspect(row['pid'])
                if not info or info['start']!=row['start_ticks']:
                    raise RuntimeError('Lost unreaped child identity: '+key)
                timed=time.time()>=min(deadline,row['started']+budgets[key])
                if info['state']=='Z' or timed:
                    # Zombies retain the kernel wait status while old parent is stopped.
                    rc=os.waitstatus_to_exitcode(info['exit']) if info['state']=='Z' else None
                    C.terminate(known[key])
                    row.update(status='timeout' if timed else ('completed' if rc==0 else 'failed'),
                               returncode=rc,ended=time.time())
                    del active[key]
                    print('DRAINED',key,row['status'],flush=True)
            if active:time.sleep(2)
        # Every active child finished or reached its already-authorized timeout.
        if inspect(pid)['start']!=identity:raise RuntimeError('Manager identity changed')
        C.write(work/'admission.json',{'atomic43':{'state':str(recovery/'campaign-state.json'),'job':'probe-decision'}})
        C.write(work/'priority-handoff.json',{'old_pid':pid,'time':time.time(),'state':state,
            'policy':'atomic43 waits for recovery probe-decision terminal; technical failure permits baseline only'})
        os.kill(pid,signal.SIGKILL);replaced=True
        for _ in range(100):
            info=inspect(pid)
            if not info or info['state']=='Z':break
            time.sleep(.05)
        C.write(work/'campaign-state.json',state)
        log=(work/'manager-priority.log').open('ab')
        child=subprocess.Popen(['/usr/bin/python3.12','-u',str(Path(__file__).with_name('campaign.py')),str(work/'plan.json')],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('PRIORITY_MANAGER_PID',child.pid,flush=True)
    finally:
        if stopped and not replaced:
            info=inspect(pid)
            if info and info['start']==identity:os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--recovery',type=Path,required=True);args=parser.parse_args()
    run(args.work.resolve(),args.recovery.resolve())
