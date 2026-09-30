"""Prepare fresh endpoint eval for completed campaign checkpoints; no retraining."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/evaluation'))
import contract_eval as E
import queue_data as D
import eval_queue as Q


def main(original,out):
    prior=E.read_json(original/'campaign-state.json')
    previous=E.read_json(original/'plan.json')
    deadline=prior['started']+previous['hours']*3600
    remaining=deadline-time.time()
    if remaining<300:raise ValueError('Less than five minutes remain in original campaign')
    plans=[E.read_json(original/f'eval-{mode}/plan.json') for mode in ('fixed','atomic')]
    for mode in ('fixed','atomic'):
        if prior['jobs'][mode+'43']['status']!='completed':raise ValueError('Training not completed: '+mode)
    plan=plans[0]
    if plans[1]['data']!=plan['data'] or plans[1]['protocol']!=plan['protocol']:
        raise ValueError('Endpoint protocols differ')
    jobs=plan['jobs']+plans[1]['jobs']
    for job in jobs:
        if E.checkpoint_identity(Path(job['checkpoint']['path']))!=job['checkpoint']:
            raise ValueError('Checkpoint identity changed')
    for data in plan['data'].values():
        if E.file_hash(data['path'])!=data['sha256']:raise ValueError('Data changed')
    qualification=Q.preflight('/usr/bin/python3.12')
    out.mkdir(parents=True,exist_ok=False)
    target=out/'eval-endpoints';target.mkdir()
    plan.update(jobs=jobs,source=D.script_hashes(),hours=remaining/3600,admit_hours=remaining/3600)
    E.write_new(target/'plan.json',plan)
    now=time.time()
    Q.atomic_json(target/'state.json',dict(plan_sha256=E.file_hash(target/'plan.json'),started=now,
        deadline=deadline,admit_until=deadline,jobs={},durations=[],qualification=qualification))
    tasks=[]
    for gpu in (0,1):
        tasks.append(dict(id=f'generate-{gpu}',gpu=gpu,self_locks=True,budget_seconds=max(1,int(deadline-time.time())),
            argv=['/usr/bin/python3.12',str(ROOT/'scripts/evaluation/eval_queue.py'),'worker','--phase','generate',
                  '--gpu',str(gpu),'--concurrency','128','--score-buffer','256','--plan',str(target/'plan.json'),'--internal-code-execution']))
    tasks.append(dict(id='score',gpu=None,budget_seconds=max(1,int(deadline-time.time())),
        argv=['/usr/bin/python3.12',str(ROOT/'scripts/evaluation/eval_queue.py'),'score-spool','--plan',str(target/'plan.json'),'--internal-code-execution']))
    # Keep admission budgets below remaining time despite preparer/launcher delay.
    for task in tasks:task['budget_seconds']=max(1,task['budget_seconds']-120)
    campaign=dict(hours=previous['hours'],source=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),jobs=tasks)
    E.write_new(out/'plan.json',campaign)
    Q.atomic_json(out/'campaign-state.json',dict(plan_sha256=E.file_hash(out/'plan.json'),started=prior['started'],jobs={}))
    print('READY',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.original.resolve(),a.out.resolve())
