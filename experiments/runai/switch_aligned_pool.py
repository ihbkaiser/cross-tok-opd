"""Move only the authorized old eval workers to the aligned shared pool."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
POOL=BASE/'aligned-seed43-pool'
sys.path.insert(0,str(BASE/'job-manager'))
from job_manager.store import connect,rows,submit,TERMINAL
from job_manager.__main__ import snapshot


def context():
    host=socket.gethostname()
    if host=='tungks-0-1':
        return json.loads((BASE/'owner-tungks-0-1/campaign.json').read_text()),'old-eval',0
    if host=='tungdd11-sparse-vllm-core-0-0':
        return json.loads((BASE/'extra-eval-gpu2/campaign.json').read_text()),'old-eval-gpu2',2
    raise ValueError('Unexpected host')


def stop(cfg,key):
    subprocess.run([sys.executable,'-m','job_manager','--state',cfg['state'],'cancel',key],
                   cwd=BASE/'job-manager',check=True)
    db=connect(Path(cfg['state']))
    try:
        until=time.time()+180
        while True:
            job=next(j for j in rows(db) if j['id']==key)
            if job['status']=='lost':raise ValueError('Lost worker requires process inspection')
            if job['status'] in TERMINAL:break
            if time.time()>until:raise ValueError('Old worker still active; migration refused')
            time.sleep(2)
    finally:db.close()
    print('OLD_EVAL_STOPPED',socket.gethostname(),key,flush=True)


def join(cfg,key,gpu):
    receipt=json.loads((POOL/'migration.json').read_text())
    if receipt['status']!='verified':raise ValueError('Pool migration not verified')
    import hashlib
    if hashlib.sha256((POOL/'plan.json').read_bytes()).hexdigest()!=receipt['new_plan_sha256']:
        raise ValueError('Pool plan changed')
    source=Path(receipt['source'])
    deadline=min(receipt['deadline'],cfg.get('deadline') or receipt['deadline'])
    if time.time()+600>=deadline:raise ValueError('Insufficient worker allocation time')
    db=connect(Path(cfg['state']))
    try:
        state=snapshot(db)
        if not state['manager']['running'] or state['manager']['tick_age_seconds']>30 or state['paused'] or state['quarantined']:
            raise ValueError('Manager not ready')
        old=next(j for j in rows(db) if j['id']==key)
        if old['status'] not in TERMINAL or old['status']=='lost':raise ValueError('Old eval still active')
        common=dict(version=1,project='simct-aligned-seed43',cwd=str(source),
                    env={},cpu_slots=4,timeout_seconds=deadline-60-time.time(),deadline=deadline-60,
                    dependencies=[],dependency_policy='success')
        exe=source/'scripts/evaluation/eval_queue.py'
        jobs=[dict(common,id=f'aligned-gen-{gpu}',gpus=old['spec']['gpus'],
            argv=[sys.executable,str(exe),'worker','--plan',str(POOL/'plan.json'),'--gpu',str(gpu),
                  '--phase','generate','--concurrency','256','--score-buffer','256','--internal-code-execution'])]
        if gpu==0:
            jobs.append(dict(common,id='aligned-score-owner',gpus=0,
                argv=[sys.executable,str(exe),'score-spool','--plan',str(POOL/'plan.json'),
                      '--score-workers','4','--internal-code-execution']))
        existing={j['id'] for j in rows(db)}
        if any(j['id'] in existing for j in jobs):raise ValueError('Pool jobs already submitted; inspect status')
        submit(db,jobs)
        print(json.dumps(snapshot(db),indent=2))
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stop','owner-start','join'])
    args=p.parse_args();cfg,key,gpu=context()
    if args.action=='stop':stop(cfg,key)
    elif args.action=='owner-start':
        if gpu!=0:raise ValueError('Owner node only')
        stop(cfg,key)
        subprocess.run([sys.executable,str(ROOT/'experiments/runai/aligned_eval_pool.py'),'--out',str(POOL)],check=True)
        join(cfg,key,gpu)
    else:join(cfg,key,gpu)
