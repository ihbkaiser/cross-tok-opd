"""Dependency-driven soft50 evaluation and adapter oracle follow-up."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
WORK=BASE/'soft50-next-v1'
PLAN=WORK/'eval/plan.json'
READY=WORK/'ready.json'
sys.path.insert(0,str(ROOT/'experiments/runai'))
import borrowed_campaign as B


def prepare():
    runs=BASE/'owner-tungks-0-1/soft50-followup/runs'
    candidates=list(runs.glob('*/checkpoint/run-summary.json'))
    if len(candidates)!=1:raise ValueError('Expected exactly one soft50 summary')
    summary=json.loads(candidates[0].read_text());run=candidates[0].parents[1]
    if summary['status']!='completed' or summary['optimizer_updates']!=50:
        raise ValueError('Soft50 did not complete successfully')
    if Path(str(run)+'.exitcode').read_text().strip()!='0':raise ValueError('Soft exit not zero')
    launch=json.loads((run/'launch-config.json').read_text())
    if launch['options']['seed']!=42 or launch['options']['mp_opd_mode']!='soft':
        raise ValueError('Unexpected soft seed/mode')
    cp=run/'checkpoint/step50'
    if not cp.is_dir():raise ValueError('Missing step50')
    WORK.mkdir(exist_ok=True)
    B.build_eval(WORK,'eval',[('soft-seed42-50','soft',50,cp)],time.time()+86400)
    B.write(READY,dict(checkpoint=str(cp),launch=str(run/'launch-config.json'),source=str(ROOT)))


def wait_ready():
    while not READY.exists():
        if time.time()+60>=float(os.environ.get('JM_DEADLINE',time.time()+3600)):
            raise ValueError('Allocation ended before soft checkpoint ready')
        print('WAIT_SOFT50_READY',flush=True);time.sleep(30)
    receipt=json.loads(READY.read_text())
    if receipt['source']!=str(ROOT) or not PLAN.exists():raise ValueError('Ready/source mismatch')


def oracle():
    ready=json.loads(READY.read_text());launch=json.loads(Path(ready['launch']).read_text())
    # Recovery energy is a symlink to the original guide-energy directory.
    energy=(B.BASE/'partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt').resolve()
    guide=energy.parent.parent/'probe-data.json'
    if not guide.is_file():raise ValueError('Missing original guide data: '+str(guide))
    argv=['bash',str(ROOT/'experiments/runai/python-b200-host.sh'),
          str(ROOT/'experiments/mp_opd/real_oracle.py'),'run',
          '--student',ready['checkpoint'],'--teacher',launch['options']['teacher_name_or_path'],
          '--data',str(guide),'--output',str(WORK/'oracle'),
          '--adapter-module','model.layers.25.self_attn.q_proj','--device','cuda:0',
          '--rank','4','--virtual-lr','0.1','--model-dtype','float32','--max-span','2',
          '--weighting-steps','1','--select-counts','1','4','--norm-controls',
          '--max-new-tokens','128','--max-prompt-tokens','4096','--max-reference-tokens','4096']
    B.write(WORK/'oracle-provenance.json',dict(argv=argv,guide_sha256=B.sha(guide),
        scope='Adapter diagnostic on soft50; reused guide; not an independent benchmark or frozen-soft selector comparison'))
    subprocess.run(argv,check=True)


def submit():
    owner=socket.gethostname()=='tungks-0-1'
    if not owner and socket.gethostname()!='tungdd11-sparse-vllm-core-0-0':raise ValueError('Wrong host')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Dirty source')
    if owner:
        energy=(B.BASE/'partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt').resolve()
        if not (energy.parent.parent/'probe-data.json').is_file():
            raise ValueError('Original oracle guide missing: '+str(energy.parent.parent/'probe-data.json'))
    cfg=json.loads((BASE/('owner-tungks-0-1' if owner else 'extra-eval-gpu2')/'campaign.json').read_text())
    sys.path.insert(0,str(cfg.get('manager') or BASE/'job-manager'))
    from job_manager.store import connect,rows,submit as put,setting,transaction,event
    from job_manager.__main__ import snapshot
    db=connect(Path(cfg['state']))
    try:
        state=snapshot(db)
        if not state['manager']['running'] or state['paused'] or state['quarantined']:raise ValueError('Manager not ready')
        jobs=rows(db);old=next(j for j in jobs if j['id']==('aligned-gen-0' if owner else 'aligned-gen-2'))
        exe=str(ROOT/'scripts/evaluation/eval_queue.py');self=str(Path(__file__).resolve())
        common=dict(version=1,project='soft50-next-v1',cwd=str(ROOT),env={},gpus=0,cpu_slots=1,
                    timeout_seconds=86400,dependencies=[],dependency_policy='success')
        def job(key,argv,deps,gpu=False,cpu=1):
            s=copy.deepcopy(common);s.update(id=key,argv=argv,dependencies=deps,cpu_slots=cpu)
            if gpu:s['gpus']=old['spec']['gpus']
            return s
        if owner:
            pending=[job('soft50-prepare',[sys.executable,self,'prepare'],['soft-parity-50']),
                     job('soft50-oracle',[sys.executable,self,'oracle'],['soft50-prepare','aligned-gen-0'],True,8),
                     job('soft50-score',[sys.executable,exe,'score-spool','--plan',str(PLAN),
                         '--score-workers','4','--internal-code-execution'],['soft50-prepare'],cpu=4),
                     job('soft50-lcbfix',[sys.executable,str(ROOT/'scripts/evaluation/lcbfix.py'),
                         '--plan',str(PLAN),'--out',str(WORK/'lcbfix'),'--workers','4','--internal-code-execution'],
                         ['soft50-score'],cpu=4)]
            pending[1]['timeout_seconds']=7200
        else:
            pending=[job('soft50-wait',[sys.executable,self,'wait'],[]),
                     job('soft50-gen',[sys.executable,exe,'worker','--plan',str(PLAN),'--gpu','2',
                         '--phase','generate','--concurrency','256','--score-buffer','256',
                         '--internal-code-execution'],['soft50-wait','aligned-gen-2'],True,4)]
            deadline=setting(db,'config')['deadline']
            for s in pending:
                s['deadline']=deadline-60;s['timeout_seconds']=deadline-time.time()-60
                if s['timeout_seconds']<1800:raise ValueError('Insufficient borrowed allocation')
        existing={j['id'] for j in jobs}
        pending=[s for s in pending if s['id'] not in existing]
        if pending:
            with transaction(db):
                c=setting(db,'config');before=c['cpu_slots'];c['cpu_slots']=max(before,40 if owner else 24)
                db.execute("UPDATE settings SET value=? WHERE key='config'",(json.dumps(c),))
                event(db,None,'soft_followup_capacity',dict(before=before,after=c['cpu_slots']))
            put(db,pending)
        print(json.dumps(snapshot(db),indent=2))
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','prepare','wait','oracle'])
    action=p.parse_args().action
    {'submit':submit,'prepare':prepare,'wait':wait_ready,'oracle':oracle}[action]()
