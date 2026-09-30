"""Audited seed43 curve migration; source pools must have stopped writers."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/evaluation'))
import eval_queue as Q
E,D=Q.E,Q.D
sys.path.insert(0,str(ROOT/'experiments/runai'))
import borrowed_campaign as B
STEPS=(40,80,120,156,200,240,280,312)


def compatible(reference, candidate):
    if reference['seeds'] != [42,43,44] or candidate['seeds'] != reference['seeds']:
        raise ValueError('Evaluation seeds changed')
    if candidate['profile'] != reference['profile']:
        raise ValueError('Evaluation profile changed')
    for plan in (reference,candidate):
        for name,digest in D.script_hashes().items():
            if name!='eval_queue.py' and plan['source'].get(name)!=digest:
                raise ValueError('Non-scheduling evaluation source changed: '+name)
    clean=lambda p:{k:v for k,v in p['protocol'].items() if k!='scope'}
    if clean(reference)!=clean(candidate):raise ValueError('Decoding/protocol changed')
    if set(reference['data'])!=set(candidate['data']):raise ValueError('Benchmark set changed')
    for key in reference['data']:
        a,b=reference['data'][key],candidate['data'][key]
        if (a['sha256'],a['count'])!=(b['sha256'],b['count']):raise ValueError('Benchmark data changed: '+key)


def migrate_cell(src,dst,oldplan,oldhash,oldjob,newhash,newjob,benchmark,seed):
    before={p.name:E.file_hash(p) for p in src.iterdir() if p.is_file() and p.suffix!='.lock'}
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('*.lock'))
    contract=E.read_json(dst/'contract.json')
    if contract!=Q.cell_contract(oldhash,oldjob,oldplan['data'],benchmark,seed,contract['server']):
        raise ValueError('Old cell contract mismatch')
    metrics=E.read_json(dst/'metrics.json') if (dst/'metrics.json').exists() else None
    if metrics:
        if metrics['contract']!=contract or metrics['status']!='completed':raise ValueError('Old metrics contract')
        for name in ('responses','scores'):
            if E.file_hash(dst/(name+'.jsonl'))!=metrics[name+'_sha256']:raise ValueError('Completed journal hash mismatch')
    responses=Q.journal(dst/'responses.jsonl');scores=Q.journal(dst/'scores.jsonl')
    items={x['id']:x for x in E.read_json(oldplan['data'][benchmark]['path'])['items']}
    if not set(scores)<=set(responses)<=set(items):raise ValueError('Unknown journal ID')
    for key,row in responses.items():
        expected=E.digest(E.encoded(E.generation_payload('eval-gemma',items[key],benchmark,seed)))
        if row['seed']!=seed or row['request_sha256']!=expected:raise ValueError('Generation request drift')
        E.validate_response(row['response'])
    for key,row in scores.items():
        if type(row.get('passed')) is not bool or row['response_sha256']!=E.digest(E.encoded(responses[key])):
            raise ValueError('Scoring provenance mismatch')
    fresh=Q.cell_contract(newhash,newjob,oldplan['data'],benchmark,seed,contract['server'])
    if metrics:
        if len(scores)!=len(items) or metrics['count']!=len(items) or metrics['score']!=sum(x['passed'] for x in scores.values())/len(items):
            raise ValueError('Score/count mismatch')
        metrics['contract']=fresh;Q.atomic_json(dst/'metrics.json',metrics)
    marker=dst/'generation-complete.json'
    if marker.exists():
        value=E.read_json(marker)
        if value['contract']!=contract or value['count']!=len(items) or value['responses_sha256']!=E.file_hash(dst/'responses.jsonl'):
            raise ValueError('Old spool marker mismatch')
    if len(responses)==len(items):
        Q.atomic_json(marker,dict(contract=fresh,count=len(items),responses_sha256=E.file_hash(dst/'responses.jsonl')))
    Q.atomic_json(dst/'contract.json',fresh)
    after={p.name:E.file_hash(p) for p in src.iterdir() if p.is_file() and p.suffix!='.lock'}
    if before!=after:raise ValueError('Source cell changed during migration')
    return dict(responses=len(responses),scores=len(scores),metrics=int(metrics is not None))


def prepare(out):
    if socket.gethostname()!='tungks-0-1':raise ValueError('Prepare on owner node only')
    base=B.BASE/'borrow8-8MgodXcM'
    cfg=E.read_json(B.OLD_WORK/'config.json')
    reference=E.read_json(Path(cfg['eval_template']))
    paths=[base/'work/eval-old/plan.json',B.BASE/'endpoint-recovery-y8cQZzRW/work/eval-endpoints/plan.json']
    plans=[E.read_json(p) for p in paths]
    for plan in plans:compatible(reference,plan)
    for data in reference['data'].values():
        if E.file_hash(data['path'])!=data['sha256']:raise ValueError('Data hash mismatch')
    # Existing worker allocation was explicitly moved to the owner's 24h window.
    state=E.read_json(paths[0].parent/'state.json');deadline=state['deadline']
    if time.time()+600>=deadline:raise ValueError('Pool deadline expired; no implicit extension')
    with contextlib.ExitStack() as locks:
        for p,plan in zip(paths,plans):
            for name in ['scoring.lock','state.lock']+[j['id']+'.lock' for j in plan['jobs']]:
                if locks.enter_context(Q.locked(p.parent/name,blocking=False)) is None:
                    raise ValueError('Old writer active: '+str(p.parent/name))
        jobs=[];inventory=[]
        for mode,run in [('fixed',B.OLD_FIXED),('atomic',B.OLD_ATOMIC),('random',B.OLD_RANDOM)]:
            launch=E.read_json(run/'launch-config.json')
            if launch['options']['seed']!=43 or launch['options']['mp_opd_mode']!=mode:
                raise ValueError('Training seed/mode mismatch: '+str(run))
            for step in STEPS:
                cp=run/'checkpoint'/f'step{step}'
                if not cp.is_dir():
                    inventory.append(dict(mode=mode,step=step,status='checkpoint_missing'));continue
                print('HASH_CHECKPOINT',mode,step,flush=True)
                identity=B.checkpoint_stable(cp)
                jobs.append(dict(id=f'{mode}-seed43-{step}',mode=mode,step=step,tier=0,checkpoint=identity))
        new=copy.deepcopy(reference)
        new.update(jobs=jobs,source=D.script_hashes(),hours=(deadline-time.time())/3600,
                   admit_hours=(deadline-time.time())/3600)
        new['protocol']['scope']='Seed43 checkpoint curve aligned to historical seed42 steps; three evaluation seeds, not training replicates'
        new['migration']={'from_plans':[dict(path=str(p),sha256=E.file_hash(p)) for p in paths],
                          'reference_seed42_plan':str(cfg['eval_template']),
                          'reference_seed42_sha256':E.file_hash(Path(cfg['eval_template']))}
        out.mkdir(parents=True,exist_ok=False)
        E.write_new(out/'migration.json',dict(status='incomplete'))
        E.write_new(out/'plan.json',new);ph=E.file_hash(out/'plan.json')
        index={j['id']:j for j in jobs};totals=dict(responses=0,scores=0,metrics=0)
        for path,plan in zip(paths,plans):
            oldhash=E.file_hash(path)
            for job in plan['jobs']:
                target=index.get(job['id'])
                if target is None:raise ValueError('Unexpected old checkpoint '+job['id'])
                if target['checkpoint']!=job['checkpoint']:raise ValueError('Checkpoint identity changed')
                for benchmark in plan['data']:
                    for seed in plan['seeds']:
                        src=path.parent/'cells'/job['id']/benchmark/str(seed)
                        if not src.exists():continue
                        dst=out/'cells'/job['id']/benchmark/str(seed)
                        dst.parent.mkdir(parents=True,exist_ok=True)
                        counts=migrate_cell(src,dst,plan,oldhash,job,ph,target,benchmark,seed)
                        for key,value in counts.items():totals[key]+=value
        qualification=Q.preflight('/usr/bin/python3.12')
        if qualification!=state.get('qualification'):raise ValueError('Grader runtime drift')
        E.write_new(out/'state.json',dict(plan_sha256=ph,started=state['started'],deadline=deadline,
            admit_until=deadline-300,jobs={},durations=[],qualification=qualification))
        receipt=dict(status='verified',new_plan_sha256=ph,source=str(ROOT),totals=totals,
                     checkpoint_count=len(jobs),cell_count=len(jobs)*12,missing=inventory,deadline=deadline)
        Q.atomic_json(out/'migration.json',receipt)
        print('MIGRATION_PASS',json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out.resolve())
