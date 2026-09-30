"""Replace only six-B200 gen adapters; retain immutable evaluator and manifests."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

BASE=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
WORK=BASE/'six-b200-v1'
OLD_COMMIT='2d4d3ea016027c8e4c9fe6d6bcd305b0c16ff733'


def is_adapter_error(value):
    message=value.get('error','')
    return "has no attribute 'min_free_gib'" in message


def load_original():
    cfg=json.loads((WORK/'campaign.json').read_text());root=Path(cfg['source'])
    if cfg['commit']!=OLD_COMMIT:raise ValueError('Repair targets only original six-B200 campaign')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=OLD_COMMIT:
        raise ValueError('Pinned source changed')
    spec=importlib.util.spec_from_file_location('six_original',root/'experiments/runai/six_b200.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install_adapter(Q):
    original=Q.run_checkpoint
    def repaired(root,plan,ph,job,args,deadline):
        if not hasattr(args,'min_free_gib'):args.min_free_gib=20
        return original(root,plan,ph,job,args,deadline)
    Q.run_checkpoint=repaired


def recover():
    M=load_original();E,D,Q=M.B.eval_modules();host=socket.gethostname()
    if host not in (M.OWNER,M.BORROWED):raise ValueError('Wrong host')
    receipt=E.read_json(WORK/(host+'.json'));state=Path(receipt['state'])
    sys.path.insert(0,str(BASE/'job-manager'))
    from job_manager.store import connect,rows,cancel,submit
    from job_manager.__main__ import snapshot
    db=connect(state)
    try:
        with Q.locked(WORK/(host+'.gen-repair.lock')):
            current=rows(db)
            if not snapshot(db)['manager']['running']:raise ValueError('Manager not running')
            old=[j for j in current if j['id'] in {f'gen-{i}' for i in range(4)}]
            if not old:raise ValueError('No original generation jobs found')
            # Guard against applying this diagnosis to unrelated generation failures.
            errors=list((WORK/'pool').glob('*/generation-error.json'))
            for error in errors:
                value=E.read_json(error)
                print('GENERATION_ERROR',error.parent.name,json.dumps(value),flush=True)
            prior=list((WORK/'pool').glob('*/generation-error-before-minfree-*.json'))
            if errors and not any(is_adapter_error(E.read_json(p)) for p in errors+prior):
                raise ValueError('No min_free_gib failure found; inspect printed errors first')
            for j in old:
                if j['status'] in ('queued','starting','running'):cancel(db,j['id'])
            end=time.monotonic()+50
            while any(j['id'] in {o['id'] for o in old} and j['status'] in ('starting','running') for j in rows(db)):
                if time.monotonic()>end:raise RuntimeError('Gen shutdown pending; rerun repair after inspecting supervisor logs')
                time.sleep(1)
            for error in (WORK/'pool').glob('*/generation-error.json'):
                with Q.locked(error.parent/'generation-dispatch.lock',blocking=False) as claim:
                    if claim is None:continue
                    if error.exists() and is_adapter_error(E.read_json(error)):
                        target=error.with_name(f'generation-error-before-minfree-{time.time_ns()}.json')
                        error.rename(target)
                        print('RETRY_READY',error.parent.name,flush=True)
            existing={j['id'] for j in rows(db)};pending=[]
            for oldjob in old:
                s=copy.deepcopy(oldjob['spec']);slot=int(oldjob['id'].split('-')[-1])
                s['id']=f'gen-minfree-{slot}'
                if s['id'] in existing:continue
                s['argv']=[sys.executable,str(Path(__file__).resolve()),'worker','--slot',str(slot)]
                pending.append(s)
            if pending:submit(db,pending)
            Q.atomic_json(WORK/(host+'.gen-repair.json'),dict(time=time.time(),source=str(Path(__file__).resolve()),
                state=str(state),replacement_ids=[f'gen-minfree-{j["id"].split("-")[-1]}' for j in old],
                scope='Only add evaluator min_free_gib=20; original source, pool, training and deadlines preserved'))
            print(json.dumps(snapshot(db),indent=2))
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['recover','worker']);p.add_argument('--slot',type=int,choices=range(4))
    args=p.parse_args()
    if args.action=='recover':recover()
    else:
        M=load_original();_,_,Q=M.B.eval_modules();install_adapter(Q);M.generate(args.slot)
