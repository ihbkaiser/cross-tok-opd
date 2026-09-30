"""Isolated CPU diagnostics. Never modifies benchmark scores or source journals."""
import argparse
import builtins
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def child():
    import internal_worker as original
    trace = os.fdopen(os.dup(2),'w',buffering=1)
    def mark(message):
        trace.write('STAGE '+message+'\n');trace.flush()
    load = original.load_lcb
    def traced_load():
        module = load()
        guard = module.reliability_guard
        def traced_guard(*a,**kw):
            guard(*a,**kw)
            faulthandler.enable(file=trace,all_threads=True)
            mark('after_reliability_guard')
        module.reliability_guard = traced_guard
        return module
    original.load_lcb = traced_load
    real_import = builtins.__import__
    def traced_import(name,*a,**kw):
        target = name in ('scipy','scipy.spatial')
        if target:mark('before_import '+name)
        try:
            module = real_import(name,*a,**kw)
        except BaseException:
            if target:traceback.print_exc(file=trace)
            raise
        if target:mark('after_import '+name)
        if name=='scipy.spatial' and hasattr(module,'ConvexHull'):
            hull = module.ConvexHull
            if not getattr(hull,'_diagnostic',False):
                def wrapped(*args,**kwargs):
                    mark('before_ConvexHull')
                    try:
                        result=hull(*args,**kwargs)
                    except BaseException:
                        mark('ConvexHull_exception');traceback.print_exc(file=trace);raise
                    mark('after_ConvexHull');return result
                wrapped._diagnostic=True
                module.ConvexHull=wrapped
        return module
    builtins.__import__=traced_import
    mark('before_internal_worker')
    try:
        original.main()
    except BaseException:
        traceback.print_exc(file=trace);raise
    finally:
        mark('worker_returned')


def main():
    import contract_eval as E
    import queue_data as D
    import eval_queue as Q
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    base=Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
    pool=base/'aligned-seed43-pool'
    cell=pool/'cells/fixed-seed43-156/live-code-bench-v6/44'
    plan=E.read_json(pool/'plan.json');data=plan['data']['live-code-bench-v6']
    if E.file_hash(data['path'])!=data['sha256']:raise ValueError('Dataset changed')
    rows=[json.loads(line) for line in (cell/'responses.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['id']=='abc312_e')
    item=next(i for i in E.read_json(data['path'])['items'] if i['id']=='abc312_e')
    decoded=D.score_item('live-code-bench-v6',item)
    first=dict(decoded,tests=decoded['tests'][:1])
    payload=dict(profile=D.PROFILE,benchmark='live-code-bench-v6',item=first,
                 text=row['response']['choices'][0]['message']['content'])
    control=dict(profile=D.PROFILE,benchmark='live-code-bench-v6',
                 item=dict(tests=[dict(input='',output='5\n')],fn_name=None),text='print(5)')
    scipy_control=dict(control,text='from scipy.spatial import ConvexHull\nprint(5)')
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=dict(source_plan_sha256=E.file_hash(pool/'plan.json'),
                 responses_sha256=E.file_hash(cell/'responses.jsonl'),
                 item_id='abc312_e',scope='first testcase diagnostic; not a benchmark rescore',cases=[])
    env=dict(PATH='/usr/bin:/bin',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',LANG='C.UTF-8')
    cases=[('simple-control',control,True),('scipy-import-control',scipy_control,True),
           ('candidate-original',payload,False),('candidate-traced',payload,True)]
    for name,job,traced in cases:
        worker=Path(__file__).resolve() if traced else E.HERE/'internal_worker.py'
        command=[sys.executable,str(worker)]+(['--child'] if traced else [])
        with tempfile.TemporaryDirectory(prefix='lcb-diag-') as wd:
            with (args.out/(name+'.stdout')).open('wb') as stdout,(args.out/(name+'.stderr')).open('wb') as stderr:
                proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,
                    cwd=wd,env=dict(env,HOME=wd,TMPDIR=wd),start_new_session=True)
                timed_out=False
                try:proc.communicate(E.encoded(job),timeout=120)
                except subprocess.TimeoutExpired:timed_out=True
                finally:Q.stop_group(proc)
                result=dict(case=name,returncode=proc.returncode,outer_timeout=timed_out)
        receipt['cases'].append(result)
        Epath=args.out/'receipt.json';Q.atomic_json(Epath,receipt)
        print(json.dumps(result),flush=True)
        print((args.out/(name+'.stderr')).read_text(errors='replace')[-12000:],flush=True)
        print((args.out/(name+'.stdout')).read_text(errors='replace')[-2000:],flush=True)
    if E.file_hash(cell/'responses.jsonl')!=receipt['responses_sha256']:
        raise ValueError('Source journal changed during diagnostic')
    print('DIAGNOSTIC_OUTPUT',args.out,flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--child']:child()
    else:main()
