#!/usr/bin/env python3
"""Stop exact old workers, fork verified journals, add SimCT; never launch GPUs."""
import argparse, contextlib, copy, hashlib, json, os, shutil, signal, sys, time
from pathlib import Path


def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        raw=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),ppid=int(raw[1]),group=int(raw[2]),start=raw[19],state=raw[0],
                    argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
    except (OSError, UnicodeError): return None


def stop_workers(plan,queue,pids):
    handles=[]; roots=[]
    try:
        for pid in pids:
            info=proc(pid)
            if info is None: continue
            args=info['argv']
            if str(queue) not in args or not any(x in args for x in ('worker','score-spool')) or '--plan' not in args or args[args.index('--plan')+1]!=str(plan):
                raise ValueError('PID identity mismatch: '+str(pid))
            fd=os.pidfd_open(pid); again=proc(pid)
            if not again or again['start']!=info['start']: raise ValueError('PID changed')
            handles.append(fd); roots.append(info)
        # Freeze only validated workers so they cannot start a new job during teardown.
        for fd in handles: signal.pidfd_send_signal(fd,signal.SIGSTOP)
        table={int(p.name):proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
        descendants={r['pid'] for r in roots}
        while True:
            more={pid for pid,x in table.items() if x and x['ppid'] in descendants}
            if more<=descendants: break
            descendants |= more
        children=[]
        for pid in descendants-{r['pid'] for r in roots}:
            before=table[pid]
            try: fd=os.pidfd_open(pid)
            except ProcessLookupError: continue
            after=proc(pid)
            if not after or after['start']!=before['start']: os.close(fd);continue
            children.append(fd)
        try:
            for fd in children+handles:
                try: signal.pidfd_send_signal(fd,signal.SIGTERM)
                except ProcessLookupError: pass
            for fd in handles:
                try: signal.pidfd_send_signal(fd,signal.SIGCONT)
                except ProcessLookupError: pass
            limit=time.monotonic()+30
            while time.monotonic()<limit:
                alive=[pid for pid in descendants if (x:=proc(pid)) and x['state']!='Z' and x['start']==table[pid]['start']]
                if not alive: return
                time.sleep(.5)
            raise RuntimeError('Some owned processes remain; no migration performed: '+str(alive))
        finally:
            for fd in children: os.close(fd)
    finally:
        for fd in handles:
            try: signal.pidfd_send_signal(fd,signal.SIGCONT)
            except ProcessLookupError: pass
            os.close(fd)


def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--source',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--worker-pids',nargs='+',type=int)
    a.add_argument('--queue-update',type=Path)
    a.add_argument('--simct',type=Path);args=a.parse_args()
    old=args.plan.resolve(strict=True); source=args.source.resolve(strict=True); out=args.out.resolve()
    if out.exists() or out.is_relative_to(old.parent): raise ValueError('Output must be a new sibling directory')
    sys.path.insert(0,str(source/'scripts/evaluation'))
    import eval_queue as Q
    E,D=Q.E,Q.D
    plan=E.read_json(old); oldhash=E.file_hash(old)
    if plan['source']!=D.script_hashes(): raise ValueError('Source differs from running plan')
    expected_count=25 if args.queue_update else 17
    allowed=('sft','atomic','fixed','simct') if args.queue_update else ('sft','atomic','fixed')
    if len(plan['jobs'])!=expected_count or any(j['mode'] not in allowed for j in plan['jobs']): raise ValueError('Unexpected old jobs')
    update_bytes=None
    if args.queue_update:
        update_bytes=args.queue_update.read_bytes()
        if plan['source']['eval_queue.py'] not in ('c64478394072d27fff38b432e6976e900a9b9a6a9065105590257ab34e8fa36d','616acf5249f5b7365b3f93ff7a17546c96b9d71ffdf5306b7981a6079bd0ce3e','af7cdaa0fab262f597babc64a17d5e1861ec2fbb35644612a096785962678a51','c4c23d92039237e6a0717f9221f302d4965df98a1e02544e869a3e79987cde6f','4645c8327a9610e1fc277a6b2b291e9162d07640b0d06714461779541d14040a','9d79384ae837af2b980d446e123c0f0cd5e5f396b74ffbf4da5d2c74ef2cf371'): raise ValueError('Unsupported original queue version')
        if hashlib.sha256(update_bytes).hexdigest()!='2e496d087348ef919f00dc585c273030068c557e81e4bef7a66be979ddef469d': raise ValueError('Unsupported queue update')
    initial_state=E.read_json(old.parent/'state.json')
    if time.time()>=initial_state['admit_until']: raise ValueError('Admission expired; workers left untouched')
    for j in plan['jobs']:
        if j['mode'] in D.RUNS and D.RUNS[j['mode']] not in Path(j['checkpoint']['path']).parts: raise ValueError('Wrong historical MP run')
    extra=[]
    if not args.queue_update:
        if args.simct is None: raise ValueError('--simct required')
        summary=E.read_json(args.simct/'run-summary.json')
        if summary['kd_algorithm']!='span_ctkd' or summary['status']!='completed' or summary['optimizer_updates']!=312: raise ValueError('SimCT summary invalid')
        if summary['student']!=next(j['checkpoint']['path'] for j in plan['jobs'] if j['mode']=='sft'): raise ValueError('Different SFT initialization')
        # Hash before stopping to avoid wasting idle GPU time on checkpoint inventory.
        for step in (312,156,80,240,40,200,120,280):
            print('HASH_SIMCT',step,flush=True)
            identity=E.checkpoint_identity(args.simct/f'step{step}')
            tier=next(j['tier'] for j in plan['jobs'] if j['mode']=='atomic' and j['step']==step)
            extra.append(dict(id=f'simct-{step}',mode='simct',step=step,tier=tier,checkpoint=identity))
    for data in plan['data'].values():
        if E.file_hash(data['path'])!=data['sha256']: raise ValueError('Data changed')
    pids=args.worker_pids
    if pids is None:
        pids=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            info=proc(int(entry.name))
            if info and any(x in info['argv'] for x in ('worker','score-spool')) and '--plan' in info['argv']:
                argv=info['argv']
                if argv[argv.index('--plan')+1]==str(old): pids.append(info['pid'])
        print('MATCHED_WORKERS',pids,flush=True)
    stop_workers(old,source/'scripts/evaluation/eval_queue.py',pids)
    with contextlib.ExitStack() as stack:
        for gpu in (0,1):
            if stack.enter_context(Q.locked(Path(f'/tmp/simct-eval-gpu{gpu}.lock'),blocking=False)) is None: raise ValueError('GPU worker still active')
        if stack.enter_context(Q.locked(old.parent/'scoring.lock',blocking=False)) is None: raise ValueError('Scorer still active')
        stack.enter_context(Q.locked(old.parent/'state.lock'))
        for j in plan['jobs']:
            if stack.enter_context(Q.locked(old.parent/(j['id']+'.lock'),blocking=False)) is None: raise ValueError('Job still active')
        state=E.read_json(old.parent/'state.json')
        if state['plan_sha256']!=oldhash or E.file_hash(old)!=oldhash: raise ValueError('Plan changed')
        if time.time()>=state['admit_until']: raise ValueError('Original admission window expired; no implicit extension')
        out.mkdir()
        receipt={'from_plan':str(old),'old_plan_sha256':oldhash,'source':str(source),'concurrency':16,'original_files':{},'status':'incomplete'}
        E.write_new(out/'migration.json',receipt)
        new=copy.deepcopy(plan);new['jobs']+=extra;new['jobs'].sort(key=lambda j:j['tier'])
        if update_bytes is not None:
            target=out/'source'
            shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','__pycache__','remote_artifacts'))
            (target/'scripts/evaluation/eval_queue.py').write_bytes(update_bytes)
            new['source']={name:E.file_hash(target/'scripts/evaluation'/name) for name in plan['source']}
            if {k for k in new['source'] if new['source'][k]!=plan['source'][k]}!={'eval_queue.py'}: raise ValueError('Unexpected source changes')
            receipt.update(source=str(target),previous_source=str(source),concurrency_by_gpu={"0":256,"1":128},score_workers=16,score_buffer=128)
            receipt['execution']='separate-generation-and-scoring'
            receipt.pop('concurrency',None)
            new['execution_transition']={'generation_concurrency_by_gpu':{'0':256,'1':128},'score_workers':16,'score_buffer':128,'old_source':plan['source']}

        new['migration']={'from_plan':str(old),'sha256':oldhash,'reason':('Decouple generation/scoring; preserve journals and clock' if update_bytes is not None else 'Add eight SimCT checkpoints; preserve journals and original clock')}
        E.write_new(out/'plan.json',new);newhash=E.file_hash(out/'plan.json')
        if (old.parent/'cells').exists(): shutil.copytree(old.parent/'cells',out/'cells')
        jobs={j['id']:j for j in plan['jobs']};datasets={b:{x['id']:x for x in E.read_json(d['path'])['items']} for b,d in plan['data'].items()}
        totals={'responses':0,'scores':0,'metrics':0}
        for f in (old.parent/'cells').rglob('*'):
            if f.is_file(): receipt['original_files'][str(f.relative_to(old.parent))]=E.file_hash(f)
        for cell in (out/'cells').glob('*/*/*'):
            if not cell.is_dir(): continue
            jid,b,seed=cell.relative_to(out/'cells').parts;seed=int(seed)
            if jid not in jobs or b not in datasets or seed not in plan['seeds']: raise ValueError('Unknown cell')
            contract=E.read_json(cell/'contract.json')
            expected=Q.cell_contract(oldhash,jobs[jid],plan['data'],b,seed,contract['server'])
            if contract!=expected: raise ValueError('Old cell contract mismatch')
            metrics=E.read_json(cell/'metrics.json') if (cell/'metrics.json').exists() else None
            if metrics:
                if metrics['contract']!=contract or metrics['status']!='completed': raise ValueError('Old metrics mismatch')
                for name in ('responses','scores'):
                    if E.file_hash(cell/(name+'.jsonl'))!=metrics[name+'_sha256']: raise ValueError('Completed journal changed')
            responses=Q.journal(cell/'responses.jsonl');scores=Q.journal(cell/'scores.jsonl');items=datasets[b]
            if not set(scores)<=set(responses)<=set(items): raise ValueError('Unknown result ID')
            for key,row in responses.items():
                payload=E.generation_payload('eval-gemma',items[key],b,seed)
                if row['seed']!=seed or row['request_sha256']!=E.digest(E.encoded(payload)): raise ValueError('Request mismatch')
                E.validate_response(row['response'])
            for key,row in scores.items():
                if type(row.get('passed')) is not bool or row['response_sha256']!=E.digest(E.encoded(responses[key])): raise ValueError('Score mismatch')
            if metrics:
                if len(scores)!=len(items) or metrics['count']!=len(items) or metrics['score']!=sum(x['passed'] for x in scores.values())/len(items): raise ValueError('Metric count/score mismatch')
                metrics['contract']={**contract,'plan_sha256':newhash};Q.atomic_json(cell/'metrics.json',metrics);totals['metrics']+=1
            marker=cell/'generation-complete.json'
            if marker.exists():
                value=E.read_json(marker)
                if value['contract']!=contract or value['count']!=len(items) or len(responses)!=len(items) or value['responses_sha256']!=E.file_hash(cell/'responses.jsonl'): raise ValueError('Generation marker mismatch')
                value['contract']={**contract,'plan_sha256':newhash};Q.atomic_json(marker,value)
            Q.atomic_json(cell/'contract.json',{**contract,'plan_sha256':newhash})
            totals['responses']+=len(responses);totals['scores']+=len(scores)
        for rel,digest in receipt['original_files'].items():
            if E.file_hash(old.parent/rel)!=digest: raise ValueError('Source journal changed during migration')
        state['plan_sha256']=newhash
        state['jobs']={k:v for k,v in state['jobs'].items() if v['status']=='completed'}
        for jid in state['jobs']:
            if not all((out/'cells'/jid/b/str(s)/'metrics.json').exists() for b in E.CAPS for s in plan['seeds']): raise ValueError('Completed job lacks cells')
        E.write_new(out/'state.json',state)
        if (old.parent/'generation-state.json').exists():
            gs=E.read_json(old.parent/'generation-state.json')
            if gs['plan_sha256']!=oldhash: raise ValueError('Generation state mismatch')
            gs['plan_sha256']=newhash
            gs['jobs']={k:v for k,v in gs['jobs'].items() if v['status']=='completed'}
            for jid in gs['jobs']:
                if not all(any((out/'cells'/jid/b/str(seed)/name).exists() for name in ('metrics.json','generation-complete.json')) for b in E.CAPS for seed in plan['seeds']): raise ValueError('Generation job lacks complete cells')
            E.write_new(out/'generation-state.json',gs)

        receipt.update(status='verified',new_plan_sha256=newhash,totals=totals,deadline=state['deadline'])
        Q.atomic_json(out/'migration.json',receipt)
        print('MIGRATION_PASS',json.dumps(totals),flush=True);print('NEW_PLAN='+str(out/'plan.json'));print('NEW_SOURCE='+receipt['source']);print('No GPU launched. Original deadline retained.')

if __name__=='__main__': main()
