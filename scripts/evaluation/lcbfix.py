"""Versioned LCB rescore of immutable historical generations. CPU only."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import sys
import time
import eval_queue as Q
import queue_data as D
from lcbfix_worker import extract

E = Q.E


def read_only(path):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in result:
            raise ValueError('Duplicate ID: '+str(path))
        result[row['id']] = row
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--internal-code-execution',action='store_true',required=True)
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('Workers must be 1..16')
    src = args.plan.resolve().parent
    out = args.out.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise ValueError('Output must be separate from source pool')
    plan = E.read_json(args.plan)
    if plan['profile'] != D.PROFILE or plan['seeds'] != [42,43,44]:
        raise ValueError('Expected historical internal profile and three eval seeds')
    # The only changed scoring logic is the versioned extractor.
    for name, digest in D.script_hashes().items():
        if name != 'eval_queue.py' and plan['source'].get(name) != digest:
            raise ValueError('Historical scoring source drift: '+name)
    data = plan['data']['live-code-bench-v6']
    if E.file_hash(data['path']) != data['sha256']:
        raise ValueError('Dataset changed')
    items = {i['id']:i for i in E.read_json(data['path'])['items']}
    contract = dict(version='lcbfix-v1',source_plan_sha256=E.file_hash(args.plan),
                    dataset_sha256=data['sha256'],seeds=plan['seeds'],
                    scripts={name:E.file_hash(E.HERE/name) for name in
                        ('lcbfix.py','lcbfix_worker.py','internal_worker.py','eval_queue.py')},
                    policy='reuse identical extracted code; rescore changed code; no regeneration')
    out.mkdir(parents=True,exist_ok=True)
    with Q.locked(out/'worker.lock',blocking=False) as lock:
        if lock is None:
            raise ValueError('Another lcbfix coordinator is active')
        cp = out/'contract.json'
        if cp.exists():
            if E.read_json(cp)!=contract:
                raise ValueError('Output contract mismatch')
        else:
            E.write_new(cp,contract)
        qualification = Q.preflight(sys.executable)
        if qualification != E.read_json(src/'state.json')['qualification']:
            raise ValueError('Historical grader runtime qualification changed')
        for code, expected in [('print(5)',True),('print(0)',False)]:
            payload = dict(profile=D.PROFILE,benchmark='live-code-bench-v6',
                           item=dict(tests=[dict(input='',output='5\n')],fn_name=None),
                           text='```python3\n'+code+'\n```')
            result = Q.score_job(sys.executable,payload,time.time()+180,worker=E.HERE/'lcbfix_worker.py')
            if result['passed'] != expected or result.get('timeout'):
                raise ValueError('LCBfix worker qualification failed')
        helpers = E.author_helpers()
        summary = []
        for job in plan['jobs']:
            for seed in plan['seeds']:
                cell = src/'cells'/job['id']/'live-code-bench-v6'/str(seed)
                metrics = E.read_json(cell/'metrics.json')
                if metrics['status']!='completed':
                    raise ValueError('Source cell incomplete: '+str(cell))
                old_contract = metrics['contract']
                if old_contract != Q.cell_contract(contract['source_plan_sha256'],job,plan['data'],
                                                    'live-code-bench-v6',seed,old_contract['server']):
                    raise ValueError('Source metric contract mismatch')
                for name in ('responses','scores'):
                    if E.file_hash(cell/(name+'.jsonl')) != metrics[name+'_sha256']:
                        raise ValueError('Source journal hash mismatch')
                responses = read_only(cell/'responses.jsonl')
                scores = read_only(cell/'scores.jsonl')
                if set(responses)!=set(scores) or set(scores)!=set(items):
                    raise ValueError('Source IDs differ from dataset')
                if metrics['count']!=len(items) or metrics['score']!=sum(r['passed'] for r in scores.values())/len(items):
                    raise ValueError('Source score/count mismatch')
                target = out/'cells'/job['id']/str(seed)
                target.mkdir(parents=True,exist_ok=True)
                identity = dict(contract=contract,source_metrics_sha256=E.file_hash(cell/'metrics.json'))
                identity_path = target/'contract.json'
                if identity_path.exists():
                    if E.read_json(identity_path)!=identity:
                        raise ValueError('Source metrics changed across resume')
                else:
                    E.write_new(identity_path,identity)
                existing = Q.journal(target/'scores.jsonl')
                if not set(existing)<=set(items):
                    raise ValueError('Unknown resumed ID')
                def grade(key):
                    row = responses[key]
                    if scores[key]['response_sha256'] != E.digest(E.encoded(row)):
                        raise ValueError('Source response/score mismatch')
                    raw = row['response']['choices'][0]['message']['content']
                    text = helpers['strip_thinking_content'](raw)[0]
                    changed = extract(text)!=helpers['_extract_code_block'](text)
                    if changed:
                        payload = dict(profile=D.PROFILE,benchmark='live-code-bench-v6',
                                       item=D.score_item('live-code-bench-v6',items[key]),text=raw)
                        result = Q.score_job(sys.executable,payload,time.time()+180,
                                             worker=E.HERE/'lcbfix_worker.py')
                    else:
                        result = dict(passed=scores[key]['passed'],timeout=scores[key].get('timeout',False))
                    return dict(result,id=key,old_passed=scores[key]['passed'],extraction_changed=changed,
                                provenance='rescored' if changed else 'identical-code-reuse',
                                response_sha256=E.digest(E.encoded(row)))
                with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    for row in executor.map(grade,[key for key in items if key not in existing]):
                        Q.append(target/'scores.jsonl',row)
                        existing[row['id']] = row
                count = len(existing)
                old = sum(r['old_passed'] for r in existing.values())
                new = sum(r['passed'] for r in existing.values())
                report = dict(job=job['id'],eval_seed=seed,count=count,old_passed=old,new_passed=new,
                              old_score=old/count,new_score=new/count,
                              changed=sum(r['extraction_changed'] for r in existing.values()),
                              gained=sum(r['passed'] and not r['old_passed'] for r in existing.values()),
                              lost=sum(r['old_passed'] and not r['passed'] for r in existing.values()),
                              scores_sha256=E.file_hash(target/'scores.jsonl'))
                Q.atomic_json(target/'metrics.json',report)
                summary.append(report)
                Q.atomic_json(out/'summary.json',dict(status='running',cells=summary))
                print('LCBFIX_CELL',json.dumps(report),flush=True)
        Q.atomic_json(out/'summary.json',dict(status='completed',cells=summary))
        print('LCBFIX_COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
