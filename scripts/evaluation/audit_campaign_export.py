"""Audit exported logs and journals without executing generated model code."""
import argparse, collections, hashlib, json, math, re, statistics
from pathlib import Path

BENCHES = ('gsm8k','math500','mbpp','live-code-bench-v6')
COUNTS = dict(zip(BENCHES,(1319,500,500,1055)))
SEEDS = (42,43,44)
PAIR = re.compile(r'([A-Za-z_][\w/]*):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf)(?=,|$)')

def sha(data): return hashlib.sha256(data).hexdigest()
def read(p): return json.loads(p.read_text())

def training_rows(text):
    rows, issues = {}, []
    for line in text.splitlines():
        if 'on_policy_kd_trainer.py:logging:' not in line or 'step [' not in line: continue
        step = int(re.search(r'step \[(\d+)/',line)[1])
        values = {k:float(v) for k,v in PAIR.findall(line)}
        bad = [k for k,v in values.items() if not math.isfinite(v)]
        if bad: issues.append({'step':step,'nonfinite':bad})
        values = {k:v for k,v in values.items() if math.isfinite(v)}
        if step in rows and rows[step] != values: issues.append({'step':step,'conflicting_duplicate':True})
        rows[step] = values
    return [{'step':s,**v} for s,v in sorted(rows.items())],issues

def aggregate(cells):
    """Each independent training run contributes one mean over all 3 eval seeds."""
    by_run = collections.defaultdict(dict)
    for c in cells:
        if c['valid']: by_run[(c['run'],c['step'],c['benchmark'])][c['eval_seed']] = c['score']
    result=[]
    for (run,step,b),values in sorted(by_run.items()):
        complete = set(values)==set(SEEDS)
        result.append({'run':run,'step':step,'benchmark':b,'eval_scores':values,
                       'eval_n':len(values),'mean':statistics.mean(values.values()) if complete else None,
                       'eval_std':statistics.stdev(values.values()) if complete else None})
    return result

def grouped(points,runs):
    buckets=collections.defaultdict(list)
    for p in points:
        run=runs[p['run']]
        if p['mean'] is not None and run.get('primary') and run.get('train_seed') is not None:
            buckets[(run['group'],p['step'],p['benchmark'])].append((run['train_seed'],p['mean'],p['run']))
    result=[]
    for (group,step,b),v in sorted(buckets.items()):
        seeds=[x[0] for x in v]
        if len(seeds)!=len(set(seeds)): raise ValueError(('duplicate training seed in group',group,step,seeds))
        vals=[x[1] for x in v]
        result.append({'group':group,'step':step,'benchmark':b,'train_n':len(v),'train_seeds':seeds,
                       'mean':statistics.mean(vals),'train_std':statistics.stdev(vals) if len(v)>1 else None,
                       'run_means':{x[2]:x[1] for x in v}})
    return result

def audit(root,backup):
    manifest=read(root/'manifest.json'); paths={e['member']:root/e['member'] for e in manifest['files']}
    remote={r['name']:r for r in read(backup)} if backup else {}
    runs, issues, cells = {}, [], []
    def make(name,mode,seed=None):
        if name not in runs:
            runs[name]={'name':name,'method':mode,'train_seed':seed,'rows':[], 'training_status':'unverified',
                        'primary':True,'group':'company-v2-'+mode,'evidence':[], 'remote_id':remote.get(name,{}).get('id')}
        return runs[name]
    for member,p in paths.items():
        if p.name!='launch-config.json': continue
        d=read(p); o=d['options']; mode='simct' if o['kd_algorithm']=='span_ctkd' else o['mp_opd_mode']
        r=make(p.parent.name,mode,o.get('seed'));r['launch']=d;r['evidence'].append(member)
        r['target']=o.get('diagnostic_max_updates') or 312
        r['primary']=r['target']==312
        if not r['primary']:r['group']+=f'-pilot{r["target"]}'
        if mode=='soft':r['group']+='-frozen-'+str(d.get('energy_sha256','unknown'))[:8]
        log=Path(str(p.parent)+'.log'); summary=p.parent/'checkpoint/run-summary.json'; ex=Path(str(p.parent)+'.exitcode')
        if log.exists():
            r['rows'],bad=training_rows(log.read_text(errors='replace'));r['log_sha256']=sha(log.read_bytes())
            r['evidence'].append(str(log.relative_to(root)));issues.extend({'run':r['name'],'training':x} for x in bad)
        r['summary']=read(summary) if summary.exists() else None
        r['exitcode']=int(ex.read_text().strip()) if ex.exists() else None
        for f in (summary,ex):
            if f.exists():r['evidence'].append(str(f.relative_to(root)))
        r['last_step']=max((v['step'] for v in r['rows']),default=0)
        complete=(r['summary'] or {}).get('status')=='completed' and r['exitcode']==0 and r['last_step']==r['target']
        r['training_status']='completed' if complete else 'failed' if r['exitcode'] not in (None,0) else 'interrupted_or_unclosed'
        r['missing_log_steps']=sorted(set(range(1,r['last_step']+1))-{v['step'] for v in r['rows']})
        if complete and (r['missing_log_steps'] or r['summary']['optimizer_updates']!=r['target']):
            r['training_status']='inconsistent';issues.append({'run':r['name'],'error':'completion evidence inconsistent'})
    plans=[(k,p) for k,p in paths.items() if p.name=='plan.json' and
           (k=='root1/plan.json' or k=='root0/aligned-seed43-pool/plan.json' or
            k=='root0/soft50-next-v1/eval/plan.json' or k.startswith('root0/six-b200-v1/pool/'))]
    for member,p in plans:
        plan=read(p)
        for j in plan['jobs']:
            ck=j['checkpoint']['path'];name=ck.split('/checkpoint')[0].rsplit('/',1)[-1]
            seed=43 if 'seed43-' in j['id'] else None
            r=make(name,j['mode'],seed)
            if r['train_seed'] is None:
                config=remote.get(name,{}).get('config',{})
                r['train_seed']=config.get('train',{}).get('seed',config.get('seed'))
            if r['train_seed'] is None and member=='root1/plan.json' and j['mode']!='sft':
                # Confirmed legacy source config is required before including in aggregates.
                r['seed_unverified']=True;r['primary']=False
            if '20260911-042532-895995' in name:
                r['primary']=False;r['group']='company-v2-random-early-attempt'
            r.setdefault('checkpoint_paths',{})[str(j['step'])]=ck
            for b in BENCHES:
                for seed in SEEDS:
                    cell=p.parent/'cells'/j['id']/b/str(seed);metrics=cell/'metrics.json'
                    c={'run':name,'step':j['step'],'benchmark':b,'eval_seed':seed,'profile':plan['profile'],
                       'cell':str(cell.relative_to(root)),'valid':False,'generated':(cell/'generation-complete.json').exists()}
                    if metrics.exists():
                        d=read(metrics); journal=cell/'scores.jsonl'; c['score']=d['score'];c['count']=d['count']
                        c['responses_sha256']=d.get('responses_sha256');c['scores_sha256']=d.get('scores_sha256')
                        reasons=[]
                        if d.get('status')!='completed':reasons.append('not completed')
                        if d['count']!=COUNTS[b]:reasons.append('count mismatch')
                        if not math.isfinite(d['score']) or not 0<=d['score']<=1:reasons.append('invalid score')
                        contract=d['contract'];c['checkpoint_sha256']=contract['checkpoint_sha256']
                        if contract['checkpoint_sha256']!=j['checkpoint']['sha256']:reasons.append('checkpoint mismatch')
                        if contract['seed']!=seed or contract['benchmark']!=b:reasons.append('cell contract mismatch')
                        if journal.exists():
                            raw=journal.read_bytes();rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
                            if sha(raw)!=d['scores_sha256']:reasons.append('journal hash mismatch')
                            ids=[x['id'] for x in rows]
                            if len(ids)!=len(set(ids)) or len(rows)!=COUNTS[b]:reasons.append('journal IDs/count mismatch')
                            if rows and abs(sum(bool(x['passed']) for x in rows)/len(rows)-d['score'])>1e-12:reasons.append('journal mean mismatch')
                        else:reasons.append('journal missing')
                        c['valid']=not reasons
                        if reasons:issues.append({'cell':c['cell'],'errors':reasons})
                    else:c['missing_metrics']=True
                    cells.append(c)
    # Enrich existing legacy run rows from exported logs; never replay to W&B.
    for name,r in runs.items():
        lp=root/'root2'/name/'train.log'
        if not r['rows'] and lp.exists():
            text=lp.read_text(errors='replace')
            rows,bad=training_rows(text);r['rows']=rows
            r['evidence'].append(str(lp.relative_to(root)));r['last_step']=max((x['step'] for x in rows),default=0)
            match=re.search(r'(?<![\w])seed=(\d+)',text)
            if r['train_seed'] is None and match:
                r['train_seed']=int(match[1]);r.pop('seed_unverified',None);r['primary']=True
        if r['training_status']=='unverified' and name in remote:
            r['training_status']='previously_imported_'+remote[name]['state']
        r['eval_cells']=sum(c['valid'] for c in cells if c['run']==name)
    for member,p in paths.items():
        if p.name in ('score-error.json','generation-error.json','lcbfix-error.json'):
            issues.append({'marker':member,'error':read(p).get('error')})
    points=aggregate(cells)
    # Macro requires all benchmarks and all evaluation seeds for a checkpoint.
    by=collections.defaultdict(dict)
    for p in points:by[(p['run'],p['step'])][p['benchmark']]=p
    for (run,step),v in by.items():
        if set(v)==set(BENCHES) and all(p['mean'] is not None for p in v.values()):
            vals={s:statistics.mean(v[b]['eval_scores'][s] for b in BENCHES) for s in SEEDS}
            points.append({'run':run,'step':step,'benchmark':'macro','eval_scores':vals,'eval_n':3,
                           'mean':statistics.mean(vals.values()),'eval_std':statistics.stdev(vals.values())})
    return {'schema':2,'scope':'raw scorer; LCBfix failures retained; eval seeds are not training replicates',
            'runs':runs,'cells':cells,'points':points,'groups':grouped(points,runs),'issues':issues,
            'transfer_sha256':'8c7889ad1138457157dc7e21ed10d36e53f06af55a4636c07334fac49c459efe'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--backup',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    data=audit(a.root,a.backup);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'audit.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    for r in data['runs'].values():print(r['method'],r['train_seed'],r['training_status'],r.get('last_step'),r['eval_cells'],r['name'])
    print('TOTAL',len(data['cells']),'VALID',sum(c['valid'] for c in data['cells']),'ISSUES',len(data['issues']))
if __name__=='__main__':main()
