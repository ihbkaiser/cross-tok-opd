"""Read-only audit of aggregate report metadata and per-evaluation-seed macro."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

BENCHES=('gsm8k','math500','mbpp','live-code-bench-v6')

def macro_by_seed(cell):
    """Return paired macro replicates; never treat benchmark names as seeds."""
    maps=[cell.get(b,{}).get('seeds',{}) for b in BENCHES]
    common=set(maps[0]).intersection(*maps[1:])
    return {s:statistics.fmean(m[s] for m in maps) for s in sorted(common)
        if s.isdigit() and all(isinstance(m[s],(int,float)) for m in maps)}

def audit(payload):
    rows=[];issues=[]
    for group,runs in payload['groups'].items():
        for run in runs:
            rid=run.get('id') or run.get('run') or ''
            match=re.search(r'(?:^|-)s(\d+)(?:-|$)',rid)
            named_seed=match.group(1) if match else None
            declared=str(run.get('train_seed',''))
            if named_seed and named_seed!=declared:
                issues.append({'kind':'training_seed_conflict','group':group,'run':rid,
                    'declared':declared,'id_seed':named_seed,'needs_launch_config_confirmation':True})
            for step,cell in run.get('steps',{}).items():
                if any(cell.get(b,{}).get('partial') for b in (*BENCHES,'macro')):
                    continue
                values=macro_by_seed(cell)
                if len(values)<2:continue
                mean=statistics.fmean(values.values());std=statistics.stdev(values.values())
                macro=cell.get('macro',{});reported=macro.get('eval_std')
                rows.append({'group':group,'run':rid,'declared_train_seed':declared,'id_train_seed':named_seed,
                    'step':step,'macro_mean_recomputed':mean,'macro_std_recomputed':std,
                    'reported_macro_std':reported,'paired_eval_replicates':len(values),
                    **{'macro_seed_'+s:v for s,v in values.items()}})
                if isinstance(reported,(int,float)) and abs(std-reported)>1e-8:
                    issues.append({'kind':'macro_std_mismatch','group':group,'run':rid,'step':step,
                        'reported':reported,'recomputed':std})
                # Historical table imports rounded benchmark means to six decimals.
                if isinstance(macro.get('mean'),(int,float)) and abs(mean-macro['mean'])>5e-7:
                    issues.append({'kind':'macro_mean_mismatch','group':group,'run':rid,'step':step,
                        'reported':macro['mean'],'recomputed':mean})
    return rows,issues

def main():
    p=argparse.ArgumentParser();p.add_argument('payload',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    rows,issues=audit(json.loads(a.payload.read_text(encoding='utf-8')))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'report-statistics-audit.json').write_text(json.dumps({'issues':issues,'rows':rows},indent=2))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output/'macro-recomputed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    from collections import Counter
    print(json.dumps({'cells_recomputed':len(rows),'issue_counts':dict(Counter(i['kind'] for i in issues))}))

if __name__=='__main__':main()
