"""Add evaluation-seed dispersion without rewriting earlier training/eval history."""
import argparse,json,hashlib,statistics as S
from pathlib import Path
import wandb
from publish_wandb_eval import RUNS,PROJECT,BENCHES

def stats(v):
 return dict(mean=S.mean(v),std=S.stdev(v),variance=S.variance(v),min=min(v),max=max(v),**{f"seed{s}":x for s,x in zip((42,43,44),v)})
def values(checkpoint,benchmark):
 if benchmark=="macro": return [S.mean(checkpoint["benchmarks"][b]["scores"][i] for b in BENCHES) for i in range(3)]
 return checkpoint["benchmarks"][benchmark]["scores"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 data=json.loads(a.summary.read_text());assert data['seeds']==[42,43,44]
 digest=hashlib.sha256(a.summary.read_bytes()).hexdigest();a.out.mkdir(parents=True,exist_ok=True);api=wandb.Api(timeout=120);receipt=[]
 for mode,(rid,name) in RUNS.items():
  remote=api.run(PROJECT+'/'+rid);assert remote.name==name and remote.state=='finished'
  assert remote.summary.get('eval/source_sha256')==digest
  marker=remote.summary.get('eval/dispersion_sha256')
  if marker: assert marker==digest;continue
  assert not any(k.startswith('eval/dispersion/') for k in remote.summary.keys()),'partial import; inspect before retry'
  before=list(remote.scan_history(page_size=1000));selected=sorted((int(n.split('-')[-1]),v) for n,v in data['checkpoints'].items() if n.startswith(mode+'-'))
  run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume='must',dir=str(a.out),settings=wandb.Settings(x_disable_stats=True))
  try:
   run.define_metric('eval/dispersion_step')
   run.define_metric('eval/dispersion/*',step_metric='eval/dispersion_step')
   table=wandb.Table(columns=['step','benchmark','mean','std','variance','min','max','seed42','seed43','seed44'])
   for step,cp in selected:
    row={'eval/dispersion_step':step}
    for b in (*BENCHES,'macro'):
     z=stats(values(cp,b));table.add_data(step,b,*z.values())
     for k,v in z.items():row[f'eval/dispersion/{b}/{k}']=v
    run.log(row)
   plots={'eval/dispersion_table':table}
   for b in (*BENCHES,'macro'):
    xs=[s for s,_ in selected];zs=[stats(values(cp,b)) for _,cp in selected]
    ys=[[z[k] for z in zs] for k in ['mean','min','max']]+[[z['mean']-z['std'] for z in zs],[z['mean']+z['std'] for z in zs]]
    plots[f'eval/charts/{b}_spread']=wandb.plot.line_series(xs=xs,ys=ys,keys=['mean','min','max','mean - std','mean + std'],title=b+' evaluation seeds: spread (not CI)',xname='checkpoint step')
    plots[f'eval/charts/{b}_seeds']=wandb.plot.line_series(xs=xs,ys=[[z[f'seed{s}'] for z in zs] for s in (42,43,44)],keys=['seed 42','seed 43','seed 44'],title=b+' evaluation seeds',xname='checkpoint step')
    for k,v in zs[-1].items():run.summary[f'eval/endpoint_dispersion/{b}/{k}']=v
   run.log(plots)
   artifact=wandb.Artifact('eval-dispersion-'+rid+'-'+digest[:12],type='evaluation',metadata={'sample_variance_ddof':1,'units':'scores in [0,1]; variance in squared score units','scope':'three evaluation seeds; not confidence interval or training-seed variability','summary_sha256':digest})
   artifact.add(table,'dispersion');run.log_artifact(artifact).wait()
   run.summary['eval/dispersion_sha256']=digest
  finally:run.finish()
  api.flush();after=api.run(PROJECT+'/'+rid);history=list(after.scan_history(page_size=1000))
  assert history[:len(before)]==before
  assert after.summary.get('eval/dispersion_sha256')==digest
  for b in (*BENCHES,'macro'):
   for k,v in stats(values(selected[-1][1],b)).items():assert abs(after.summary[f'eval/endpoint_dispersion/{b}/{k}']-v)<1e-12
  assert 'evaluated' in after.tags and after.state=='finished'
  receipt.append({'mode':mode,'id':rid,'url':after.url,'history_preserved':True,'verified':True})
  (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('DISPERSION_VERIFIED',mode,flush=True)
if __name__=='__main__':main()
