"""Align recovered runs with the existing native eval tables/scalars and tag schema."""
import argparse, collections, json, netrc, os, statistics, sys
from pathlib import Path
from publish_campaign_recovery import PROJECT, dump, signature
from audit_campaign_export import BENCHES, SEEDS
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kdflow.wandb_schema import build_wandb_tags

CHECKPOINT_COLUMNS=['checkpoint_step','benchmark','mean','sample_std','seed42','seed43','seed44']
DISPERSION_COLUMNS=['step','benchmark','mean','std','variance','min','max','seed42','seed43','seed44']
def native_payload(points):
    rows={};checkpoints=[];dispersion=[]
    for p in sorted(points,key=lambda x:(x['step'],x['benchmark'])):
        step,b=p['step'],p['benchmark'];values=[p['eval_scores'].get(str(s),p['eval_scores'].get(s)) for s in SEEDS]
        row=rows.setdefault(step,{'eval/checkpoint_step':step,'eval/dispersion_step':step})
        for s,v in zip(SEEDS,values):
            if v is not None:row[f'eval/dispersion/{b}/seed{s}']=v
        row[f'eval/completeness/{b}/eval_n']=p['eval_n']
        if p['mean'] is not None:
            z=[p['mean'],p['eval_std'],statistics.variance(values),min(values),max(values),*values]
            for k,v in zip(DISPERSION_COLUMNS[2:],z):row[f'eval/dispersion/{b}/{k}']=v
            row['eval/curve/'+b]=p['mean'];dispersion.append([step,b,*z])
        else:dispersion.append([step,b,None,None,None,None,None,*values])
        if b!='macro':checkpoints.append([step,b,p['mean'],p['eval_std'],*values])
    return rows,checkpoints,dispersion

def existing_values(history):
    result={}
    for row in history:
        for key,value in row.items():
            if value is None:continue
            axis='eval/dispersion_step' if key.startswith('eval/dispersion/') else 'eval/checkpoint_step'
            if key.startswith(('eval/curve/','eval/dispersion/','eval/completeness/')) and row.get(axis) is not None:
                index=(int(row[axis]),key)
                if index in result and abs(result[index]-value)>1e-12:raise ValueError(('conflicting existing value',index))
                result[index]=value
    return result

def missing_rows(rows,history):
    old=existing_values(history);out=[]
    for step,row in rows.items():
        add={'eval/checkpoint_step':step,'eval/dispersion_step':step}
        for k,v in row.items():
            if k in add:continue
            if (step,k) in old:
                if abs(old[step,k]-v)>1e-12:raise ValueError(('protocol/value conflict',step,k,old[step,k],v))
            else:add[k]=v
        if len(add)>2:out.append(add)
    return out

def tags_for(r,current):
    if r['method']=='sft':return list(dict.fromkeys([*current,'campaign:company-b200-202609','eval-protocol:company-internal-v1','eval-replicates:3']))
    mp=r['method']!='simct';target=r.get('target') or (312 if r['primary'] else 'unknown')
    canonical=build_wandb_tags(method='mp-opd' if mp else 'simct',regime='on-policy',
        objective='path-credit' if mp else 'span-ctkd',platform='runai',accelerator='b200x1',
        budget=f'{target}-update',stage='train',student='gemma2-sft-paper',teacher='qwen2.5-7b-instruct',
        variant='fixed2' if r['method']=='fixed' else r['method'],extras=[('import','historical-log'),
        ('campaign','company-b200-202609'),('train-seed',str(r['train_seed']) if r['train_seed'] is not None else 'unknown'),
        ('eval-replicates','3'),('eval-protocol','company-internal-v1')]).split(',')
    # Replace only dimensions standardized here; keep other legacy information.
    dimensions={t.split(':')[0] for t in canonical}
    return list(dict.fromkeys([t for t in current if ':' not in t or t.split(':')[0] not in dimensions]+canonical+(['evaluated'] if r['eval_cells'] else [])))

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--audit',type=Path,required=True);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--publish',action='store_true');args=pa.parse_args()
    import wandb
    os.environ['WANDB_API_KEY']=netrc.netrc('/mnt/c/Users/Tung/_netrc').authenticators('api.wandb.ai')[2]
    data=json.loads(args.audit.read_text());receipts=json.loads((args.out/'receipts.json').read_text());api=wandb.Api(timeout=120)
    settings=wandb.Settings(x_disable_stats=True,disable_git=True,disable_code=True,quiet=True)
    template=json.loads((args.out/'before/mpbackfill-ec7c67c50e92a2e6/metadata.json').read_text())['config']
    donepath=args.out/'native-receipts.json';done=json.loads(donepath.read_text()) if donepath.exists() else {}
    for name,r in data['runs'].items():
        rid=receipts[name]['id'];remote=api.run(PROJECT+'/'+rid);points=[p for p in data['points'] if p['run']==name]
        rows,cp,dp=native_payload(points);digest=signature({'rows':rows,'group':r['group'],'schema':'native-eval-v1'})
        if name in done and remote.summary.get('eval/recovery_native_sha256')==digest:continue
        backup=args.out/'native-before'/rid;backup.mkdir(parents=True,exist_ok=True)
        history=list(remote.scan_history(page_size=1000))
        if not (backup/'metadata.json').exists():
            dump(backup/'metadata.json',{'config':dict(remote.config),'summary':dict(remote.summary),'tags':remote.tags,'group':remote.group,'raw_config':remote._attrs.get('config')})
            dump(backup/'history.json',history)
        new=missing_rows(rows,history);print('NATIVE_READY',rid,len(new),flush=True)
        if not args.publish:continue
        run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume='must',dir=str(args.out),settings=settings)
        try:
            options=r.get('launch',{}).get('options',{})
            config={}
            for section in ('kd','train','data','model','rollout'):
                value=dict(remote.config.get(section,{}))
                value.update({key:options[key] for key in template[section] if key in options})
                if section=='train' and r['train_seed'] is not None:value['seed']=r['train_seed']
                if value:config[section]=value
            config.update({'experiment':{'schema':'experiment-v1','method':'mp-opd' if r['method'] in ('atomic','fixed','random','soft') else r['method'],
                'variant':r['method'],'training_seed':r['train_seed'],'evaluation_seeds':list(SEEDS),'group':r['group'],
                'source_status':r['training_status'],'configuration_evidence':'launch' if options else 'legacy metadata/plan; launch unavailable'},'historical_import':True})
            run.config.update(config,allow_val_change=True)
            run.define_metric('eval/checkpoint_step');run.define_metric('eval/curve/*',step_metric='eval/checkpoint_step')
            run.define_metric('eval/dispersion_step');run.define_metric('eval/dispersion/*',step_metric='eval/dispersion_step')
            run.define_metric('eval/completeness/*',step_metric='eval/checkpoint_step')
            for row in new:run.log(row)
            # Same columns and chart keys as publish_wandb_eval / publish_wandb_uncertainty.
            tables={'eval/checkpoints':wandb.Table(columns=CHECKPOINT_COLUMNS,data=cp),
                    'eval/dispersion_table':wandb.Table(columns=DISPERSION_COLUMNS,data=dp)}
            for b in (*BENCHES,'macro'):
                pp=[p for p in points if p['benchmark']==b];pp.sort(key=lambda p:p['step'])
                if not pp:continue
                xs=[];ys=[]
                for seed in SEEDS:
                    xy=[(p['step'],p['eval_scores'].get(str(seed),p['eval_scores'].get(seed))) for p in pp]
                    xy=[(x,y) for x,y in xy if y is not None];xs.append([x for x,y in xy]);ys.append([y for x,y in xy])
                tables[f'eval/charts/{b}_seeds']=wandb.plot.line_series(xs=xs,ys=ys,keys=['seed 42','seed 43','seed 44'],title=b+' evaluation seeds',xname='checkpoint step')
                complete=[p for p in pp if p['mean'] is not None]
                if complete:
                    vals=[list(p['eval_scores'].values()) for p in complete];means=[p['mean'] for p in complete];stds=[p['eval_std'] for p in complete]
                    tables[f'eval/charts/{b}_spread']=wandb.plot.line_series(xs=[p['step'] for p in complete],
                        ys=[means,[min(v) for v in vals],[max(v) for v in vals],[m-s for m,s in zip(means,stds)],[m+s for m,s in zip(means,stds)]],
                        keys=['mean','min','max','mean - std','mean + std'],title=b+' evaluation seeds: spread (not CI)',xname='checkpoint step')
                    end=complete[-1];run.summary['eval/endpoint/'+b]=end['mean'];run.summary['eval/endpoint_std/'+b]=end['eval_std']
            if points:run.log(tables)
            run.summary.update({'eval/profile':'company-internal-v1','eval/seeds':list(SEEDS),
                'eval/scope':data['scope'],'eval/recovery_native_sha256':digest,'eval/training_seed':r['train_seed']})
        finally:run.finish()
        api.flush();remote=api.run(PROJECT+'/'+rid);remote.group=r['group'];remote.tags=tags_for(r,remote.tags);remote.update()
        after=list(remote.scan_history(page_size=1000));assert not missing_rows(rows,after),('native readback missing',rid)
        done[name]={'id':rid,'group':remote.group,'verified':True,'table_columns':CHECKPOINT_COLUMNS,'appended_steps':len(new)};dump(donepath,done)
        print('NATIVE_VERIFIED',rid,flush=True)
if __name__=='__main__':main()
