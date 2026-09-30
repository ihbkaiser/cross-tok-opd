"""Back up existing W&B runs and append seed-aware campaign evidence idempotently."""
import argparse, collections, hashlib, json, netrc, os, time
from pathlib import Path
from campaign_dashboard import render
from audit_campaign_export import BENCHES, SEEDS

PROJECT='kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction'
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False,default=str))
def signature(d):return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()
def history_set(rows):return {json.dumps(r,sort_keys=True,default=str) for r in rows}

def verify_training(api, wandb, rid, rows, out, settings):
    source={x["step"]:x for x in rows if "loss" in x}
    repaired=[]
    for attempt in range(3):
        api.flush()
        remote=api.run(PROJECT+"/"+rid)
        history=list(remote.scan_history(page_size=1000))
        observed={int(x["train/global_step"]):x for x in history if "train/global_step" in x and "train/loss" in x}
        missing=sorted(set(source)-set(observed))
        for step in set(source)&set(observed):
            assert abs(observed[step]["train/loss"]-source[step]["loss"])<1e-12
        if not missing:return history,repaired
        if attempt==2:raise ValueError((rid,"training rows still missing",missing))
        # Append only absent source steps; custom x-axis preserves training order.
        run=wandb.init(entity=PROJECT.split("/")[0],project=PROJECT.split("/")[1],id=rid,
                       resume="must",dir=str(out),settings=settings)
        try:
            for step in missing:
                row=source[step]
                run.log({"train/global_step":step,**{"train/"+k:v for k,v in row.items() if k!="step"}})
            repaired.extend(missing)
        finally:run.finish()

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--only',help='Exact source run name');a=p.parse_args()
    import wandb
    auth=netrc.netrc('/mnt/c/Users/Tung/_netrc').authenticators('api.wandb.ai');assert auth
    os.environ['WANDB_API_KEY']=auth[2]
    data=json.loads(a.audit.read_text());a.out.mkdir(parents=True,exist_ok=True)
    api=wandb.Api(timeout=120)
    existing={r.name:r for r in api.runs(PROJECT,filters={'displayName':{'$in':list(data['runs'])}})}
    receipt_path=a.out/'receipts.json';receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    settings=wandb.Settings(x_disable_stats=True,disable_git=True,disable_code=True,quiet=True)
    for name,r in data['runs'].items():
        if a.only and name!=a.only:continue
        points=[p for p in data['points'] if p['run']==name]
        payload={'run':r,'points':points,'scope':data['scope'],'archive':data['transfer_sha256']}
        digest=signature(payload);remote=existing.get(name)
        rid=remote.id if remote else 'campaign-'+hashlib.sha256(name.encode()).hexdigest()[:16]
        marker='recovery_v2/source_sha256'
        if remote and remote.summary.get(marker)==digest:
            if name not in receipts:
                if not a.publish:raise ValueError(("unverified prior upload",rid))
                history,repaired=verify_training(api,wandb,rid,r["rows"],a.out,settings)
                observed_eval={int(x["eval_v2/checkpoint_step"]):x for x in history if "eval_v2/checkpoint_step" in x and any(k.startswith("eval_v2/raw/") for k in x)}
                for point in points:
                    for seed,v in point["eval_scores"].items():
                        assert abs(observed_eval[point["step"]]["eval_v2/raw/"+point["benchmark"]+"/eval_seed"+str(seed)]-v)<1e-12
                remote.group=r["group"];remote.update()
                receipts[name]={"id":rid,"url":remote.url,"verified":True,"reconciled_training_steps":repaired,"group":r["group"]}
                dump(receipt_path,receipts)
            continue
        if remote and remote.summary.get(marker):raise ValueError(('changed prior import',name))
        backup=a.out/'before'/rid;backup.mkdir(parents=True,exist_ok=True)
        before=[]
        if remote:
            history_file=backup/'history.json'
            if not history_file.exists():
                before=list(remote.scan_history(page_size=1000));dump(history_file,before)
                dump(backup/'metadata.json',{'id':rid,'name':remote.name,'group':remote.group,'state':remote.state,
                    'config':dict(remote.config),'summary':dict(remote.summary),'tags':remote.tags,'notes':remote.notes,
                    'raw_config':remote._attrs.get('config')})
                files=[]
                for f in remote.files():
                    files.append({'name':f.name,'size':f.size,'md5':f.md5})
                    if f.name.startswith('media/') and f.size<20*1024*1024:
                        f.download(root=str(backup/'files'),replace=False)
                dump(backup/'files.json',files)
            else:before=json.loads(history_file.read_text())
            if any(any(k.startswith('eval_v2/') or k.startswith('recovery_v2/') for k in row) for row in before):
                raise ValueError(('partial previous import needs reconciliation',name))
        print('READY',rid,r['method'],r['train_seed'],len(r['rows']),len(points),'existing',bool(remote),flush=True)
        if not a.publish:continue
        evidence=a.out/(rid+'-evidence.json');dump(evidence,payload)
        config={'campaign_v2':{'method':r['method'],'training_seed':r['train_seed'],'evaluation_seeds':list(SEEDS),
            'group':r['group'],'primary':r['primary'],'source_training_status':r['training_status'],
            'target_updates':r.get('target'),'logged_updates':r.get('last_step'),'source_commit':r.get('launch',{}).get('source_commit'),
            'energy_sha256':r.get('launch',{}).get('energy_sha256'),'archive_sha256':data['transfer_sha256'],
            'scope':data['scope'],'timestamps':'Historical text-log import; upload time is not training time',
            'telemetry':'Logged GPU metrics are node snapshots and may include concurrent jobs'}}
        run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,name=name,
            group=r['group'],resume='must' if remote else 'never',job_type='historical-import' if r['rows'] else 'evaluation-evidence',
            dir=str(a.out),settings=settings)
        try:
            run.config.update(config,allow_val_change=False)
            if not remote:
                run.define_metric('train/global_step');run.define_metric('train/*',step_metric='train/global_step')
                for row in r['rows']:
                    run.log({'train/global_step':row['step'],**{'train/'+k:v for k,v in row.items() if k!='step'}})
            run.define_metric('eval_v2/checkpoint_step')
            run.define_metric('eval_v2/raw/*',step_metric='eval_v2/checkpoint_step')
            by_step=collections.defaultdict(list)
            for point in points:by_step[point['step']].append(point)
            table=wandb.Table(columns=['checkpoint_step','benchmark','training_seed','eval_n','eval42','eval43','eval44','mean','eval_sample_std'])
            for step,pp in sorted(by_step.items()):
                row={'eval_v2/checkpoint_step':step}
                for point in pp:
                    prefix='eval_v2/raw/'+point['benchmark']+'/'
                    for s,v in point['eval_scores'].items():row[prefix+'eval_seed'+str(s)]=v
                    row[prefix+'eval_n']=point['eval_n']
                    if point['mean'] is not None:
                        row[prefix+'mean']=point['mean'];row[prefix+'eval_std']=point['eval_std']
                    table.add_data(step,point['benchmark'],r['train_seed'],point['eval_n'],
                        *[point['eval_scores'].get(str(s),point['eval_scores'].get(s)) for s in SEEDS],point['mean'],point['eval_std'])
                run.log(row)
            plots={'eval_v2/checkpoints':table}
            single=a.out/(rid+'-dashboard.html')
            render({'runs':{name:r},'points':points},single)
            plots['eval_v2/interactive']=wandb.Html(single.read_text(),inject=False)
            for b in (*BENCHES,'macro'):
                pp=sorted([p for p in points if p['benchmark']==b],key=lambda p:p['step'])
                if not pp:continue
                xs=[];ys=[]
                for s in SEEDS:
                    values=[(p['step'],p['eval_scores'].get(str(s),p['eval_scores'].get(s))) for p in pp]
                    values=[(x,y) for x,y in values if y is not None]
                    xs.append([x for x,y in values]);ys.append([y for x,y in values])
                plots['eval_v2/charts/'+b+'_eval_seeds']=wandb.plot.line_series(xs=xs,ys=ys,
                    keys=['eval seed '+str(s) for s in SEEDS],title=b+' raw score: 3 eval seeds of ONE training run',xname='checkpoint step')
            run.log(plots)
            artifact=wandb.Artifact(rid+'-recovery-v2',type='evaluation',metadata={'source_sha256':digest,'scope':data['scope']})
            artifact.add_file(str(evidence),name='evidence.json');artifact.add(table,'eval-table')
            run.log_artifact(artifact).wait()
            run.summary.update({'recovery_v2/training_status':r['training_status'],'recovery_v2/logged_updates':r.get('last_step'),
                'recovery_v2/valid_eval_cells':r['eval_cells'],'recovery_v2/source_log_sha256':r.get('log_sha256'),marker:digest})
        finally:run.finish()
        api.flush();after=api.run(PROJECT+'/'+rid)
        after.group=r['group'];after.tags=list(dict.fromkeys([*after.tags,'campaign-recovery-v2','eval-seeds-not-training-seeds']));after.update()
        history=list(after.scan_history(page_size=1000))
        assert history_set(before)<=history_set(history),'old history changed'
        assert after.summary[marker]==digest
        if not remote:
            history,repaired=verify_training(api,wandb,rid,r["rows"],a.out,settings)
        observed_eval={int(x['eval_v2/checkpoint_step']):x for x in history if 'eval_v2/checkpoint_step' in x and any(k.startswith('eval_v2/raw/') for k in x)}
        for point in points:
            row=observed_eval[point['step']]
            for s,v in point['eval_scores'].items():assert abs(row['eval_v2/raw/'+point['benchmark']+'/eval_seed'+str(s)]-v)<1e-12
        dump(a.out/(rid+'-history-after.json'),history)
        receipts[name]={'id':rid,'url':after.url,'verified':True,'old_history_preserved':bool(remote),'training_rows':len(r['rows']),
                        'eval_points':len(points),'group':after.group,'source_sha256':digest}
        dump(receipt_path,receipts);print('UPLOAD_VERIFIED',rid,flush=True)
    if not a.publish or a.only:return
    dashboard=a.out/'campaign-dashboard.html';render(data,dashboard)
    rid='campaign-recovery-20260914-v2';digest=signature({'points':data['points'],'groups':data['groups']})
    matches=list(api.runs(PROJECT,filters={'name':rid}))
    if matches and matches[0].summary.get('source_sha256')==digest:return
    run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,
        name='Campaign recovery · training seed groups · 2026-09-14',group='campaign-analysis-v2',
        job_type='analysis',resume='must' if matches else 'never',dir=str(a.out),settings=settings)
    try:
        run.log({'eval_v2/interactive_dashboard':wandb.Html(dashboard.read_text(),inject=False)})
        table=wandb.Table(columns=['group','step','benchmark','training_n','training_seeds','mean','training_sample_std'],
            data=[[g['group'],g['step'],g['benchmark'],g['train_n'],str(g['train_seeds']),g['mean'],g['train_std']] for g in data['groups']])
        run.log({'eval_v2/training_seed_aggregation':table})
        artifact=wandb.Artifact('campaign-recovery-20260914',type='analysis')
        artifact.add_file(str(a.audit),name='audit.json');artifact.add_file(str(dashboard),name='dashboard.html')
        run.log_artifact(artifact).wait();run.summary['source_sha256']=digest
    finally:run.finish()
    api.flush();after=api.run(PROJECT+'/'+rid);assert after.summary['source_sha256']==digest
    dump(a.out/'dashboard-receipt.json',{'id':rid,'url':after.url,'verified':True})
    print('DASHBOARD_VERIFIED',after.url,flush=True)

if __name__=='__main__':main()
