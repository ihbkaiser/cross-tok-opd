"""Verify and append supplemental training evidence to existing campaign runs."""
import argparse, copy, hashlib, json, netrc, os, tarfile
from pathlib import Path, PurePosixPath
from audit_campaign_export import audit
from publish_campaign_recovery import PROJECT, dump
from align_campaign_wandb_schema import tags_for

def unpack(archive, expected, dest):
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected: raise ValueError('archive checksum mismatch')
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or '\\' in m.name:
                raise ValueError('unsafe archive member')
        manifest=json.load(tar.extractfile('manifest.json'))
        if manifest.get('errors'):raise ValueError('collection errors')
        entries={e['member']:e for e in manifest['files']}
        if set(names)!=set(entries)|{'manifest.json'}:raise ValueError('manifest membership mismatch')
        for m in members:
            raw=tar.extractfile(m).read()
            if m.name!='manifest.json':
                e=entries[m.name]
                if e.get('changed_during_collection') or len(raw)!=e['bytes'] or hashlib.sha256(raw).hexdigest()!=e['sha256']:
                    raise ValueError(('file verification failed',m.name))
            p=dest/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=raw:raise ValueError('existing evidence conflict')
            p.write_bytes(raw)
    return digest,manifest

def missing_training(rows, history):
    observed={}
    for row in history:
        if row.get('train/global_step') is None:continue
        step=int(row['train/global_step'])
        for k,v in row.items():
            if k.startswith('train/') and k!='train/global_step' and v is not None:
                if (step,k) in observed and observed[step,k]!=v:raise ValueError(('remote training conflict',step,k))
                observed[step,k]=v
    missing=[]
    for row in rows:
        step=row['step'];new={'train/global_step':step}
        for k,v in row.items():
            if k=='step':continue
            key='train/'+k
            if (step,key) in observed:
                if abs(observed[step,key]-v)>1e-12:raise ValueError(('source training conflict',step,key))
            else:new[key]=v
        if len(new)>1:missing.append(new)
    return missing

def plain(value):
    if hasattr(value,'keys'):return {k:plain(value[k]) for k in value.keys()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
    out=a.base/'extra-training';out.mkdir(exist_ok=True)
    digest,manifest=unpack(a.archive,a.sha256,out/'verified-evidence')
    extra=audit(out/'verified-evidence',None)
    if extra['issues']:raise ValueError(extra['issues'])
    original=json.loads((a.base/'analysis/audit.json').read_text());merged=copy.deepcopy(original)
    roots={r['alias']:r['path'] for r in manifest['roots']}
    for name,r in extra['runs'].items():
        old=merged['runs'][name]
        if (old['method'],old['train_seed'])!=(r['method'],r['train_seed']):raise ValueError('identity mismatch')
        member=next(e for e in r['evidence'] if e.endswith('launch-config.json'))
        alias,relative=member.split('/',1);source_run=roots[alias]+'/'+relative.rsplit('/',1)[0]
        if any(not v.startswith(source_run+'/checkpoint') for v in old['checkpoint_paths'].values()):raise ValueError('checkpoint provenance mismatch')
        for key in ('launch','target','rows','summary','exitcode','last_step','training_status','missing_log_steps','log_sha256'):
            old[key]=r[key]
        old['supplemental_evidence']={'archive_sha256':digest,'members':r['evidence']}
        print('VERIFIED',name,r['train_seed'],r['last_step'],r['training_status'],'missing',len(r['missing_log_steps']),flush=True)
    # Training evidence enrichment does not recompute existing evaluation aggregates.
    assert merged['groups']==original['groups'] and merged['points']==original['points']
    dump(out/'audit.json',merged)
    dump(out/'verification.json',{'sha256':digest,'files':len(manifest['files']),'runs':list(extra['runs']),'eval_unchanged':True})
    if not a.publish:return
    import wandb
    os.environ['WANDB_API_KEY']=netrc.netrc('/mnt/c/Users/Tung/_netrc').authenticators('api.wandb.ai')[2]
    api=wandb.Api(timeout=120);settings=wandb.Settings(x_disable_stats=True,disable_git=True,disable_code=True,quiet=True)
    ids=json.loads((a.base/'wandb-v2/receipts.json').read_text());receipts={}
    for name in extra['runs']:
        r=merged['runs'][name];rid=ids[name]['id'];remote=api.run(PROJECT+'/'+rid)
        if remote.name!=name or remote.group!=r['group']:raise ValueError('remote identity mismatch')
        history=list(remote.scan_history(page_size=1000));before=out/'before'/rid;before.mkdir(parents=True,exist_ok=True)
        if not (before/'history.json').exists():
            dump(before/'history.json',history);dump(before/'metadata.json',{'config':dict(remote.config),'summary':plain(remote.summary),'tags':remote.tags,'group':remote.group})
        eval_before={k:v for k,v in plain(remote.summary).items() if k.startswith(('eval/','eval_v2/'))}
        for attempt in range(3):
            missing=missing_training(r['rows'],history)
            if attempt and not missing:break
            run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume='must',dir=str(out),settings=settings)
            try:
                run.define_metric('train/global_step');run.define_metric('train/*',step_metric='train/global_step')
                for row in missing:run.log(row)
                if attempt==0:
                    cfg=copy.deepcopy(dict(remote.config));cv=cfg['campaign_v2'];cv.update(source_training_status=r['training_status'],logged_updates=r['last_step'],target_updates=r['target'])
                    ex=cfg['experiment'];ex.update(source_status=r['training_status'],configuration_evidence='verified supplemental launch')
                    train=dict(cfg.get('train',{}));train['seed']=r['train_seed']
                    run.config.update({'campaign_v2':cv,'experiment':ex,'train':train,'supplemental_training':{'archive_sha256':digest,'launch':r['launch'],'source_log_sha256':r['log_sha256']}},allow_val_change=True)
                    run.summary.update({'recovery_v2/training_status':r['training_status'],'recovery_v2/logged_updates':r['last_step'],'recovery_v2/source_log_sha256':r['log_sha256'],'recovery_extra/archive_sha256':digest})
                    evidence=out/(rid+'.json');dump(evidence,r)
                    artifact=wandb.Artifact(rid+'-extra-training',type='training-evidence',metadata={'sha256':digest})
                    artifact.add_file(str(evidence),name='training.json');run.log_artifact(artifact).wait()
            finally:run.finish()
            api.flush();remote=api.run(PROJECT+'/'+rid);history=list(remote.scan_history(page_size=1000))
        if missing_training(r['rows'],history):raise ValueError('readback still missing')
        assert eval_before=={k:v for k,v in plain(remote.summary).items() if k.startswith(('eval/','eval_v2/'))}
        remote.tags=tags_for(r,remote.tags);remote.update()
        receipts[name]={'id':rid,'url':remote.url,'training_steps':len(r['rows']),'status':r['training_status'],'verified_all_training_values':True,'eval_unchanged':True,'group':remote.group}
        dump(out/'receipts.json',receipts);print('UPLOAD_VERIFIED',rid,flush=True)
    prior=a.base/'analysis/audit-before-extra.json'
    if not prior.exists():dump(prior,original)
    dump(a.base/'analysis/audit.json',merged)
    from campaign_dashboard import render
    render(merged,a.base/'analysis/dashboard.html')

if __name__=='__main__':main()