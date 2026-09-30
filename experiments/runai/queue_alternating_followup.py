"""Queue a matched frozen adapter control, paired reference evaluation and report."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = Path('/workspace/storage-shared/nlp/tungks/SimCT/python-b200.sh')

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(p)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8388608), b''):
            h.update(chunk)
    return h.hexdigest()

def frozen_command(case):
    cmd = read(case/'command.json')
    if '--freeze-energy' in cmd or '--resume' in cmd:
        raise ValueError('Expected fresh alternating source command')
    cmd[cmd.index('--output')+1] = str(case/'followup/frozen')
    return cmd+['--freeze-energy']

def validate_pair(a,b,data_sha):
    for p in (a,b):
        if p['schema']!='mp-alternating-resume-v2' or p['step']!=50:
            raise ValueError('Both runs must complete exactly 50 valid updates')
        if p['manifest']['data_sha256']!=data_sha:
            raise ValueError('Dataset identity mismatch')
        if p['cursor']!=len(p['results']) or p['cursor']!=len(p['traces']):
            raise ValueError('Checkpoint cursor mismatch')
        if [r['index'] for r in p['results']]!=list(range(p['cursor'])):
            raise ValueError('Noncontiguous results')
    if a['energy_updates']!=50 or b['energy_updates']!=0:
        raise ValueError('Expected alternating vs frozen energy')
    ma,mb=a['manifest'],b['manifest']
    ignored={'output','freeze_energy','resume','stop_after_groups'}
    aa={k:v for k,v in ma['args'].items() if k not in ignored}
    ab={k:v for k,v in mb['args'].items() if k not in ignored}
    if aa!=ab:
        raise ValueError('Unmatched training arguments')
    for key in ('models','source_commit','source_diff_sha256','resume_runtime','resume_source_files'):
        if ma[key]!=mb[key]:
            raise ValueError('Unmatched provenance: '+key)
    if ma['args'].get('freeze_energy') or not mb['args'].get('freeze_energy'):
        raise ValueError('Wrong energy modes')
    return max(a['cursor'],b['cursor'])

def reserved_rows(data,cursor):
    rows=[]
    for i,g in enumerate(data['groups'][cursor:],start=cursor):
        refs=g['eval'] if isinstance(g['eval'],list) else [g['eval']]
        for row in refs:
            rows.append(dict(row,group=i))
    if not rows:
        raise ValueError('No untouched reference groups remain; do not reuse training groups')
    return rows

def frozen(case):
    cmd=frozen_command(case)
    dest=case/'followup/frozen'
    if dest.exists():
        if not (dest/'latest.pt').is_file():
            raise ValueError('Incomplete setup without checkpoint; inspect before retry')
        cmd+=['--resume']
    subprocess.run(cmd,check=True)
    s=read(dest/'summary.json')
    if s['status']!='completed' or s['student_updates']!=50 or s['energy_updates']!=0:
        raise ValueError('Frozen control did not complete')

def evaluate(case):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.mp_opd.real_oracle import chat_prompt_ids,token_ids,validate_groups
    from kdflow.algorithms._mp_opd_credit import realized_token_log_probs
    paths={'alternating':case/'run/latest.pt','frozen':case/'followup/frozen/latest.pt'}
    checkpoints={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
    a,b=checkpoints['alternating'],checkpoints['frozen']
    data_path=Path(a['manifest']['args']['data'])
    data=read(data_path);validate_groups(data['groups'])
    cursor=validate_pair(a,b,sha(data_path))
    if not torch.equal(a['a'],b['a']):
        raise ValueError('Different adapter A initialization')
    rows=reserved_rows(data,cursor)
    args=a['manifest']['args'];model_path=Path(args['student'])
    # Verify exact base-model files, not merely directory names.
    for name,expected in a['manifest']['models']['student'].items():
        if sha(model_path/name)!=expected:
            raise ValueError('Base-model file changed: '+name)
    if torch.__version__!=a['manifest']['torch']:
        raise ValueError('Torch version differs from training')
    dtype=getattr(torch,a['manifest']['effective_model_dtype'].removeprefix('torch.'))
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    encoded=[]
    for row in rows:
        prefix=chat_prompt_ids(tok,row['messages'],args['max_prompt_tokens'])
        response=token_ids(tok.encode(row['reference'],add_special_tokens=False))
        if tok.eos_token_id is not None:response.append(tok.eos_token_id)
        if len(response)>args['max_reference_tokens']:
            raise ValueError('Reference exceeds cap; no truncation')
        encoded.append((prefix,response))
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,torch_dtype=dtype,attn_implementation='eager').to('cuda:0').eval()
    model.requires_grad_(False)
    module=args['adapter_module'];original=model.get_submodule(module)
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=original
            self.register_buffer('a',a['a'].to('cuda:0'))
            self.register_buffer('b',torch.zeros_like(a['b'],device='cuda:0'))
        def forward(self,x):
            z=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)
            return self.base(x)+(z/args['rank']).to(x.dtype)
    parent,child=module.rsplit('.',1);adapter=Adapter()
    setattr(model.get_submodule(parent),child,adapter)
    output=[{'id':r['id'],'group':r['group'],'tokens':len(t[1])} for r,t in zip(rows,encoded)]
    for mode in ('initial','alternating','frozen'):
        with torch.no_grad():
            if mode=='initial':adapter.b.zero_()
            else:adapter.b.copy_(checkpoints[mode]['b'].to('cuda:0'))
            for row,(prefix,response) in zip(output,encoded):
                ids=torch.tensor([prefix+response],device='cuda:0')
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
                selected=logits[0,len(prefix)-1:len(prefix)+len(response)-1]
                row[mode]=-float(realized_token_log_probs(selected,torch.tensor(response,device='cuda:0')).mean())
                if not math.isfinite(row[mode]):raise ValueError('Nonfinite evaluation NLL')
        print('EVAL_COMPLETE',mode,len(rows),flush=True)
    result={'scope':'paired reference NLL; teacher pseudo-labels; one training seed; no benchmark efficacy claim',
        'data_sha256':sha(data_path),'checkpoint_sha256':{k:sha(p) for k,p in paths.items()},
        'first_reserved_group':cursor,'groups':len(set(r['group'] for r in rows)),
        'references':len(rows),'rows':output,'torch':torch.__version__,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    write(case/'followup/paired-eval.json',result)

def report(case):
    e=read(case/'followup/paired-eval.json');rows=e['rows']
    means={mode:sum(r[mode] for r in rows)/len(rows) for mode in ('initial','alternating','frozen')}
    delta=[r['alternating']-r['frozen'] for r in rows]
    summaries={mode:read(case/path/'summary.json') for mode,path in [('alternating','run'),('frozen','followup/frozen')]}
    result={'scope':e['scope'],'lower_nll_is_better':True,'groups':e['groups'],'references':len(rows),
        'mean_reference_nll':means,'alternating_minus_frozen':sum(delta)/len(delta),
        'alternating_lower_fraction':sum(d<0 for d in delta)/len(delta),'training':summaries,
        'note':'Reference rows are correlated within groups; no independent-sample significance claim.'}
    write(case/'followup/report.json',result);print(json.dumps(result,indent=2))

def active_states(proc_root=Path('/proc')):
    states=set()
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            parts=path.read_bytes().decode().strip('\0').split('\0')
        except (OSError,UnicodeError):continue
        if 'job_manager' not in parts or 'run' not in parts or '--state' not in parts:continue
        i=parts.index('--state')
        if i+1<len(parts):states.add(str(Path(parts[i+1]).resolve()))
    return sorted(states)

def choose_state(args):
    import fcntl,time
    manager=args.manager.resolve();case=args.case.resolve()
    sys.path.insert(0,str(manager))
    from job_manager.store import initialize,connect
    from job_manager.__main__ import snapshot
    if args.state:return args.state.resolve()
    folder=case/'followup';folder.mkdir(exist_ok=True)
    with (folder/(socket.gethostname()+'.manager.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        found=active_states()
        if len(found)>1:raise ValueError('Multiple active manager states; supply --state: '+str(found))
        if found:return Path(found[0])
        state=Path('/var/tmp')/('alt-followup-'+hashlib.sha256((str(case)+socket.gethostname()).encode()).hexdigest()[:16])
        if not (state/'queue.sqlite3').exists():
            initialize(state,dict(gpus=[args.gpu_uuid],cpu_slots=4,deadline=None,
                sample_seconds=5,poll_seconds=1,kill_grace_seconds=15,low_util_percent=5,
                max_external_memory_mib=1024,max_admission_util_percent=5,
                lease_dir=f'/tmp/job-manager-{os.getuid()}-gpu-leases'))
        db=connect(state)
        try:
            if not snapshot(db)['manager']['running']:
                with (folder/'manager.log').open('ab') as log:
                    process=subprocess.Popen(['/usr/bin/python3.12','-m','job_manager','--state',str(state),'run'],
                        cwd=manager,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                for _ in range(20):
                    if process.poll() is not None:raise ValueError('Manager exited; inspect manager.log')
                    if snapshot(db)['manager']['running']:break
                    time.sleep(.5)
                else:raise ValueError('Manager startup not confirmed; inspect manager.log')
        finally:db.close()
        return state

def submit(args):
    case=args.case.resolve();manager=args.manager.resolve()
    sys.path.insert(0,str(manager))
    from job_manager.store import connect,rows,submit as put
    from job_manager.__main__ import snapshot
    case_summary=read(case/'run/summary.json')
    if case_summary['status']!='completed' or case_summary['student_updates']!=50:
        raise ValueError('Alternating50 not complete')
    if (case/'exit-code.txt').read_text().strip()!='0':raise ValueError('Alternating50 exit not zero')
    identity=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if identity!=args.gpu_uuid:raise ValueError('GPU0 UUID mismatch')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Dirty followup source')
    follow=case/'followup';follow.mkdir(exist_ok=True)
    key='alt50-'+hashlib.sha256(str(case).encode()).hexdigest()[:10]
    env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(ROOT/'experiments/modal/vendor')+':'+str(ROOT)}
    jobs=[]
    for i,phase in enumerate(('frozen','evaluate','report')):
        jobs.append(dict(version=1,id=key+'-'+phase,project='alternating50-followup',cwd=str(ROOT),
            argv=['bash',str(WRAPPER),'-u',str(Path(__file__).resolve()),phase,'--case',str(case)],
            env=env,gpus=[args.gpu_uuid] if phase!='report' else 0,cpu_slots=1,
            timeout_seconds=7200,dependencies=[] if i==0 else [jobs[-1]['id']],dependency_policy='success'))
    state=choose_state(args)
    if not (state/'queue.sqlite3').is_file():raise ValueError('Manager state missing')
    db=connect(state)
    try:
        snap=snapshot(db)
        if not snap['manager']['running'] or snap['paused'] or snap['quarantined']:
            raise ValueError('Existing manager is not ready; no manager started here')
        existing={j['id']:j for j in rows(db)}
        for j in jobs:
            if j['id'] in existing and existing[j['id']]['spec']!=j:
                raise ValueError('Existing job spec differs; no duplicate submitted')
        pending=[j for j in jobs if j['id'] not in existing]
        if pending:put(db,pending)
        write(follow/'queue-receipt.json',{'state':str(state),'host':socket.gethostname(),
            'manager':str(manager),'source':str(ROOT),'jobs':jobs})
        print(json.dumps(snapshot(db),indent=2))
        print('REPORT_PATH',follow/'report.json')
    finally:db.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['frozen','evaluate','report','submit'])
    p.add_argument('--case',type=Path,required=True);p.add_argument('--state',type=Path)
    p.add_argument('--manager',type=Path);p.add_argument('--gpu-uuid')
    args=p.parse_args()
    if args.action=='submit':
        if not args.manager:p.error('submit requires manager source')
        if not args.gpu_uuid:
            args.gpu_uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        submit(args)
    else:globals()[args.action](args.case.resolve())

if __name__=='__main__':main()
