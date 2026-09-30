"""Fresh guide-energy training, technical qualification, soft student canary/pilot."""
import argparse, copy, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def source_commit():
    explicit=os.environ.get("KDFLOW_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        # Source bundles used by pinned runtimes may omit .git.
        return "unavailable"

def qualify(work):
    summary=json.loads((work/'energy/summary.json').read_text())
    rows=[json.loads(x) for x in (work/'energy/results.jsonl').read_text().splitlines()]
    valid=[x for x in rows if not x['invalid']]
    if summary['completed_groups']!=32 or len(valid)<24:raise ValueError('Incomplete/insufficient valid guide groups')
    deltas=[x['energy_updates']['4']['parameter_delta'] for x in valid]
    if not all(math.isfinite(x) for x in deltas) or not any(x>0 for x in deltas):raise ValueError('Energy did not update')
    if any(x['learned_partition']['coverage_max_error']>1e-3 for x in valid):raise ValueError('Invalid partition coverage')
    sys.path.insert(0,str(ROOT))
    os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"
    import torch
    from kdflow.algorithms._mp_opd_energy import MPAtomEnergy,load_energy_checkpoint
    net=MPAtomEnergy(10,32,2);opt=torch.optim.AdamW(net.parameters())
    step=load_energy_checkpoint(work/'energy/energy-select-4.pt',net,opt,expected_extra_config={'max_span_length':2})
    if step<=0 or not all(torch.isfinite(p).all() for p in net.parameters()):raise ValueError('Invalid energy checkpoint')
    report=dict(technical_pass=True,energy_updates=step,valid_groups=len(valid),evidence='Technical only; eval scores do not determine pilot admission',checkpoint_sha256=hashlib.sha256((work/'energy/energy-select-4.pt').read_bytes()).hexdigest())
    (work/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

def prepare(prior, original, out):
    old=json.loads((prior/'plan.json').read_text()); state=json.loads((prior/'campaign-state.json').read_text())
    if state['jobs']['probe-1']['status']!='completed':raise ValueError('Primary probe not completed')
    deadline=state['started']+old['hours']*3600
    if deadline-time.time()<7200:raise ValueError('Less than two hours remain; do not reset clock')
    cfg=json.loads((original/'config.json').read_text())
    job=copy.deepcopy(next(x for x in old['jobs'] if x['id']=='probe-1'))
    argv=job['argv']; raw=Path(argv[argv.index('--data')+1]).read_bytes()
    expected=json.loads((prior/'preflight.json').read_text())['data_sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Guide hash changed')
    out.mkdir(parents=True,exist_ok=False);(out/'probe-data.json').write_bytes(raw)
    host=str(ROOT/'experiments/runai/python-b200-host.sh')
    argv[1]=host;argv[2]=str(ROOT/'experiments/mp_opd/real_oracle.py')
    argv[argv.index('--data')+1]=str(out/'probe-data.json')
    argv[argv.index('--output')+1]=str(out/'energy')
    argv[argv.index('--weighting-steps')+1]='4'
    argv+=['--learn-partition','--norm-controls','--energy-steps','4','--energy-lr','0.001','--energy-temperature','1.0']
    job.update(id='energy',gpu=1,budget_seconds=2700)
    jobs=[job,dict(id='qualify',gpu=None,require_success=['energy'],budget_seconds=180,argv=['bash',host,str(Path(__file__).resolve()),'qualify','--work',str(out)])]
    env=dict(MP_ALGORITHM='mp_opd',MP_STUDENT_PATH=cfg['student'],MP_TEACHER_PATH=cfg['teacher'],MP_DATASET_PATH=cfg['dataset'],MP_SEED='43',MP_PARTITION_SEED='43',MP_MAX_SPAN_LENGTH='2',MP_FIXED_SPAN_LENGTH='2',MP_PREFLIGHT_ONLY='0',MP_ENERGY_CHECKPOINT=str(out/'energy/energy-select-4.pt'))
    for name,limit,budget,dep in [('soft-canary','5',1200,'qualify'),('soft-pilot','50',5400,'soft-canary')]:
        jobs.append(dict(id=name,gpu=1,require_success=[dep],budget_seconds=budget,env=dict(env,MP_RUN_ROOT=str(out/name)),argv=['bash',str(ROOT/'experiments/runai/run_single_gpu.sh'),'1','soft',limit]))
    plan=dict(hours=old['hours'],source=str(ROOT),source_commit=source_commit(),jobs=jobs)
    path=out/'plan.json';path.write_text(json.dumps(plan,indent=2))
    (out/'campaign-state.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),started=state['started'],jobs={}),indent=2))
    print('READY',out,'ORIGINAL_DEADLINE',deadline,flush=True)

def recover(prior, out):
    old=json.loads((prior/'plan.json').read_text())
    state=json.loads((prior/'campaign-state.json').read_text())
    if state['jobs']['energy']['status']!='completed':raise ValueError('Energy must be completed')
    remaining=int(state['started']+old['hours']*3600-time.time())
    if remaining<2400:raise ValueError('Less than 40 minutes remain; preserve original clock')
    out.mkdir(parents=True,exist_ok=False)
    (out/'energy').symlink_to(prior/'energy',target_is_directory=True)
    jobs=copy.deepcopy([j for j in old['jobs'] if j['id']!='energy'])
    for j in jobs:
        j['argv']=[str(x).replace(old['source'],str(ROOT)).replace(str(prior),str(out)) for x in j['argv']]
        if j['id']=='qualify':j.pop('require_success',None)
        if 'env' in j:
            j['env']={k:str(v).replace(str(prior),str(out)) for k,v in j['env'].items()}
        if j['id']=='soft-pilot':j['budget_seconds']=min(5400,remaining-180-1200-180)
    plan=dict(hours=old['hours'],source=str(ROOT),source_commit=source_commit(),jobs=jobs)
    path=out/'plan.json';path.write_text(json.dumps(plan,indent=2))
    (out/'campaign-state.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),started=state['started'],jobs={}),indent=2))
    print('READY',out,'REUSED_ENERGY',prior/'energy','PILOT_BUDGET_SECONDS',jobs[-1]['budget_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','qualify','recover']);p.add_argument('--work',type=Path,required=True);p.add_argument('--prior',type=Path);p.add_argument('--original',type=Path)
    a=p.parse_args()
    if a.action=='qualify':qualify(a.work.resolve())
    elif a.action=='recover':recover(a.prior.resolve(),a.work.resolve())
    else:prepare(a.prior.resolve(),a.original.resolve(),a.work.resolve())
