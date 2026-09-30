import hashlib,json,time
from pathlib import Path
from experiments.runai.learned_partition_pilot import prepare

def test_pilot_gpu1_only_dependencies_and_original_clock(tmp_path):
    prior=tmp_path/'prior';prior.mkdir();original=tmp_path/'original';original.mkdir()
    data=prior/'data.json';data.write_text('{"groups": []}')
    started=time.time()-100
    argv=['bash','old-host','old-oracle','run','--data',str(data),'--output','old-out','--weighting-steps','1']
    (prior/'plan.json').write_text(json.dumps(dict(hours=20,jobs=[dict(id='probe-1',argv=argv,gpu=0)])))
    (prior/'campaign-state.json').write_text(json.dumps(dict(started=started,jobs={'probe-1':{'status':'completed'}})))
    (prior/'preflight.json').write_text(json.dumps(dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())))
    (original/'config.json').write_text(json.dumps(dict(student='s',teacher='t',dataset='d')))
    out=tmp_path/'out';prepare(prior,original,out)
    plan=json.loads((out/'plan.json').read_text())
    assert [j['gpu'] for j in plan['jobs']]==[1,None,1,1]
    assert plan['jobs'][2]['require_success']==['qualify']
    assert plan['jobs'][3]['require_success']==['soft-canary']
    assert json.loads((out/'campaign-state.json').read_text())['started']==started
    assert '--learn-partition' in plan['jobs'][0]['argv']
    assert plan['jobs'][3]['argv'][-2:]==['soft','50']


def test_qualification_fresh_process_without_xtoken(tmp_path):
    import os, subprocess, sys
    import torch
    os.environ['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT']='1'
    from kdflow.algorithms._mp_opd_energy import MPAtomEnergy,save_energy_checkpoint
    energy=tmp_path/'energy';energy.mkdir()
    (energy/'summary.json').write_text(json.dumps({'completed_groups':32}))
    row={'invalid':False,'energy_updates':{'4':{'parameter_delta':0.01}},'learned_partition':{'coverage_max_error':0.0002}}
    (energy/'results.jsonl').write_text('\n'.join(json.dumps(row) for _ in range(32)))
    net=MPAtomEnergy(10,32,2)
    save_energy_checkpoint(energy/'energy-select-4.pt',net,torch.optim.AdamW(net.parameters()),step=128,extra_config={'max_span_length':2})
    env=dict(os.environ);env.pop('KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT',None)
    script=Path(__file__).resolve().parents[2]/'experiments/runai/learned_partition_pilot.py'
    subprocess.run([sys.executable,str(script),'qualify','--work',str(tmp_path)],env=env,check=True)
    assert json.loads((tmp_path/'qualification.json').read_text())['technical_pass']


def test_recover_reuses_energy_and_preserves_deadline(tmp_path):
    from experiments.runai.learned_partition_pilot import recover
    prior=tmp_path/'prior';prior.mkdir();(prior/'energy').mkdir()
    started=time.time()-19*3600
    jobs=[{'id':'energy'}, {'id':'qualify','gpu':None,'require_success':['energy'],'argv':['bash','/old/host','/old/script','qualify','--work',str(prior)]}]
    for name,dep in [('soft-canary','qualify'),('soft-pilot','soft-canary')]:
        jobs.append(dict(id=name,gpu=1,require_success=[dep],budget_seconds=1200,argv=['bash','/old/train','1','soft','50'],env={'MP_ENERGY_CHECKPOINT':str(prior/'energy/energy-select-4.pt'),'MP_RUN_ROOT':str(prior/name)}))
    (prior/'plan.json').write_text(json.dumps(dict(source='/old',hours=20,jobs=jobs)))
    (prior/'campaign-state.json').write_text(json.dumps(dict(started=started,jobs={'energy':{'status':'completed'}})))
    out=tmp_path/'out';recover(prior,out)
    plan=json.loads((out/'plan.json').read_text())
    assert [j['id'] for j in plan['jobs']]==['qualify','soft-canary','soft-pilot']
    assert 'require_success' not in plan['jobs'][0]
    assert plan['jobs'][2]['require_success']==['soft-canary']
    assert 1900<plan['jobs'][2]['budget_seconds']<=2040
    assert (out/'energy').resolve()==prior/'energy'
    assert json.loads((out/'campaign-state.json').read_text())['started']==started
