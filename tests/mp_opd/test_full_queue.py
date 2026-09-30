import importlib.util
import os
from pathlib import Path
import pytest

path=Path(__file__).parents[2]/'experiments/runai/queue_full_alternating.py'
spec=importlib.util.spec_from_file_location('full_queue',path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def test_campaign_runs_exact_order_and_budget():
    configs=q.configurations()
    assert all(x['micro_B']==1 and x['micro_M']==4 for x in configs)
    assert q.TRAIN_SEEDS==(43,) and q.OWNER_SEEDS==(43,) and q.EXTRA_SEEDS==()
    assert [x['id'] for x in configs]==[f'ALT-{name}-s{seed}' for seed in q.TRAIN_SEEDS
        for name,_,_ in q.VARIANTS]
    assert [x['train_seed'] for x in configs]==[seed for seed in q.TRAIN_SEEDS
        for _ in q.VARIANTS]
    assert len(configs)==len(q.TRAIN_SEEDS)*len(q.VARIANTS)
    assert all(x['student_updates']==312 and x['B']==64 and x['M']==16 and x['student']=='full' for x in configs)
    assert [(x['energy_lr'],x['energy_every']) for x in configs[:3]]==[(.001,4),(.0001,1),(.001,1)]
    assert [x['id'] for x in configs[:3]]==['ALT-every4-s43','ALT-lowLR-s43','ALT-main-s43']
    assert [x['id'] for x in configs]==['ALT-every4-s43','ALT-lowLR-s43','ALT-main-s43']


def test_run_command_forwards_pinned_microbatch_not_shell(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('MP_MICRO_TRAIN_BATCH_SIZE','32')
    monkeypatch.setenv('MP_META_MICRO_BATCH_SIZE','16')
    monkeypatch.setattr(q,'checked_config',lambda _:dict(student='s',teacher='t',
        dataset='d',energy='e',meta='m',commit='test',offload_adam_moments=True))
    captured={}
    def run(argv,**kwargs):
        captured.update(kwargs['env'])
        assert argv[-2]=='312'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',run)
    monkeypatch.setattr(q,'write',lambda *args:None)
    q.run_command(tmp_path,q.configurations()[0],tmp_path/'run')
    assert captured['MP_MICRO_TRAIN_BATCH_SIZE']=='1'
    assert captured['MP_META_MICRO_BATCH_SIZE']=='4'
    assert captured['MP_OFFLOAD_ADAM_MOMENTS']=='1'
    paths=captured['PYTHONPATH'].split(os.pathsep)
    assert paths[0]==str(q.ROOT/'experiments/modal/vendor')
    assert paths[1]==str(q.ROOT)


def test_specs_forwards_case_offload_flag(tmp_path):
    (tmp_path / 'campaign.json').write_text('{"offload_adam_moments": true}')
    jobs = q.specs(tmp_path, 'GPU-test')
    assert jobs and all(j['env']['MP_OFFLOAD_ADAM_MOMENTS'] == '1' for j in jobs)


def test_dag_qualification_gate_and_eval_checkpoints(tmp_path):
    jobs=q.specs(tmp_path,'GPU-test')
    assert jobs==q.specs(tmp_path,'GPU-test')
    assert q.eval_cells()==24 and q.expected_trains(True)==3 and q.expected_trains(False)==0
    assert len(jobs)==54 and len({j['id'] for j in jobs})==54
    seen=set()
    for job in jobs:
        assert set(job['dependencies'])<=seen
        assert job['dependency_policy']=='success'
        seen.add(job['id'])
    assert jobs[0]['gpus']==0 and jobs[1]['dependencies']==[jobs[0]['id']]
    assert jobs[2]['dependencies']==[jobs[1]['id']]
    trains=[j for j in jobs if 'train' in j['argv']]
    gens=[j for j in jobs if 'generate' in j['argv']]
    scores=[j for j in jobs if 'score' in j['argv']]
    assert len(trains)==len(q.configurations())
    assert len(gens)==len(scores)==q.eval_cells()
    assert gens[0]['dependencies']==[trains[-1]['id']]
    assert set(jobs[-1]['dependencies'])=={j['id'] for j in scores}
    assert all(j['gpus']==['GPU-test'] for j in trains+gens)
    assert all(j['gpus']==0 for j in scores)
    steps=[int(j['argv'][j['argv'].index('--step')+1]) for j in gens[:8]]
    assert steps==[40,80,120,156,200,240,280,312]


def test_real_manager_accepts_and_orders_specs(tmp_path):
    import sys
    manager=Path('/mnt/d/dev/codex/job-manager')
    if not manager.exists():
        import pytest;pytest.skip('Existing local manager checkout unavailable')
    sys.path.insert(0,str(manager))
    from job_manager.store import initialize,connect,submit,rows
    state=tmp_path/'state'
    initialize(state,dict(gpus=['GPU-test'],cpu_slots=4,deadline=None))
    db=connect(state)
    try:
        jobs=q.specs(tmp_path/'case','GPU-test')
        submit(db,jobs)
        assert [r['id'] for r in rows(db)]==[j['id'] for j in jobs]
    finally:db.close()


@pytest.mark.parametrize('overlap',[False,True])
def test_audit_reads_company_queue_schema_and_blocks_overlap(tmp_path,monkeypatch,overlap):
    import sys,json,hashlib,types
    import borrowed_campaign as B
    item={'id':'test1','prompt':'train question' if overlap else 'unrelated evaluation'}
    data=tmp_path/'eval.json'
    data.write_text(json.dumps({'profile':'company-internal-v1','benchmark':'math500','items':[item]}))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (tmp_path/'eval-template.json').write_text(json.dumps({'data':{'math500':{'path':str(data),'sha256':digest(data),'count':1}}}))
    rows=[{'messages':[{'role':'user','content':'train question'}],'label':'teacher reference'}]
    pq=types.ModuleType('pyarrow.parquet')
    pq.read_table=lambda p:types.SimpleNamespace(to_pylist=lambda:rows*10000 if p=='train' else rows)
    pa=types.ModuleType('pyarrow');pa.parquet=pq
    monkeypatch.setitem(sys.modules,'pyarrow',pa);monkeypatch.setitem(sys.modules,'pyarrow.parquet',pq)
    E=types.SimpleNamespace(read_json=lambda p:json.loads(Path(p).read_text()),file_hash=digest)
    monkeypatch.setattr(B,'eval_modules',lambda:(E,types.SimpleNamespace(PROFILE='company-internal-v1'),None))
    monkeypatch.setattr(q,'checked_config',lambda case:{'dataset':'train','meta':'meta'})
    if overlap:
        with pytest.raises(ValueError,match='audit failed'):q.audit(tmp_path)
    else:q.audit(tmp_path)
    assert json.loads((tmp_path/'data-audit.json').read_text())['pass']==(not overlap)

def test_run_command_forwards_diagnostic_max_len_only_for_short_runs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(q,'checked_config',lambda _:dict(student='s',teacher='t',
        dataset='d',energy='e',meta='m',commit='test',offload_adam_moments=True))
    monkeypatch.setattr(q,'write',lambda *args:None)
    seen={}
    def run(argv,**kwargs):
        seen['env']=dict(kwargs['env'])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',run)
    monkeypatch.setenv('DIAG_MAX_LEN','2048')
    q.run_command(tmp_path,q.configurations()[0],tmp_path/'diag',limit=2)
    assert seen['env']['MP_MAX_LEN']=='2048'
    q.run_command(tmp_path,q.configurations()[0],tmp_path/'full',limit=312)
    assert 'MP_MAX_LEN' not in seen['env']
    monkeypatch.delenv('DIAG_MAX_LEN')
    q.run_command(tmp_path,q.configurations()[0],tmp_path/'diag-plain',limit=2)
    assert 'MP_MAX_LEN' not in seen['env']
