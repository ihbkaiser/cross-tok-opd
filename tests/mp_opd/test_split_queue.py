import importlib.util
import json
from pathlib import Path
import sys
import pytest

path=Path(__file__).parents[2]/'experiments/runai/queue_split_alternating.py'
spec=importlib.util.spec_from_file_location('split_queue',path)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


@pytest.mark.parametrize('host,gpus',[(q.OWNER,['GPU-owner']),(q.EXTRA,q.EXTRA_GPUS[:4])])
def test_skip_has_no_qualification_job(tmp_path,host,gpus):
    if host==q.EXTRA and not q.F.EXTRA_SEEDS:pytest.skip('this campaign has no extras seeds')
    jobs=q.specs(tmp_path,host,gpus,'skip')
    assert not any(j['argv'][4]=='qualify' for j in jobs)
    audit=next(j for j in jobs if j['argv'][4]=='audit')
    trains=[j for j in jobs if j['argv'][4]=='train']
    assert len(trains)==q.F.expected_trains(host==q.OWNER)
    assert trains[0]['dependencies']==[audit['id']]
    assert all(j['dependency_policy']=='terminal' for j in trains)
    if host==q.EXTRA:
        assert [j['dependencies'] for j in trains]==[[audit['id']] if i%2==0
            else [trains[i-1]['id']] for i in range(len(trains))]
    else:assert trains[1]['dependencies']==[trains[0]['id']]


@pytest.mark.parametrize('status',['completed','failed','timeout','cancelled','blocked','lost'])
def test_advisory_schedule_continues_after_terminal_qualification(tmp_path,status):
    manager=Path('/mnt/d/dev/codex/job-manager')
    if not manager.exists():pytest.skip('Local manager unavailable')
    sys.path.insert(0,str(manager))
    from job_manager.scheduler import readiness
    if not q.F.EXTRA_SEEDS:pytest.skip('this campaign has no extras seeds')
    jobs=q.specs(tmp_path,q.EXTRA,q.EXTRA_GPUS[:4],'advisory',1800)
    qualify=next(j for j in jobs if j['argv'][4]=='qualify')
    assert qualify['timeout_seconds']==1800
    trains=[j for j in jobs if j['argv'][4]=='train']
    # One train per GPU starts on qualification (the first seed of each variant); the later
    # seeds of that GPU wait for their own predecessor instead.
    first=[j for j in trains if j['dependencies']==[qualify['id']]]
    chained=[j for j in trains if j['dependencies']!=[qualify['id']]]
    assert len(first)==len(q.F.VARIANTS)
    assert len(chained)==len(trains)-len(q.F.VARIANTS)
    assert all(readiness({'spec':j},{qualify['id']:{'status':status}},0,{})[0]=='ready' for j in first)
    assert all(readiness({'spec':j},{qualify['id']:{'status':'running'}},0,{})[0]=='waiting' for j in first)
    assert not any(q.EXTRA_GPUS[4] in j['gpus'] for j in jobs if isinstance(j['gpus'],list))


@pytest.mark.parametrize('policy,passed,allowed', [('required',False,False),('advisory',False,True),('advisory',True,True),('skip',False,True),('skip',True,True)])
def test_train_advisory_preserves_unqualified_evidence(tmp_path,monkeypatch,policy,passed,allowed):
    config={'id':'test','energy_every':1}
    monkeypatch.setattr(q.F,'checked_config',lambda case:dict(commit='current',qualification_policy=policy,runs=[config]))
    qual=tmp_path/'qualification'
    if passed:q.F.write(qual/'PASS.json',dict(commit='current',status='EXACT_MATCH'))
    calls=[]
    def run(case,config,dest,resume):
        calls.append(resume)
        q.F.write(dest/'checkpoint/run-summary.json',dict(status='completed',optimizer_updates=312,energy_updates=312))
    monkeypatch.setattr(q.F,'run_command',run)
    if not allowed:
        with pytest.raises(ValueError,match='Qualification not passed'):q.F.train(tmp_path,'test',qual)
        assert not calls
    else:
        q.F.train(tmp_path,'test',qual)
        assert calls==[False]
        record=q.F.read(tmp_path/'train/test.qualification.json')
        assert record['status']==('skipped' if policy=='skip' else 'passed' if passed else 'unverified')
        assert (qual/'PASS.json').exists()==passed


@pytest.mark.parametrize('host,gpus',[(q.OWNER,['GPU-owner']),(q.EXTRA,q.EXTRA_GPUS)])
def test_plan_manager_contract_and_dependencies(tmp_path,host,gpus):
    if host==q.EXTRA and not q.F.EXTRA_SEEDS:pytest.skip('this campaign has no extras seeds')
    jobs=q.specs(tmp_path,host,gpus)
    assert jobs==q.specs(tmp_path,host,gpus)
    seen=set()
    for j in jobs:
        assert set(j['dependencies'])<=seen
        seen.add(j['id'])
    train=[j for j in jobs if j['argv'][4]=='train']
    qualify=next(j for j in jobs if j['argv'][4]=='qualify')
    assert len(train)==q.F.expected_trains(host==q.OWNER)
    assert train[0]['dependencies']==[qualify['id']]
    if host==q.OWNER:
        assert [j['dependencies'] for j in train[1:]]==[[t['id']] for t in train[:-1]]
        assert all(j['gpus']==gpus for j in train)
    else:
        # One variant per GPU, its seeds in sequence: every other train starts on qualify.
        assert [j['gpus'] for j in train]==[[g] for g in gpus[:3] for _ in q.F.EXTRA_SEEDS]
        assert [j['dependencies'] for j in train]==[[qualify['id']] if i%2==0
            else [train[i-1]['id']] for i in range(len(train))]
        assert len({j['env']['KDFLOW_ROLLOUT_PORT_BASE'] for j in train})==3
        gen=[j for j in jobs if j['argv'][4]=='generate']
        assert len(gen)==5
        assert [j['gpus'] for j in gen[-2:]]==[[g] for g in gpus[3:]]
    manager=Path('/mnt/d/dev/codex/job-manager')
    if manager.exists():
        sys.path.insert(0,str(manager))
        from job_manager.store import initialize,connect,submit,rows
        state=tmp_path/'state';initialize(state,dict(gpus=gpus,cpu_slots=16,deadline=None))
        db=connect(state)
        try:
            submit(db,jobs)
            assert len(rows(db))==len(jobs)
        finally:db.close()


def test_export_not_visible_before_commit_marker_and_detects_change(tmp_path):
    from kdflow.export_ready import marker,publish,validate
    folder=tmp_path/'step40';folder.mkdir();(folder/'config.json').write_text('{}')
    (folder/'model.safetensors').write_bytes(b'model')
    assert not marker(tmp_path,40).exists()
    publish(tmp_path,40)
    assert validate(tmp_path,40)==folder
    (folder/'model.safetensors').write_bytes(b'truncated')
    with pytest.raises(ValueError,match='mismatch'):validate(tmp_path,40)


@pytest.mark.parametrize('fail',[False,True])
def test_worker_ignores_uncommitted_and_preserves_failure(tmp_path,monkeypatch,fail):
    from kdflow.export_ready import publish
    monkeypatch.setattr(q.F,'configurations',lambda:[{'id':'test'}])
    monkeypatch.setattr(q.F,'STEPS',(40,80))
    root=tmp_path/'train/test/checkpoint';(root/'step40').mkdir(parents=True)
    (root/'step40/config.json').write_text('{}');publish(root,40)
    (root/'step80').mkdir() # Producer is still writing. Never consume this.
    calls=[]
    def evaluate(case,run,step,phase):
        calls.append((run,step,phase))
        if fail:raise RuntimeError('retained failure')
    monkeypatch.setattr(q.F,'evaluate',evaluate)
    monkeypatch.setattr(q,'producers_done',lambda case:True)
    monkeypatch.setattr(q.time,'sleep',lambda seconds:None)
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError,match='retained'):q.worker(tmp_path,'generate')
        else:q.worker(tmp_path,'generate')
    assert calls==[('test',40,'generate')]
    assert (tmp_path/'dispatch/test-step40'/('generate.error.json' if fail else 'generate.done.json')).exists()


def test_producers_require_both_host_status_and_all_trains_terminal(tmp_path):
    if not q.F.EXTRA_SEEDS:pytest.skip('this campaign has no extras seeds')
    for host,gpus in [(q.OWNER,['GPU-owner']),(q.EXTRA,q.EXTRA_GPUS)]:
        jobs=q.specs(tmp_path,host,gpus);dest=tmp_path/'nodes'/host;dest.mkdir(parents=True)
        (dest/'receipt.json').write_text(json.dumps({'jobs':jobs}))
        (dest/'status.json').write_text(json.dumps({'jobs':{j['id']:'completed' for j in jobs}}))
        if host==q.OWNER:assert not q.producers_done(tmp_path)
    assert q.producers_done(tmp_path)
    state=json.loads((dest/'status.json').read_text())
    job=next(j for j in jobs if j['argv'][4]=='train');state['jobs'][job['id']]='running'
    (dest/'status.json').write_text(json.dumps(state))
    assert not q.producers_done(tmp_path)


@pytest.mark.parametrize('started',[False,True])
@pytest.mark.parametrize('disconnected',[False,True])
def test_retirement_refuses_started_train_and_preserves_other_jobs(tmp_path,monkeypatch,started,disconnected):
    manager=Path('/mnt/d/dev/codex/job-manager')
    if not manager.exists():pytest.skip('Local manager unavailable')
    sys.path.insert(0,str(manager))
    from job_manager.store import initialize,connect,submit,rows
    state=tmp_path/'state';initialize(state,dict(gpus=['GPU-owner'],cpu_slots=16,deadline=None))
    jobs=q.specs(tmp_path,q.OWNER,['GPU-owner'])
    extra=dict(jobs[0],id='unrelated',dependencies=[])
    db=connect(state);submit(db,jobs+[extra])
    qualify=next(j for j in jobs if j['argv'][4]=='qualify')
    train=next(j for j in jobs if j['argv'][4]=='train')
    db.execute("UPDATE jobs SET status='failed' WHERE id=?",(qualify['id'],))
    if started:db.execute('UPDATE jobs SET started=1 WHERE id=?',(train['id'],))
    node=tmp_path/'nodes'/q.OWNER;node.mkdir(parents=True)
    (node/'receipt.json').write_text(json.dumps(dict(host=q.OWNER,state=str(state),manager=str(manager),jobs=jobs)))
    (tmp_path/'campaign.json').write_text(json.dumps({'commit':'f5d5114e1d7a3c3a29a1901f167968acb5e8b747'}))
    monkeypatch.setattr(q.socket,'gethostname',lambda:q.OWNER)
    try:
        if started and not disconnected:
            with pytest.raises(ValueError,match='Training already started'):q.retire_unstarted(tmp_path)
        else:q.retire_unstarted(tmp_path,disconnected_meta=disconnected)
        current={j['id']:j for j in rows(db)}
        assert current['unrelated']['status']=='queued'
        assert current[train['id']]['status']==('queued' if started and not disconnected else 'cancelled')
        assert current[qualify['id']]['status']=='failed'
        if disconnected:
            assert json.loads((node/'retired-disconnected-meta.json').read_text())['invalid_for_alternating_efficacy']
    finally:db.close()
