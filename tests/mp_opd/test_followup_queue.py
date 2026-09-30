import copy
import json
import pytest
from experiments.runai.queue_alternating_followup import validate_pair,reserved_rows,frozen_command,report

def pair():
    m={'data_sha256':'data','args':{'output':'a','freeze_energy':False,'virtual_lr':.001},
       'models':{},'source_commit':'commit','source_diff_sha256':'clean','resume_runtime':{},'resume_source_files':{}}
    a={'schema':'mp-alternating-resume-v2','step':50,'energy_updates':50,'cursor':50,
       'manifest':m,'results':[{'index':i} for i in range(50)],'traces':[{} for i in range(50)]}
    b=copy.deepcopy(a);b['energy_updates']=0;b['manifest']['args'].update(output='b',freeze_energy=True)
    return a,b

def test_pair_rejects_different_optimizer_or_source():
    a,b=pair();assert validate_pair(a,b,'data')==50
    b['manifest']['args']['virtual_lr']=.1
    with pytest.raises(ValueError,match='arguments'):validate_pair(a,b,'data')
    a,b=pair();b['manifest']['source_commit']='different'
    with pytest.raises(ValueError,match='provenance'):validate_pair(a,b,'data')
    with pytest.raises(ValueError,match='Dataset'):validate_pair(a,b,'wrong')

def test_reserve_excludes_even_consumed_invalid_groups():
    a,b=pair();b['cursor']=52;b['results']+=[{'index':50},{'index':51}];b['traces'] += [{},{}]
    cursor=validate_pair(a,b,'data')
    data={'groups':[{'eval':[{'id':f'{i}-{j}'} for j in range(4)]} for i in range(60)]}
    rows=reserved_rows(data,cursor)
    assert len(rows)==32 and min(r['group'] for r in rows)==52
    with pytest.raises(ValueError,match='No untouched'):reserved_rows(data,60)

def test_frozen_command_preserves_all_other_arguments(tmp_path):
    cmd=['bash','wrapper','-u','runner','run','--output','original','--data','data','--max-student-updates','50']
    (tmp_path/'command.json').write_text(json.dumps(cmd))
    got=frozen_command(tmp_path)
    assert got[-1]=='--freeze-energy'
    got[got.index('--output')+1]='original'
    assert got[:-1]==cmd

def test_report_delta_sign_and_counts(tmp_path):
    (tmp_path/'run').mkdir();(tmp_path/'followup/frozen').mkdir(parents=True)
    for p in ('run/summary.json','followup/frozen/summary.json'):
        (tmp_path/p).write_text('{}')
    (tmp_path/'followup/paired-eval.json').write_text(json.dumps({'scope':'test','groups':1,'rows':[
        {'initial':2.,'alternating':1.,'frozen':1.5}, {'initial':2.,'alternating':1.,'frozen':1.5}]}))
    report(tmp_path)
    s=json.loads((tmp_path/'followup/report.json').read_text())
    assert s['alternating_minus_frozen']==-.5 and s['references']==2

def test_manager_discovery_only_matches_manager_run(tmp_path):
    from experiments.runai.queue_alternating_followup import active_states
    for i,argv in enumerate((['python','-m','job_manager','--state','/tmp/a','run'],
                             ['python','-m','job_manager','--state','/tmp/b','status'],
                             ['python','train.py'])):
        path=tmp_path/str(i);path.mkdir();(path/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
    assert active_states(tmp_path)==['/tmp/a']

def test_submit_real_store_contract_and_idempotency(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import experiments.runai.queue_alternating_followup as m
    manager=Path('/mnt/d/dev/codex/job-manager')
    if not manager.is_dir():pytest.skip('Local job-manager integration source unavailable')
    monkeypatch.syspath_prepend(str(manager))
    from job_manager.store import initialize,connect,rows
    import job_manager.__main__ as cli
    state=tmp_path/'state';initialize(state,{'gpus':['GPU-test'],'cpu_slots':4})
    case=tmp_path/'case';(case/'run').mkdir(parents=True)
    (case/'run/summary.json').write_text(json.dumps({'status':'completed','student_updates':50}))
    (case/'exit-code.txt').write_text('0')
    monkeypatch.setattr(m,'choose_state',lambda args:state)
    monkeypatch.setattr(m.subprocess,'check_output',lambda argv,**kw:'GPU-test' if argv[0]=='nvidia-smi' else '')
    monkeypatch.setattr(cli,'snapshot',lambda db:{'manager':{'running':True},'paused':False,'quarantined':False})
    args=SimpleNamespace(case=case,manager=manager,state=state,gpu_uuid='GPU-test')
    m.submit(args);m.submit(args)
    db=connect(state)
    try:
        jobs=rows(db)
        assert len(jobs)==3
        assert jobs[1]['spec']['dependencies']==[jobs[0]['id']]
        assert jobs[2]['spec']['gpus']==0
        assert 'CUDA_VISIBLE_DEVICES' not in jobs[0]['spec']['env']
    finally:db.close()
