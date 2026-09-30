import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/runai'))
spec=importlib.util.spec_from_file_location('seed44',ROOT/'experiments/runai/queue_seed44.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def fixture():
    base=dict(version=1,project='test',cwd=str(ROOT),argv=['python'],env={'MP_TRAIN_STOP_AT':'old'},
              gpus=['GPU0'],cpu_slots=8,timeout_seconds=100,deadline=9000,dependencies=[])
    rows=[dict(id=k,spec=base) for k in ('soft42','random43')]
    rows += [dict(id=f'gen-minfree-{i}',spec=dict(base,gpus=[f'GPU{i}'])) for i in range(4)]
    return rows


def test_borrowed_handoff_dag_uses_soft_allocation_and_distinct_gpus():
    jobs=M.build_specs(NS(OWNER='owner'),'borrowed',fixture());byid={j['id']:j for j in jobs}
    assert all('deadline' not in j and j['timeout_seconds']==72*3600 for j in jobs)
    for slot,key in enumerate(M.BORROW_KEYS,1):
        j=byid[key]
        assert j['gpus']==[f'GPU{slot}'] and j['dependencies']==['seed44-handoff']
        assert j['env']=={} and j['dependency_policy']=='success'
        gen=byid[f'gen-after44-{slot}']
        assert gen['dependencies']==[key] and gen['dependency_policy']=='terminal'
    assert byid['gen-after44-0']['dependencies']==['wait-random43']
    assert byid['gen-after44-0']['gpus']==['GPU0']
    assert sum(j['cpu_slots'] for j in jobs if j['id'] in (*M.BORROW_KEYS,'seed44-state-export','wait-random43') or j['id'].startswith('score-'))+8 <= 64
    ids={j['id'] for j in jobs}
    assert all(set(j['dependencies'])<=ids for j in jobs)


def test_owner_continues_two_training_chains_without_allocation_deadline():
    jobs=M.build_specs(NS(OWNER='owner'),'owner',fixture());byid={j['id']:j for j in jobs}
    assert all('deadline' not in j for j in jobs)
    assert byid['random44']['dependencies']==['wait-soft42']
    assert byid['soft44']['dependencies']==['wait-soft43']
    assert byid['gen-after44-0']['dependencies']==['random44']
    assert byid['gen-after44-1']['dependencies']==['soft44']


def test_simct_preserves_historical_algorithm_and_attention():
    fake=NS(config=lambda:{'borrowed_until':10000},train_env=lambda *a:dict(MP_SEED='44',MP_ALGORITHM='mp_opd'))
    env=M.environment(fake,'simct44',3)
    assert env['MP_ALGORITHM']=='span_ctkd' and env['MP_ATTN_IMPLEMENTATION']=='sdpa'
    assert 'MP_TRAIN_STOP_AT' not in env
    assert 'MP_TRAIN_STOP_AT' not in M.environment(fake,'soft44',1)


def test_handoff_requires_all_initial_checkpoint_cells(tmp_path):
    # Missing checkpoint/marker must not be mistaken for an empty complete pool.
    fake=NS(WORK=tmp_path,STEPS=(40,))
    assert not M.initial_gen_done(fake,None)


def test_old_publisher_done_cannot_end_extended_consumers(tmp_path):
    import json
    path=tmp_path/'publisher-done.json'
    assert not M.extended_done(path,M.EXTRA)
    path.write_text(json.dumps({'time':1}))
    assert not M.extended_done(path,M.EXTRA)
    path.write_text(json.dumps({'training_keys':list(M.EXTRA)}))
    assert M.extended_done(path,M.EXTRA)
