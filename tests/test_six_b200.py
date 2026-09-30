"""Offline regressions for allocation, publication and job identity boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('six',ROOT/'experiments/runai/six_b200.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def test_six_gpu_specs_keep_owner_unlimited_and_borrowed_bounded():
    c={'borrowed_until':9999999}
    owner=M.specs(M.OWNER,{0:'GPU-owner0',1:'GPU-owner1'},c)
    borrowed=M.specs(M.BORROWED,{i:f'GPU-b{i}' for i in range(4)},c)
    assert all('deadline' not in s for s in owner)
    assert all(s['deadline']==9999999 for s in borrowed)
    trains=[s for s in owner+borrowed if s['argv'][2]=='train']
    assert {s['id'] for s in trains}=={'soft42','soft43','random43'}
    assert len([s for s in borrowed if s['argv'][2]=='gen'])==4
    assert all(s['dependency_policy']=='terminal' for s in owner+borrowed if s['argv'][2]=='gen')
    random=next(s for s in borrowed if s['id']=='random43')
    assert random['timeout_seconds']==12*3600 and 'require_full_budget' not in random
    assert all(s['gpus']==0 for s in borrowed if s['id'].startswith('score-'))


def test_checkpoint156_not_published_while_epoch_save_is_in_progress(tmp_path):
    run=tmp_path/'run';(run/'checkpoint/step156').mkdir(parents=True)
    log=Path(str(run)+'.log');log.write_text('completed_optimizer_updates: 156.000000')
    assert M.safe_steps(run)==[]
    log.write_text('completed_optimizer_updates: 157.000000')
    assert M.safe_steps(run)==[156]


def test_final_checkpoint_requires_successful_summary_and_exit(tmp_path):
    run=tmp_path/'run';(run/'checkpoint/step312').mkdir(parents=True)
    Path(str(run)+'.log').write_text('completed_optimizer_updates: 312.000000')
    (run/'checkpoint/run-summary.json').write_text(json.dumps({'status':'completed'}))
    assert M.safe_steps(run)==[]
    Path(str(run)+'.exitcode').write_text('1')
    assert M.safe_steps(run)==[]
    Path(str(run)+'.exitcode').write_text('0')
    assert M.safe_steps(run)==[312]


def test_finished_producers_do_not_release_dispatchers_before_final_publication(tmp_path):
    with patch.object(M,'WORK',tmp_path):
        for key in M.TRAIN:
            p=tmp_path/'train'/key;p.mkdir(parents=True)
            (p/'finished.json').write_text('{}')
        assert M.training_done()
        assert not M.producers_done()
        (tmp_path/'publisher-done.json').write_text('{}')
        assert M.producers_done()


def test_train_env_retains_frozen_energy_seed_and_scheduler_start(tmp_path):
    pilot=tmp_path/'pilot.json';pilot.write_text(json.dumps({'options':dict(
        student_name_or_path='SFT',teacher_name_or_path='teacher',train_dataset_path='data',
        mp_opd_random_seed=43,mp_opd_max_span_length=2,mp_opd_fixed_span_length=2)}))
    cfg=dict(pilot=str(pilot),energy='qualified-energy',borrowed_until=100000)
    soft=M.train_env('soft43',1,cfg);random=M.train_env('random43',0,cfg)
    assert soft['MP_SEED']=='43' and soft['MP_ENERGY_CHECKPOINT']=='qualified-energy'
    assert soft['MP_STUDENT_PATH']=='SFT' and 'MP_TRAIN_STOP_AT' not in soft
    assert random['MP_TRAIN_STOP_AT']=='99100' and 'MP_ENERGY_CHECKPOINT' not in random


def test_existing_clock_never_extends_on_reinit(tmp_path):
    c=dict(source=str(M.ROOT),borrowed_until=1000)
    (tmp_path/'campaign.json').write_text(json.dumps(c))
    before=(tmp_path/'campaign.json').read_bytes()
    with patch.object(M,'WORK',tmp_path),patch.object(M.socket,'gethostname',return_value=M.BORROWED):
        M.init(12)
    assert (tmp_path/'campaign.json').read_bytes()==before


def test_dispatch_deadline_uses_worker_and_allocation_minimum():
    with patch.object(M,'config',return_value={'borrowed_until':2000}), \
         patch.object(M.socket,'gethostname',return_value=M.BORROWED), \
         patch.dict(M.os.environ,{'JM_DEADLINE':'3000'}):
        assert M.now_deadline()==1955


def test_publication_is_idempotent_and_records_training_seed(tmp_path):
    E,D,Q=M.B.eval_modules()
    cfg=dict(qualification=[])
    (tmp_path/'template.json').write_text(json.dumps(dict(seeds=[42,43,44],data={})))
    identity=dict(path='checkpoint',sha256='hash')
    with patch.object(M,'WORK',tmp_path),patch.object(M,'config',return_value=cfg), \
         patch.object(M,'sources',return_value=[('soft43',tmp_path/'run','soft',43)]), \
         patch.object(M,'safe_steps',return_value=[40]),patch.object(M.B,'checkpoint_stable',return_value=identity):
        M.publish_available()
        path=tmp_path/'pool/soft43-step40/plan.json';before=path.read_bytes()
        M.publish_available()
        assert path.read_bytes()==before
        ready=E.read_json(path.parent/'ready.json')
        assert ready['training_seed']==43 and ready['plan_sha256']==E.file_hash(path)


def test_late_random_admission_releases_gpu_without_launch(tmp_path):
    with patch.object(M,'WORK',tmp_path),patch.object(M,'config',return_value={'borrowed_until':100}), \
         patch.object(M.time,'time',return_value=99),patch.object(M.subprocess,'run') as launch:
        M.train('random43',0)
        launch.assert_not_called()
    assert json.loads((tmp_path/'train/random43/finished.json').read_text())['status']=='skipped'


def test_generation_receives_borrowed_cutoff_and_requires_uuid_lease(tmp_path):
    E,D,Q=M.B.eval_modules()
    data=tmp_path/'data.json';data.write_text('{}')
    plan=dict(source=D.script_hashes(),seeds=[42],jobs=[{'id':'job'}],
              data={'gsm8k':dict(path=str(data),sha256=E.file_hash(data))})
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    ready=tmp_path/'ready.json';ready.write_text(json.dumps(dict(plan_sha256=E.file_hash(path))))
    received=[]
    def gen(root,plan,ph,job,args,end):
        received.append((args.gpu,args.concurrency,end))
        cell=tmp_path/'cells/job/gsm8k/42';cell.mkdir(parents=True)
        (cell/'generation-complete.json').write_text('{}')
    with patch.object(M,'config',return_value={'borrowed_until':1000}), \
         patch.object(M.socket,'gethostname',return_value=M.BORROWED), \
         patch.object(M.time,'time',return_value=0),patch.object(M.time,'sleep'), \
         patch.dict(M.os.environ,{'JM_DEADLINE':'2000','CUDA_VISIBLE_DEVICES':'GPU-two'}), \
         patch.object(M.subprocess,'check_output',return_value='GPU-two\n'), \
         patch.object(M,'plans',return_value=[ready]),patch.object(M,'producers_done',return_value=True), \
         patch.object(Q,'GENERATION_ONLY',False),patch.object(Q,'run_checkpoint',side_effect=gen):
        M.generate(2)
        assert received==[(2,256,955)]
        with patch.dict(M.os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-other'}):
            with pytest.raises(ValueError,match='lease mismatch'):M.generate(2)
