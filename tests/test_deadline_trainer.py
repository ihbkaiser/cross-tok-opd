"""Exercise real fit method with fake Ray/model boundaries; no model deps needed."""
import ast
from collections import defaultdict
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import time
from kdflow.deadline import stop_before_rollout


def test_fit_saves_existing_progress_before_next_rollout(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'kdflow/trainer/on_policy_kd_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OnPolicyKDTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in {'progress_metrics', 'format_eta'}]
    namespace = dict(time=time, os=os, json=json, timedelta=timedelta,
                     stop_before_rollout=stop_before_rollout, torch=NS(manual_seed=lambda x:None,Generator=lambda: NS(manual_seed=lambda x:None)), ray=NS(get=lambda x:x))
    exec(compile(ast.Module(body=helpers + [method], type_ignores=[]), '<real trainer fit>', 'exec'), namespace)
    saved=[]
    def save_model(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}")
        saved.append(path)
    class Loader(list):
        sampler=NS(set_epoch=lambda *args,**kwargs:None)
    args=NS(rollout=NS(diagnostic_max_updates=50, rollout_tp_size=1),
            model=NS(student_name_or_path='s',teacher_name_or_path='t'),
            train=NS(train_batch_size=64,micro_train_batch_size=4,enable_sleep=False,save_path=str(tmp_path),seed=0),
            kd=NS(kd_algorithm='mp_opd'))
    trainer=NS(args=args,max_rollout_iters=312,epochs=2,completed_optimizer_updates=7,
               _print_training_config=lambda:None,_start_resource_logger=lambda:None,_save_training_checkpoint=lambda *args:None,
               student=NS(connect_rollout_engines=lambda *x:None,async_save_model=save_model),
               rollout_group=NS(actors=[]),_resume_directory=None,train_dataloader=Loader([['unused']]),
               strategy=NS(log=lambda x:None),_wandb=None,_tensorboard=None)
    monkeypatch.setenv('MP_TRAIN_STOP_AT',str(time.time()+100))
    namespace['fit'](trainer,global_step=7)
    assert saved==[str(tmp_path/'step7')]
    summary=json.loads((tmp_path/'run-summary.json').read_text())
    assert summary['status']=='stopped'
    assert summary['stop_reason']=='deadline_checkpoint_reserve'
    assert summary['optimizer_updates']==7
    assert summary['session_start_optimizer_updates'] == 7
    assert summary['session_completed_updates'] == 0
    assert summary['scientific_total_updates'] == 312
    assert summary['diagnostic_stop_target'] == 50
    assert summary['campaign_progress'] == 7 / 312
    assert summary['diagnostic_progress'] == 7 / 50
    assert summary['eta_campaign_seconds'] is None
    assert summary['eta_diagnostic_seconds'] is None
