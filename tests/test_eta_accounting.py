"""CPU-only regression tests for session/campaign ETA accounting."""
import ast
import time
from collections import defaultdict
from datetime import timedelta
from types import SimpleNamespace
from pathlib import Path


def _symbols():
    source = Path(__file__).resolve().parents[1] / "kdflow/trainer/on_policy_kd_trainer.py"
    tree = ast.parse(source.read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"progress_metrics", "format_eta"}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "OnPolicyKDTrainer")
    logging = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                   and node.name == "logging")
    ns = {"timedelta": timedelta, "time": time}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "<eta helpers>", "exec"), ns)
    exec(compile(ast.Module(body=[logging], type_ignores=[]), "<eta logging>", "exec"), ns)
    return ns


def test_fresh_session_uses_successful_optimizer_updates():
    m = _symbols()["progress_metrics"](current_updates=4, session_start_updates=0,
        session_elapsed_seconds=800, campaign_target_updates=312, diagnostic_target_updates=4)
    assert m["session_completed_updates"] == 4
    assert m["eta_campaign_seconds"] == 61600
    assert m["eta_diagnostic_seconds"] == 0


def test_resume_four_to_eight_uses_session_delta():
    m = _symbols()["progress_metrics"](current_updates=8, session_start_updates=4,
        session_elapsed_seconds=800, campaign_target_updates=312, diagnostic_target_updates=8)
    assert m["session_completed_updates"] == 4
    assert m["eta_campaign_seconds"] == 60800
    assert m["eta_diagnostic_seconds"] == 0


def test_zero_progress_and_completed_target_are_explicit():
    f = _symbols()["progress_metrics"]
    zero = f(current_updates=4, session_start_updates=4, session_elapsed_seconds=800,
             campaign_target_updates=312, diagnostic_target_updates=8)
    assert zero["session_completed_updates"] == 0
    assert zero["eta_campaign_seconds"] is None
    assert zero["eta_diagnostic_seconds"] is None
    done = f(current_updates=8, session_start_updates=4, session_elapsed_seconds=800,
             campaign_target_updates=8, diagnostic_target_updates=8)
    assert done["eta_campaign_seconds"] == 0
    assert done["eta_diagnostic_seconds"] == 0


def test_logging_reports_completed_updates_and_separate_targets():
    ns = _symbols()
    trainer = SimpleNamespace()
    trainer.args = SimpleNamespace(log=SimpleNamespace(logging_steps=1))
    class Recorder:
        def __init__(self): self.records=[]
        def log(self, payload, step=None): self.records.append((payload, step))
    trainer.strategy = Recorder(); trainer._wandb = Recorder(); trainer._tensorboard = Recorder()
    trainer.num_rollout_iters_per_epoch = 312; trainer.epochs = 1; trainer.current_epoch = 0
    trainer.global_step = 5; trainer.completed_optimizer_updates = 4
    trainer.session_start_optimizer_updates = 0; trainer.session_start_time = time.time() - 800
    trainer.campaign_target_updates = 312; trainer.diagnostic_stop_target = 8
    trainer.log_state = defaultdict(list); trainer.log_state["loss"].append(1.0)
    ns["logging"](trainer)
    text = trainer.strategy.records[-1][0]
    assert "optimizer_updates [4/312]" in text
    assert "diagnostic_progress [50.00%]" in text
    payload = trainer._wandb.records[-1][0]
    assert payload["train/session_completed_updates"] == 4
    assert abs(payload["train/eta_campaign_seconds"] - 61600) < 2


def test_failed_rollout_attempt_does_not_change_accounting():
    ns = _symbols()
    trainer = SimpleNamespace(args=SimpleNamespace(log=SimpleNamespace(logging_steps=1)),
        strategy=SimpleNamespace(records=[], log=lambda x: None), _wandb=None, _tensorboard=None,
        num_rollout_iters_per_epoch=312, epochs=1, current_epoch=0, global_step=5,
        completed_optimizer_updates=4, session_start_optimizer_updates=0,
        session_start_time=time.time()-800, campaign_target_updates=312,
        diagnostic_stop_target=8, log_state=defaultdict(list))
    ns["logging"](trainer)
    assert trainer.completed_optimizer_updates == 4
