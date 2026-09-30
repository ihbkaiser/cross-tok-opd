from collections import defaultdict
from types import SimpleNamespace
import time

from kdflow.trainer.on_policy_kd_trainer import OnPolicyKDTrainer


class RecordingLogger:
    def __init__(self):
        self.records = []

    def log(self, payload, step=None):
        self.records.append((payload, step))


def test_logging_drops_inactive_dynamic_metrics_between_steps():
    trainer = OnPolicyKDTrainer.__new__(OnPolicyKDTrainer)
    trainer.args = SimpleNamespace(log=SimpleNamespace(logging_steps=1))
    trainer.strategy = RecordingLogger()
    trainer._wandb = RecordingLogger()
    trainer._tensorboard = RecordingLogger()
    trainer.num_rollout_iters_per_epoch = 2
    trainer.epochs = 1
    trainer.current_epoch = 0
    trainer.start_time = time.time() - 1
    trainer.log_state = defaultdict(list)

    trainer.global_step = 1
    trainer.log_state["loss"].append(2.0)
    trainer.log_state["mp_opd_invalid_reason_normalization_mismatch"].append(1.0)
    trainer.logging()

    first_wandb = trainer._wandb.records[-1][0]
    assert first_wandb["train/loss"] == 2.0
    assert first_wandb["train/mp_opd_invalid_reason_normalization_mismatch"] == 1.0
    assert not trainer.log_state

    trainer.global_step = 2
    trainer.log_state["loss"].append(4.0)
    trainer.log_state["mp_opd_invalid_reason_normalization_mismatch"]
    trainer.logging()

    second_wandb = trainer._wandb.records[-1][0]
    assert second_wandb["train/loss"] == 4.0
    assert "train/mp_opd_invalid_reason_normalization_mismatch" not in second_wandb
    assert "mp_opd_invalid_reason_normalization_mismatch" not in trainer.strategy.records[-1][0]
    assert trainer._tensorboard.records[-1] == (second_wandb, 2)
    assert not trainer.log_state
