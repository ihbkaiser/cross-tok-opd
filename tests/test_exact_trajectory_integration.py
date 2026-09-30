from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kdflow.trainer.on_policy_kd_trainer import OnPolicyKDTrainer
from kdflow.backend.sglang._engine_requests import _handle_generate


class Tokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        # Full-string retokenization deliberately merges the boundary.
        return {"input_ids": {"prefix": [1, 8], "ab": [31], "prefixab": [1, 99]}[text]}

    def decode(self, ids, **kwargs):
        assert ids == [20, 21]
        return "ab"


def trainer():
    obj = OnPolicyKDTrainer.__new__(OnPolicyKDTrainer)
    obj.student_processor = obj.teacher_processor = Tokenizer()
    return obj


def test_sample_preserves_noncanonical_generated_ids_and_prompt_boundary():
    output = {"prompt_ids": [1, 8], "output_ids": [20, 21, 2], "text": "ab",
              "meta_info": {"output_token_logprobs": [[-1, 20], [-2, 21], [-3, 2]]}}
    batch = trainer()._build_exact_rollout_sample("prefix", "prefix", output, "", None)
    assert batch["stu_input_ids"].tolist() == [1, 8, 20, 21, 2]
    assert batch["tea_input_ids"].tolist() == [1, 8, 31, 2]
    selected = batch["stu_input_ids"].roll(-1)[batch["stu_loss_mask"]]
    assert selected.tolist() == [20, 21, 2]
    assert batch["stu_behavior_log_probs"][batch["stu_loss_mask"]].tolist() == [-1, -2, -3]


def test_behavior_label_mismatch_is_rejected():
    output = {"prompt_ids": [1, 8], "output_ids": [20, 21],
              "meta_info": {"output_token_logprobs": [[-1, 31]]}}
    with pytest.raises(RuntimeError, match="log-prob IDs"):
        trainer()._build_exact_rollout_sample("prefix", "prefix", output, "", None)


def test_teacher_engine_receives_exact_ids_and_selects_same_predictors():
    seen = {}
    values = np.arange(15).reshape(5, 3)
    def generate(**kwargs):
        seen.update(kwargs)
        return [{"meta_info": {"hidden_states": [values]}}]
    sent = []
    engine = SimpleNamespace(generate=generate)
    queue = SimpleNamespace(put=lambda x: None)
    socket = SimpleNamespace(send=lambda x, **kwargs: sent.append(x))
    mask = np.array([False, True, True, True, False])
    request = {"kwargs": {"prompt": ["unused text"], "input_ids": [[1, 8, 20, 21, 2]],
                           "sampling_params": {"max_new_tokens": 0}, "loss_masks": [mask]}}
    _handle_generate(engine, request, socket, None, queue)
    assert seen["input_ids"] == [[1, 8, 20, 21, 2]]
    assert "prompt" not in seen
    np.testing.assert_array_equal(sent[1], values[mask])


def test_teacher_hidden_length_mismatch_is_rejected():
    engine = SimpleNamespace(generate=lambda **kwargs: [{"meta_info": {"hidden_states": [np.zeros((2, 3))]}}])
    request = {"kwargs": {"input_ids": [[1, 8, 2]], "sampling_params": {},
                           "loss_masks": [np.array([False, True, False])]}}
    acknowledgements = []
    with pytest.raises(RuntimeError, match="hidden-state length"):
        _handle_generate(engine, request, None, None, SimpleNamespace(put=acknowledgements.append))
    assert not acknowledgements


def test_nonfinite_behavior_is_rejected():
    output = {"prompt_ids": [1, 8], "output_ids": [20, 21],
              "meta_info": {"output_token_logprobs": [[float("nan"), 20], [-1, 21]]}}
    with pytest.raises(RuntimeError, match="non-finite sampled"):
        trainer()._build_exact_rollout_sample("prefix", "prefix", output, "", None)


def test_collapse_stops_before_update_and_saves_completed_policy(tmp_path, monkeypatch):
    import json
    from collections import defaultdict
    import kdflow.trainer.on_policy_kd_trainer as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    saved = []
    batches = []
    obj = OnPolicyKDTrainer.__new__(OnPolicyKDTrainer)
    obj.args = SimpleNamespace(
        rollout=SimpleNamespace(diagnostic_max_updates=30, rollout_tp_size=1),
        train=SimpleNamespace(train_batch_size=64, micro_train_batch_size=1, enable_sleep=True,
                              save_path=str(tmp_path),seed=42),
        model=SimpleNamespace(student_name_or_path="student", teacher_name_or_path="teacher"),
        kd=SimpleNamespace(kd_algorithm="mp_opd"),
    )
    obj.student = SimpleNamespace(connect_rollout_engines=lambda *args: None,
        wakeup=lambda: None, sleep=lambda: None, async_save_model=lambda path: saved.append(path))
    obj.teacher = SimpleNamespace(forward=lambda *args: pytest.fail("teacher must not run after collapse"))
    obj.rollout_group = SimpleNamespace(actors=[])
    class Loader(list):
        sampler = SimpleNamespace(set_epoch=lambda epoch,consumed_samples=0: None)
    obj.train_dataloader = Loader([[{"stu_prompt": "prefix"}]])
    obj.epochs = 2
    obj.completed_optimizer_updates = 3
    obj.max_rollout_iters = 100
    obj.generate_kwargs = {}
    obj._resume_directory = None
    obj.meta_sampler = None
    obj.log_state = defaultdict(list)
    obj._wandb = obj._tensorboard = None
    obj.strategy = SimpleNamespace(log=lambda message: None)
    obj._print_training_config = obj._start_resource_logger = obj.logging = lambda: None
    def rollout(batch, **kwargs):
        batches.append(batch)
        obj._collapse_stop = True
        obj.stop_reason = "zero_valid_samples_before_update"
        return []
    obj.rollout = rollout
    obj.fit(global_step=3)
    result = json.loads((tmp_path / "run-summary.json").read_text())
    assert result["optimizer_updates"] == 3
    assert result["rollout_iterations"] == 4
    assert result["status"] == "stopped"
    assert saved == [str(tmp_path / "step3")]
    assert len(batches) == 1


def test_rendered_chat_template_does_not_add_second_bos():
    obj=trainer();obj.args=SimpleNamespace(data=SimpleNamespace(apply_chat_template=True))
    def tokenizer(text,add_special_tokens):
        assert text=='<bos>rendered'
        return {'input_ids':([1] if add_special_tokens else [])+[1,8]}
    assert obj._encode_prompt_ids(tokenizer,'<bos>rendered')==[1,8]
    obj.args.data.apply_chat_template=False
    assert obj._encode_prompt_ids(tokenizer,'<bos>rendered')==[1,1,8]
