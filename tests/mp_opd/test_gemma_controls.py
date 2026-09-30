"""CPU regressions for Gemma content tokens and exact sampled event retention."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer, mp_content_ids
from kdflow.trajectory import trajectory_tokens


class Tok:
    eos_token_id = 1
    all_special_ids = [1, 107]
    def get_added_vocab(self):
        return {"<eos>": 1, "<end_of_turn>": 107, "\n": 10, "  ": 11}
    def decode(self, ids, **kwargs):
        return "".join({2: "x", 10: "\n", 11: "  ", 107: "<end_of_turn>", 1: "<eos>"}[i] for i in ids)
    def __call__(self, text, **kwargs):
        assert text == "x\n  "
        return {"input_ids": [2, 10, 11]}


def test_content_added_tokens_allowed_and_terminal_controls_excluded():
    sampled = [2, 10, 11, 107, 1]
    result = SimCTAtomizer(Tok(), Tok()).atomize(sampled, [2, 10, 11, 1], sample_id="s")
    assert result.valid
    assert result.covered_student_events == result.covered_teacher_events == 3
    assert result.masked_student_eos == 2
    assert sampled == [2, 10, 11, 107, 1]


def test_internal_control_still_rejected():
    result = SimCTAtomizer(Tok(), Tok()).atomize([2, 107, 2], [2, 107, 2], sample_id="s")
    assert not result.valid and result.failure_reason == "unsupported_added_token"


def test_terminal_only_is_empty():
    assert mp_content_ids([107, 1], Tok()) == ([], 2)
    result = SimCTAtomizer(Tok(), Tok()).atomize([107, 1], [1], sample_id="s")
    assert not result.valid and result.failure_reason == "empty_response"


def test_exact_builder_keeps_sampled_ids_and_behavior_with_terminal_eot():
    # Execute the actual method without importing Ray/SGLang GPU dependencies.
    path = Path(__file__).parents[2] / "kdflow/trainer/on_policy_kd_trainer.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_build_exact_rollout_sample")
    namespace = {"torch": torch, "math": math, "trajectory_tokens": trajectory_tokens}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    obj = SimpleNamespace(
        student_processor=Tok(), teacher_processor=Tok(),
        args=SimpleNamespace(kd=SimpleNamespace(kd_algorithm="mp_opd")),
        _encode_prompt_ids=lambda tok, prompt: [9],
    )
    sampled = [2, 10, 11, 107]
    output = {"output_ids": sampled, "prompt_ids": [9],
              "meta_info": {"output_token_logprobs": [[-0.1, i] for i in sampled]}}
    result = namespace["_build_exact_rollout_sample"](obj, "prompt", "teacher", output, "", None)
    assert sampled == [2, 10, 11, 107]
    assert result["stu_input_ids"].tolist() == [9, 2, 10, 11, 107, 1]
    assert result["tea_input_ids"].tolist() == [9, 2, 10, 11, 1]
    assert result["stu_responses"] == ["x\n  "]
    behavior = result["stu_behavior_log_probs"][result["stu_loss_mask"]]
    assert torch.allclose(behavior[:4], torch.full((4,), -0.1))
    assert torch.isnan(behavior[4])
