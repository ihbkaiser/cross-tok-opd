from types import SimpleNamespace

import torch

from kdflow.algorithms.span_ctkd import (
    SpanCrossTokenizerKD,
    memory_efficient_reverse_kl,
)
from kdflow.arguments.distillation_args import DistillationArguments


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 3

    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2, "<eos>": 3}

    def decode(self, ids):
        return {0: "a", 1: "b", 2: "c", 3: "<eos>"}[ids[0]]


class _Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(1, 4, 4))

    def forward(self, *_args, **_kwargs):
        return {"logits": self.logits}


def test_training_step_reports_exact_simct_token_and_span_metrics():
    args = SimpleNamespace(
        kd=DistillationArguments(kd_algorithm="span_ctkd", kd_loss_fn="rkl")
    )
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    tokenizer = _Tokenizer()
    teacher_head = torch.nn.Linear(4, 4, bias=False)
    algorithm = SpanCrossTokenizerKD(
        strategy,
        _Student(),
        teacher_head,
        tokenizer,
        tokenizer,
    )
    batch = {
        "stu_input_ids": torch.tensor([[0, 1, 2, 3]]),
        "stu_attn_mask": torch.ones(1, 4),
        "stu_loss_mask": torch.tensor([[True, True, True, False]]),
        "tea_input_ids": torch.tensor([[0, 1, 2, 3]]),
        "tea_attn_mask": torch.ones(1, 4),
        "tea_loss_mask": torch.tensor([[True, True, True, False]]),
        "teacher_hiddens": torch.randn(3, 4),
        "avg_micro_batch_token_num": torch.tensor(3.0),
    }

    metrics = algorithm.training_step(batch)

    expected = {
        "loss",
        "kd_loss",
        "align_ratio",
        "gh_mask_ratio",
        "valid_student_tokens",
        "valid_teacher_tokens",
        "aligned_token_count",
        "aligned_segment_count",
        "span_segment_count",
        "span_segment_ratio",
        "gh_kept_segment_count",
        "gh_masked_segment_count",
        "teacher_candidate_mass_mean",
        "teacher_candidate_mass_min",
        "teacher_candidate_mass_max",
    }
    assert expected <= metrics.keys()
    assert metrics["valid_student_tokens"].item() == 3
    assert metrics["valid_teacher_tokens"].item() == 3
    assert metrics["aligned_segment_count"].item() == 3
    assert metrics["span_segment_count"].item() == 0
    assert all(torch.isfinite(metrics[key]).item() for key in expected)


def test_vectorized_virtual_vocab_matches_segment_reference():
    args = SimpleNamespace(
        kd=DistillationArguments(kd_algorithm="span_ctkd", kd_loss_fn="rkl")
    )
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    tokenizer = _Tokenizer()
    algorithm = SpanCrossTokenizerKD(
        strategy,
        _Student(),
        torch.nn.Linear(4, 4, bias=False),
        tokenizer,
        tokenizer,
    )
    segments = [(0, 1, 0, 1), (1, 3, 1, 2), (3, 4, 2, 4)]
    stu_ids = [0, 1, 2, 3]
    tea_ids = [0, 1, 2, 3]
    stu_logits = torch.randn(4, 4, requires_grad=True)
    tea_logits = torch.randn(4, 4)

    actual_student, actual_teacher = algorithm._build_virtual_vocab_logits(
        segments, stu_logits, tea_logits, stu_ids, tea_ids
    )

    span_indices = [1, 2]
    expected_student = []
    expected_teacher = []
    for segment_index, (ts, te, ss, se) in enumerate(segments):
        student_row = stu_logits[ss][algorithm.student_overlap_token_ids]
        teacher_row = tea_logits[ts][algorithm.teacher_overlap_token_ids]
        student_spans = torch.full((len(span_indices),), -1e9)
        teacher_spans = torch.full((len(span_indices),), -1e9)
        if segment_index in span_indices:
            span_column = span_indices.index(segment_index)
            student_spans[span_column] = torch.stack(
                [stu_logits[position, token] for position, token in zip(range(ss, se), stu_ids[ss:se])]
            ).mean()
            teacher_spans[span_column] = torch.stack(
                [tea_logits[position, token] for position, token in zip(range(ts, te), tea_ids[ts:te])]
            ).mean()
        expected_student.append(torch.cat([student_row, student_spans]))
        expected_teacher.append(torch.cat([teacher_row, teacher_spans]))

    torch.testing.assert_close(actual_student, torch.stack(expected_student))
    torch.testing.assert_close(actual_teacher, torch.stack(expected_teacher))
    actual_student.sum().backward()
    assert stu_logits.grad is not None
    assert torch.isfinite(stu_logits.grad).all()


def test_vectorized_teacher_candidate_mass_matches_softmax_reference():
    args = SimpleNamespace(
        kd=DistillationArguments(kd_algorithm="span_ctkd", kd_loss_fn="rkl")
    )
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    tokenizer = _Tokenizer()
    algorithm = SpanCrossTokenizerKD(
        strategy,
        _Student(),
        torch.nn.Linear(4, 4, bias=False),
        tokenizer,
        tokenizer,
    )
    segments = [(0, 1, 0, 1), (1, 3, 1, 2), (3, 4, 2, 4)]
    teacher_ids = [0, 1, 2, 3]
    logits = torch.randn(4, 7)

    actual = algorithm._compute_segment_teacher_Z(segments, logits, teacher_ids)
    expected = []
    for ts, te, ss, se in segments:
        mass = torch.softmax(logits[ts].float(), dim=-1)[
            algorithm.teacher_overlap_token_ids
        ].sum()
        if (te - ts) > 1 or (se - ss) > 1:
            logps = torch.stack(
                [
                    torch.log_softmax(logits[position].float(), dim=-1)[token]
                    for position, token in zip(range(ts, te), teacher_ids[ts:te])
                ]
            )
            mass = mass + torch.exp(logps.mean())
        expected.append(mass.clamp(max=1.0))

    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)


def test_memory_efficient_reverse_kl_matches_reference_value_and_gradient():
    torch.manual_seed(7)
    teacher = torch.randn(11, 37)
    reference_student = torch.randn(11, 37, requires_grad=True)
    actual_student = reference_student.detach().clone().requires_grad_(True)
    weights = torch.linspace(0.1, 1.1, 11)
    temperature = 0.73

    reference_log_probs = torch.log_softmax(reference_student / temperature, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher / temperature, dim=-1)
    reference = (
        reference_log_probs.exp() * (reference_log_probs - teacher_log_probs)
    ).sum(dim=-1)
    (reference * weights).sum().backward()

    actual = memory_efficient_reverse_kl(
        actual_student, teacher, temperature=temperature, row_chunk_size=3
    )
    (actual * weights).sum().backward()

    torch.testing.assert_close(actual, reference.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual_student.grad, reference_student.grad, rtol=2e-5, atol=2e-6
    )
