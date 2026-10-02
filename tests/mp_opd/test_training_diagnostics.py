import pytest
import torch

from kdflow.algorithms._mp_opd_training_diagnostics import (
    logit_gradient_metrics,
    partition_metrics,
)


def test_partition_metrics_expose_local_and_group_signals():
    base = torch.tensor([1.0, 1.2, -0.8, -1.0], dtype=torch.float64)
    weight = torch.ones(4, dtype=torch.float64)
    metrics = partition_metrics(base, weight, ((0, 3), (3, 4)), shuffle_seed=7)

    assert metrics["mp_opd_diag_group_count"].item() == 2
    assert metrics["mp_opd_diag_group_length_mean"].item() == 2
    assert abs(metrics["mp_opd_diag_rate_conservation_residual"].item()) < 1e-12
    assert metrics["mp_opd_diag_group_sign_flip_fraction"].item() > 0
    assert metrics["mp_opd_diag_pooled_abs_mass_ratio"].item() < 1
    assert metrics["mp_opd_diag_abs_mass_cancelled_fraction"].item() > 0
    assert torch.isfinite(metrics["mp_opd_diag_lag1_corr"])
    assert torch.isfinite(metrics["mp_opd_diag_shuffled_within_group_rate_rmse"])


def test_logit_gradient_probe_is_exact_for_constant_credit():
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 2, 3])
    token_nll = -torch.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    rate = torch.full((3,), 0.75, dtype=torch.float64)
    weight = torch.ones(3, dtype=torch.float64)

    metrics = logit_gradient_metrics(
        logits,
        token_nll,
        rate,
        weight,
        ((0, 2), (2, 3)),
        shuffle_seed=11,
    )

    assert metrics["mp_opd_diag_logit_grad_cosine"].item() == pytest.approx(1.0)
    assert metrics["mp_opd_diag_logit_grad_delta_norm"].item() == pytest.approx(0.0)
    assert metrics["mp_opd_diag_logit_grad_norm_ratio"].item() == pytest.approx(1.0)


def test_logit_gradient_probe_places_non_atomized_tokens_at_zero():
    """A real microbatch leaves masked-EOS tokens outside every atom (B200, 2026-10-02).

    The probe must ignore those tokens exactly like the production loss does. Changing
    the NLL of an uncovered token must not move any reported metric.
    """
    torch.manual_seed(3)
    logits = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 4, 5])
    token_nll = -torch.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    rate = torch.tensor([0.75, -0.25], dtype=torch.float64)
    weight = torch.tensor([2.0, 2.0], dtype=torch.float64)
    ranges = ((0, 2), (3, 5))          # token 2 belongs to no atom
    partition = ((0, 1), (1, 2))

    baseline = logit_gradient_metrics(
        logits, token_nll, rate, weight, partition, shuffle_seed=5, atom_ranges=ranges
    )
    perturbed_nll = token_nll.clone()
    perturbed_nll[2] = 1e3
    perturbed = logit_gradient_metrics(
        logits, perturbed_nll, rate, weight, partition, shuffle_seed=5, atom_ranges=ranges
    )
    for key, value in baseline.items():
        assert torch.equal(value, perturbed[key]), key

    # Without the ranges the old behaviour stays fail-closed rather than misaligning.
    with pytest.raises(ValueError, match="do not cover token_nll"):
        logit_gradient_metrics(logits, token_nll, rate, weight, partition, shuffle_seed=5)

    # A range that disagrees with the atom's token count is a caller bug, not a gap.
    with pytest.raises(ValueError, match="atom token count and atom range disagree"):
        logit_gradient_metrics(
            logits, token_nll, rate, weight, partition, shuffle_seed=5,
            atom_ranges=((0, 3), (3, 5)),
        )
