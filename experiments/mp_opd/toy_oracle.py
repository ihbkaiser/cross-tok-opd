#!/usr/bin/env python3
"""Deterministic oracle falsification harness for MP-OPD v0.

The harness is intentionally an adapter-only analytic fixture. It validates
the sign/unit convention, DP oracle, actual one-step virtual updates and a
semi-Markov surrogate hypergradient. It is not LLM training evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

os.environ.setdefault("KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT", "1")

from kdflow.algorithms._mp_opd_credit import hard_partition_loss
from kdflow.algorithms._mp_opd_functional import centered_finite_difference
from kdflow.algorithms._mp_opd_oracle import (
    enumerate_partitions,
    hard_max_partition,
    partition_score,
    span_utility_table,
)
from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        if name.startswith("experiments/mp_opd/results/"):
            continue
        path = ROOT / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return {"git_head": head, "working_tree_sha256": digest.hexdigest()}


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    x, y = ranks(left), ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("toy_oracle_v0.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "toy_oracle_v0.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.manual_seed(config["seed"])
    started = time.perf_counter()

    # Stable, disjoint B/M fixtures. Only the four-dimensional adapter is trainable.
    x = torch.tensor(
        [[1.0, 0.2, -0.3, 0.5], [-0.4, 0.8, 0.1, 0.2], [0.3, -0.7, 1.1, 0.0],
         [0.5, 0.4, -0.2, -0.8], [-0.6, 0.1, 0.7, 0.9], [0.2, -0.5, -0.4, 1.2]],
        dtype=torch.float64,
    )
    theta = torch.tensor([0.15, -0.25, 0.4, -0.1], dtype=torch.float64, requires_grad=True)
    teacher_scores = torch.tensor([-0.3, -0.8, -0.25, -0.9, -0.5, -0.35], dtype=torch.float64)
    weights = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.float64)
    old_scores = x @ theta
    base = (teacher_scores - old_scores.detach()).detach()
    current_nll = -(x @ theta)

    meta_a = torch.tensor(
        [[1.0, -0.2, 0.3, 0.1], [0.2, 0.9, -0.4, 0.5], [-0.5, 0.1, 0.8, -0.3]],
        dtype=torch.float64,
    )
    meta_y = torch.tensor([0.2, -0.3, 0.65], dtype=torch.float64)

    def outer(value: torch.Tensor) -> torch.Tensor:
        residual = meta_a @ value - meta_y
        return 0.5 * residual.square().mean()

    outer_before = outer(theta)
    v = torch.autograd.grad(outer_before, theta, retain_graph=True)[0]
    z_explicit = torch.stack(
        [(v * torch.autograd.grad((x @ theta)[i], theta, retain_graph=True)[0]).sum() for i in range(x.shape[0])]
    ).detach()
    utilities, valid = span_utility_table(base, weights, z_explicit, config["max_span_length"])
    oracle = hard_max_partition(utilities, valid)
    atomic = tuple((i, i + 1) for i in range(x.shape[0]))

    def virtual(partition):
        loss = hard_partition_loss(current_nll, base, weights, partition)
        gradient = torch.autograd.grad(loss, theta, retain_graph=True)[0]
        updated = theta.detach() - config["virtual_learning_rate"] * gradient.detach()
        return loss.detach(), gradient.detach(), updated, outer(updated).detach()

    atomic_loss, atomic_gradient, theta_atomic, outer_atomic = virtual(atomic)
    oracle_loss, oracle_gradient, theta_oracle, outer_oracle = virtual(oracle.partition)
    partitions = list(enumerate_partitions(x.shape[0], config["max_span_length"]))
    predicted = [float(partition_score(utilities, part)) for part in partitions]
    actual = []
    for part in partitions:
        _loss, _gradient, _updated, value = virtual(part)
        actual.append(float(outer_before.detach() - value))

    # Surrogate hypergradient: -eta sum_c mu(c) stopgrad(U_c).
    energy = torch.linspace(-0.3, 0.4, utilities.numel(), dtype=torch.float64).reshape_as(utilities)
    energy = energy.requires_grad_(True)
    distribution = semi_markov_partition(energy, temperature=config["partition_temperature"], valid_mask=valid)
    masked_utilities = torch.where(valid, utilities.double(), torch.zeros_like(utilities, dtype=torch.float64))
    surrogate = -config["virtual_learning_rate"] * (distribution.marginals.double() * masked_utilities.detach()).sum()
    hypergradient = torch.autograd.grad(surrogate, energy)[0]
    direction = torch.randn_like(energy)

    def surrogate_at(candidate):
        q = semi_markov_partition(candidate, temperature=config["partition_temperature"], valid_mask=valid)
        return -config["virtual_learning_rate"] * (q.marginals.double() * masked_utilities).sum()

    finite = centered_finite_difference(
        surrogate_at, energy.detach(), direction, config["finite_difference_epsilon"]
    )
    analytic = (hypergradient * direction).sum()
    fd_error = float((finite - analytic).abs())
    delta_pred = config["virtual_learning_rate"] * (
        float(oracle.score) - float(partition_score(utilities, atomic))
    )
    delta_actual = float(outer_atomic - outer_oracle)
    result = {
        "schema_version": "mp-opd-result-v0",
        "epistemic_label": "oracle_diagnostic",
        "source": source_identity(),
        "config_sha256": sha256(args.config),
        "config": config,
        "split_disjoint": len({item for values in config["split"].values() for item in values})
        == sum(len(values) for values in config["split"].values()),
        "oracle": {
            "atomic_partition": atomic,
            "oracle_partition": oracle.partition,
            "partition_count": len(partitions),
            "delta_pred": delta_pred,
            "delta_actual": delta_actual,
            "sign_agreement": (delta_pred >= 0) == (delta_actual >= 0),
            "spearman_predicted_actual": correlation(predicted, actual),
            "outer_nll_before": float(outer_before.detach()),
            "outer_nll_after_atomic": float(outer_atomic),
            "outer_nll_after_oracle": float(outer_oracle),
            "atomic_gradient_norm": float(atomic_gradient.norm()),
            "oracle_gradient_norm": float(oracle_gradient.norm()),
            "real_parameter_unchanged": bool(torch.equal(theta.detach(), torch.tensor([0.15, -0.25, 0.4, -0.1], dtype=torch.float64))),
        },
        "semi_markov": {
            "log_z": float(distribution.log_z.detach()),
            "entropy": float(distribution.entropy.detach()),
            "expected_span_count": float(distribution.expected_span_count.detach()),
            "expected_span_length": float(distribution.expected_span_length.detach()),
            "coverage_max_error": float(distribution.coverage_max_error.detach()),
            "finite_difference_directional_error": fd_error,
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    numeric = [
        delta_pred, delta_actual, result["oracle"]["spearman_predicted_actual"],
        result["semi_markov"]["log_z"], result["semi_markov"]["entropy"], fd_error,
    ]
    result["test_gate"] = {
        "finite": all(math.isfinite(value) for value in numeric),
        "split_disjoint": result["split_disjoint"],
        "real_parameter_unchanged": result["oracle"]["real_parameter_unchanged"],
        "dp_coverage_lt_1e_6": result["semi_markov"]["coverage_max_error"] < 1e-6,
        "finite_difference_lt_1e_6": fd_error < 1e-6,
        "oracle_actual_headroom_positive": delta_actual > 0,
    }
    result["test_gate"]["passed"] = all(result["test_gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "test_gate": result["test_gate"]}, sort_keys=True))
    return 0 if result["test_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
