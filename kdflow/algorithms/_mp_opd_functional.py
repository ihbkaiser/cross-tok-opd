"""Mutation-free adapter-subspace helpers for MP-OPD virtual updates."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def virtual_sgd(
    parameters: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    if learning_rate <= 0:
        raise ValueError("virtual learning rate must be positive")
    if parameters.keys() != gradients.keys():
        raise ValueError("parameter and gradient keys differ")
    return {
        name: value - float(learning_rate) * gradients[name]
        for name, value in parameters.items()
    }


def explicit_directional_scores(
    atom_log_scores: torch.Tensor,
    parameters: torch.Tensor,
    outer_gradient: torch.Tensor,
) -> torch.Tensor:
    """Return z_i=<v,grad score_i> without mutating parameters."""
    if atom_log_scores.ndim != 1 or parameters.shape != outer_gradient.shape:
        raise ValueError("invalid score/parameter/outer-gradient shapes")
    scores = []
    for index in range(atom_log_scores.numel()):
        gradient = torch.autograd.grad(
            atom_log_scores[index], parameters, retain_graph=True, create_graph=False
        )[0]
        scores.append((outer_gradient.detach() * gradient).sum())
    return torch.stack(scores).detach()


def centered_finite_difference(function, value: torch.Tensor, direction: torch.Tensor, epsilon: float = 1e-4):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    with torch.no_grad():
        return (
            function(value + epsilon * direction) - function(value - epsilon * direction)
        ) / (2.0 * epsilon)
