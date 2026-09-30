"""Small detached-feature energy model and independent checkpoint state."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ._mp_opd_semimarkov import semi_markov_partition


class MPAtomEnergy(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, layers: int = 2):
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or layers <= 0:
            raise ValueError("feature_dim, hidden_dim and layers must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.encoder = nn.GRU(
            feature_dim, hidden_dim, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=0.0 if layers == 1 else 0.1,
        )
        encoded = 2 * hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(3 * encoded + 1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_features: torch.Tensor, max_span_length: int) -> torch.Tensor:
        if atom_features.ndim != 2 or atom_features.shape[1] != self.feature_dim:
            raise ValueError("atom_features must have shape [n,feature_dim]")
        n = atom_features.shape[0]
        length = min(max(int(max_span_length), 1), max(n, 1))
        if n == 0:
            return atom_features.new_empty((0, length), dtype=torch.float32)
        # cuDNN inference-mode RNN forward does not retain backward reserves.
        # Native GRU supports eval-mode gradients while keeping dropout disabled.
        # Use the same backend for eval scoring and eval-mode energy learning.
        backend = nullcontext() if self.encoder.training else torch.backends.cudnn.flags(enabled=False)
        with backend:
            encoded, _ = self.encoder(atom_features.detach().float().unsqueeze(0))
        encoded = encoded.squeeze(0)
        prefix = torch.cat((encoded.new_zeros((1, encoded.shape[1])), encoded.cumsum(0)), dim=0)
        output = encoded.new_full((n, length), -torch.inf)
        for start in range(n):
            for offset in range(min(length, n - start)):
                end = start + offset + 1
                pooled = (prefix[end] - prefix[start]) / float(end - start)
                span = torch.cat((encoded[start], encoded[end - 1], pooled, encoded.new_tensor([(end - start) / length])))
                output[start, offset] = self.scorer(span).squeeze(-1)
        return output

    def config(self) -> dict[str, int]:
        return {"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim, "layers": self.layers}


def energy_surrogate_loss(
    model: MPAtomEnergy,
    atom_features: torch.Tensor,
    span_utilities: torch.Tensor,
    *,
    max_span_length: int,
    temperature: float,
    virtual_learning_rate: float,
    valid_mask: torch.Tensor | None = None,
):
    """Exact stated first-order surrogate with detached features/utilities."""
    if virtual_learning_rate <= 0:
        raise ValueError("virtual_learning_rate must be positive")
    energies = model(atom_features.detach(), max_span_length)
    distribution = semi_markov_partition(
        energies, temperature=temperature, valid_mask=valid_mask
    )
    if span_utilities.shape != distribution.marginals.shape:
        raise ValueError("span utilities must match the [n,L] energy layout")
    finite_utilities = torch.where(
        torch.isfinite(span_utilities), span_utilities.detach(), torch.zeros_like(span_utilities)
    )
    objective = -float(virtual_learning_rate) * (
        distribution.marginals * finite_utilities
    ).sum()
    return objective, distribution


def config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_energy_checkpoint(
    path: str | Path,
    model: MPAtomEnergy,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    extra_config: dict[str, Any] | None = None,
) -> None:
    config = {**model.config(), **(extra_config or {})}
    payload = {
        "format": "mp-opd-energy-v0",
        "step": int(step),
        "config": config,
        "config_sha256": config_sha256(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_energy_checkpoint(
    path: str | Path,
    model: MPAtomEnergy,
    optimizer: torch.optim.Optimizer,
    *,
    expected_extra_config: dict[str, Any] | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "mp-opd-energy-v0":
        raise ValueError("unsupported MP-OPD energy checkpoint")
    config = {**model.config(), **(expected_extra_config or {})}
    if payload.get("config_sha256") != config_sha256(config):
        raise ValueError("MP-OPD energy checkpoint config mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["step"])
