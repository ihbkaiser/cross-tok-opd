"""Serving-contract flags for the rollout engine (light module: no ray/torch import).

A single source of truth so the launcher, the CLI and the diagnostics agree on
what the serving process is told. Defaults reproduce the historical launch dict.
"""
from __future__ import annotations

from typing import Any, Dict

DETERMINISTIC_REQUIRED = ("enable_deterministic_inference", "attention_backend", "disable_radix_cache")


def build_extra_server_args(args: Any) -> Dict[str, Any]:
    """Serving flags handed verbatim to SGLang ServerArgs."""
    rollout = args.rollout
    extra: Dict[str, Any] = {}
    if getattr(rollout, "rollout_disable_piecewise_cuda_graph", False):
        extra["disable_piecewise_cuda_graph"] = True
    backend = str(getattr(rollout, "rollout_attention_backend", "") or "")
    if backend:
        extra["attention_backend"] = backend
    deterministic = bool(getattr(rollout, "rollout_deterministic_inference", False))
    if deterministic or getattr(rollout, "rollout_disable_radix_cache", False):
        # Pinned off so a backend swap stays single-variable: deterministic
        # FlashInfer already normalises to radix cache off.
        extra["disable_radix_cache"] = True
    if deterministic:
        extra["enable_deterministic_inference"] = True
        if getattr(rollout, "rollout_random_seed", -1) >= 0:
            extra["random_seed"] = int(rollout.rollout_random_seed)
        # SGLang raises when deterministic inference meets an unsupported backend
        # that the model handler already filled in (Gemma2/B200 -> trtllm_mha).
        if not backend:
            extra["attention_backend"] = "flashinfer"
    return extra


def assert_serving_contract(extra: Dict[str, Any], *, deterministic: bool) -> None:
    """Fail closed when a deterministic serving contract loses a required flag."""
    if not deterministic:
        return
    missing = [name for name in DETERMINISTIC_REQUIRED if name not in extra]
    if missing:
        raise ValueError("Deterministic serving contract is missing required flags: " + ",".join(missing))


def serving_marker(extra: Dict[str, Any]) -> str:
    return ",".join(f"{key}={extra[key]}" for key in sorted(extra))
