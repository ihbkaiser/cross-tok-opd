"""The campaign serving contract must reach the worker env, or fail closed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/runai"))
sys.path.insert(0, str(ROOT))

from queue_full_alternating import SERVING_ATTENTION_BACKENDS, serving_env  # noqa: E402

DETERMINISTIC = {
    "serving": {
        "deterministic": True,
        "random_seed": 42,
        "attention_backend": "triton",
        "disable_radix_cache": True,
    }
}


def test_absent_serving_block_keeps_the_historical_launch():
    env = serving_env({})
    assert env == {
        "MP_ROLLOUT_DETERMINISTIC": "0",
        "MP_ROLLOUT_SEED": "-1",
        "MP_ROLLOUT_ATTENTION_BACKEND": "",
        "MP_ROLLOUT_DISABLE_RADIX_CACHE": "0",
    }


def test_complete_deterministic_contract_is_forwarded():
    env = serving_env(DETERMINISTIC)
    assert env["MP_ROLLOUT_DETERMINISTIC"] == "1"
    assert env["MP_ROLLOUT_SEED"] == "42"
    assert env["MP_ROLLOUT_ATTENTION_BACKEND"] == "triton"
    assert env["MP_ROLLOUT_DISABLE_RADIX_CACHE"] == "1"


def test_backend_only_campaign_is_allowed_and_stays_non_deterministic():
    env = serving_env({"serving": {"attention_backend": "flashinfer"}})
    assert env["MP_ROLLOUT_DETERMINISTIC"] == "0"
    assert env["MP_ROLLOUT_ATTENTION_BACKEND"] == "flashinfer"


@pytest.mark.parametrize("backend", ["", None, "trtllm_mha", "flash_attention"])
def test_deterministic_without_a_supported_backend_fails_closed(backend):
    campaign = {"serving": dict(DETERMINISTIC["serving"], attention_backend=backend)}
    with pytest.raises(ValueError):
        serving_env(campaign)


def test_deterministic_without_pinned_cache_fails_closed():
    campaign = {"serving": dict(DETERMINISTIC["serving"], disable_radix_cache=False)}
    with pytest.raises(ValueError):
        serving_env(campaign)


def test_deterministic_without_a_seed_fails_closed():
    campaign = {"serving": dict(DETERMINISTIC["serving"], random_seed=-1)}
    with pytest.raises(ValueError):
        serving_env(campaign)


def test_non_integer_seed_is_rejected():
    campaign = {"serving": dict(DETERMINISTIC["serving"], random_seed="many")}
    with pytest.raises(ValueError):
        serving_env(campaign)


def test_supported_backend_set_matches_the_pinned_sglang_allowlist():
    assert SERVING_ATTENTION_BACKENDS == frozenset({"triton", "flashinfer", "fa3"})
