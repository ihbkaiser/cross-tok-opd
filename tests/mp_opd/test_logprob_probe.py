"""Fixed-prefix prompt-logprob parser: degenerate evidence must never pass."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.logprob_probe import (  # noqa: E402
    parse_probe_batch,
    parse_prompt_logprobs,
)

PROMPT = [10, 11, 12, 13]


def _series(values=(None, -1.5, -2.5, -3.5)):
    return [[value, token, ""] for value, token in zip(values, PROMPT)]


def test_valid_series_passes():
    result = parse_prompt_logprobs({"input_token_logprobs": _series()}, PROMPT)
    assert result["status"] == "pass", result
    assert result["entries"] == 4
    assert result["non_null"] == 3
    assert result["token_id_mismatches"] == 0


def test_missing_field_fails():
    assert parse_prompt_logprobs({}, PROMPT)["reason"] == "field_missing"


def test_empty_series_fails():
    assert parse_prompt_logprobs({"input_token_logprobs": []}, PROMPT)["reason"] == "empty"


def test_wrong_field_type_fails():
    result = parse_prompt_logprobs({"input_token_logprobs": "-1.0,-2.0"}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "field_not_a_list"


def test_list_of_scalars_fails():
    result = parse_prompt_logprobs({"input_token_logprobs": [-1.0, -2.0]}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "bad_entry_shape"


def test_entry_without_token_id_fails():
    result = parse_prompt_logprobs({"input_token_logprobs": [[-1.0], [-2.0]]}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "bad_entry_shape"


def test_all_null_fails_even_with_correct_length():
    result = parse_prompt_logprobs({"input_token_logprobs": _series((None, None, None, None))}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "all_null"


def test_sentinel_only_fails():
    result = parse_prompt_logprobs({"input_token_logprobs": _series((None,))}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] in {"too_few_entries", "entry_count_mismatch"}


def test_token_alignment_is_checked():
    series = _series()
    series[2][1] = 999
    result = parse_prompt_logprobs({"input_token_logprobs": series}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "token_id_alignment"


def test_entry_count_must_match_prompt():
    result = parse_prompt_logprobs({"input_token_logprobs": _series((None, -1.0, -2.0))}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "entry_count_mismatch"


def test_non_finite_only_fails():
    series = _series((None, float("nan"), float("inf"), float("-inf")))
    result = parse_prompt_logprobs({"input_token_logprobs": series}, PROMPT)
    assert result["status"] == "fail"
    assert result["reason"] == "all_non_finite"


def test_single_finite_value_among_non_finite_passes():
    series = _series((None, float("nan"), float("inf"), -1.0))
    result = parse_prompt_logprobs({"input_token_logprobs": series}, PROMPT)
    assert result["status"] == "pass", result
    assert result["finite"] == 1


def test_batch_payload_requires_every_row_to_pass():
    good = {"meta_info": {"input_token_logprobs": _series()}}
    bad = {"meta_info": {"input_token_logprobs": []}}
    assert parse_probe_batch([good, good], [PROMPT, PROMPT])["status"] == "pass"
    mixed = parse_probe_batch([good, bad], [PROMPT, PROMPT])
    assert mixed["status"] == "fail"
    assert mixed["failed_rows"] == 1
    assert mixed["first_failure"]["reason"] == "empty"


def test_single_payload_shape_is_supported():
    payload = {"meta_info": {"input_token_logprobs": _series()}}
    assert parse_probe_batch(payload, [PROMPT])["status"] == "pass"


def test_response_count_mismatch_fails():
    payload = [{"meta_info": {"input_token_logprobs": _series()}}]
    assert parse_probe_batch(payload, [PROMPT, PROMPT])["reason"] == "response_count_mismatch"


def test_unsupported_payload_shape_fails():
    assert parse_probe_batch(None, [PROMPT])["reason"] == "unsupported_payload_shape"
    assert parse_probe_batch([], [PROMPT])["reason"] == "unsupported_payload_shape"