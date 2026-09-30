"""Validation for fixed-prefix prompt-logprob probes.

The probe exists to separate "served weights and kernels are reproducible" from
"the sampler reproduces". It must never pass on degenerate evidence, so an
empty, missing, sentinel-only or misaligned field is a failure, not a pass.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

FIELD = "input_token_logprobs"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _entry_ok(entry: Any) -> bool:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return False
    head = entry[0]
    if head is not None and not _is_number(head):
        return False
    return isinstance(entry[1], int) and not isinstance(entry[1], bool)


def parse_prompt_logprobs(
    meta_info: Any,
    prompt_ids: Sequence[int],
    *,
    min_entries: int = 2,
) -> dict:
    """Validate one response's prompt logprobs against the prompt it came from."""
    report: dict = {"field": FIELD, "prompt_tokens": len(prompt_ids)}
    if not isinstance(meta_info, dict):
        return {**report, "status": "fail", "reason": "meta_info_not_a_mapping"}
    if FIELD not in meta_info:
        return {**report, "status": "fail", "reason": "field_missing"}
    series = meta_info[FIELD]
    if not isinstance(series, list):
        return {**report, "status": "fail", "reason": "field_not_a_list"}
    report["entries"] = len(series)
    if not series:
        return {**report, "status": "fail", "reason": "empty"}
    if len(series) < min_entries:
        return {**report, "status": "fail", "reason": "too_few_entries"}

    bad_shape = [index for index, entry in enumerate(series) if not _entry_ok(entry)]
    if bad_shape:
        return {**report, "status": "fail", "reason": "bad_entry_shape",
                "bad_entry_indices": bad_shape[:8]}

    values = [entry[0] for entry in series]
    non_null = [value for value in values if value is not None]
    finite = [value for value in non_null if math.isfinite(float(value))]
    report.update({
        "non_null": len(non_null),
        "finite": len(finite),
        "null_positions": [index for index, value in enumerate(values) if value is None][:8],
    })
    if not non_null:
        return {**report, "status": "fail", "reason": "all_null"}
    if not finite:
        return {**report, "status": "fail", "reason": "all_non_finite"}

    if len(series) != len(prompt_ids):
        return {**report, "status": "fail", "reason": "entry_count_mismatch",
                "series_len": len(series), "prompt_len": len(prompt_ids)}

    mismatched = [index for index, entry in enumerate(series)
                  if entry[1] != prompt_ids[index]]
    report["token_id_mismatches"] = len(mismatched)
    if mismatched:
        return {**report, "status": "fail", "reason": "token_id_alignment",
                "mismatch_indices": mismatched[:8]}

    report.update({
        "status": "pass",
        "reason": "aligned_prompt_logprobs",
        "head": [round(float(value), 9) for value in finite[:4]],
        "tail": [round(float(value), 9) for value in finite[-4:]],
        "sum": round(sum(float(value) for value in finite), 6),
    })
    return report


def normalise_prompt_logprob_payloads(payload: Any) -> tuple[list[Any], str]:
    """Return (meta_infos, shape) for single-prompt and batched responses."""
    if isinstance(payload, dict):
        return [payload.get("meta_info", payload)], "single"
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return [item.get("meta_info", item) for item in payload], "batch"
    return [], "unsupported"


def parse_probe_batch(payload: Any, prompt_ids: Sequence[Sequence[int]]) -> dict:
    """Validate a probe payload; a mixed batch passes only if every row passes."""
    metas, shape = normalise_prompt_logprob_payloads(payload)
    if shape == "unsupported":
        return {"status": "fail", "reason": "unsupported_payload_shape", "shape": shape}
    if len(metas) != len(prompt_ids):
        return {"status": "fail", "reason": "response_count_mismatch", "shape": shape,
                "responses": len(metas), "requests": len(prompt_ids)}
    rows = [parse_prompt_logprobs(meta, ids) for meta, ids in zip(metas, prompt_ids)]
    failed = [row for row in rows if row["status"] != "pass"]
    return {
        "status": "fail" if failed else "pass",
        "shape": shape,
        "rows": len(rows),
        "failed_rows": len(failed),
        "first_failure": failed[0] if failed else None,
        "reasons": sorted({row.get("reason") for row in failed}),
    }
