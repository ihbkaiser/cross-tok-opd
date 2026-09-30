"""Terminal-state and counter classification for training runs.

Kept separate from the Modal harness so the rules can be unit-tested on CPU:
the session delta always comes from the session counters, never from cumulative
rollout iterations, and a failed app is never relabelled as completed.
"""
from __future__ import annotations

from typing import Any, Optional

COUNTER_FIELDS = (
    "completed_optimizer_updates",
    "current_optimizer_updates",
    "optimizer_updates",
    "session_completed_updates",
    "session_start_optimizer_updates",
    "energy_updates",
    "energy_updates_total",
    "rollout_iterations",
    "scientific_total_updates",
    "status",
    "stop_reason",
)


def parse_run_counters(summary: Any) -> dict:
    """Extract the counters that matter, keeping absent fields absent."""
    if not isinstance(summary, dict):
        return {"error": "summary_not_a_mapping"}
    return {field: summary[field] for field in COUNTER_FIELDS if field in summary}


def optimizer_updates(summary: dict) -> Optional[int]:
    for field in ("completed_optimizer_updates", "current_optimizer_updates", "optimizer_updates"):
        if field in summary:
            return int(summary[field])
    return None


def session_delta(summary: dict) -> Optional[int]:
    """Delta of the current session, derived only from session counters."""
    if "session_completed_updates" in summary:
        return int(summary["session_completed_updates"])
    if "session_start_optimizer_updates" in summary and optimizer_updates(summary) is not None:
        return optimizer_updates(summary) - int(summary["session_start_optimizer_updates"])
    return None


def classify_terminal(*, app_state: str, child_exit: Optional[int], summary: Any,
                      expected_start: Optional[int] = None,
                      expected_total: Optional[int] = None,
                      expected_session_delta: Optional[int] = None) -> dict:
    """Independent verdict: app state, child exit and counters are reported apart."""
    counters = parse_run_counters(summary)
    total = optimizer_updates(summary) if isinstance(summary, dict) else None
    delta = session_delta(summary) if isinstance(summary, dict) else None
    start = summary.get("session_start_optimizer_updates") if isinstance(summary, dict) else None
    reasons: list[str] = []
    if counters.get("error"):
        reasons.append("counters_unavailable")
    if child_exit != 0:
        reasons.append(f"child_exit={child_exit}")
    if summary is None:
        reasons.append("summary_missing")
    if expected_start is not None and start is None:
        reasons.append("start_absent")
    elif expected_start is not None and start is not None and int(start) != expected_start:
        reasons.append(f"start={start}!={expected_start}")
    if expected_total is not None and total is None:
        reasons.append("total_absent")
    elif expected_total is not None and total is not None and total != expected_total:
        reasons.append(f"total={total}!={expected_total}")
    if expected_session_delta is not None and delta is None:
        reasons.append("session_delta_absent")
    elif expected_session_delta is not None and delta != expected_session_delta:
        reasons.append(f"session_delta={delta}!={expected_session_delta}")
    return {
        "app_state": app_state,
        "child_exit": child_exit,
        "counter_status": "ok" if not counters.get("error") else "unavailable",
        "optimizer_updates_total": total,
        "session_start_optimizer_updates": start,
        "session_delta": delta,
        "counter_fields": counters,
        "verdict": "pass" if not reasons else "fail",
        "reasons": reasons,
        "note": "app_state and child_exit stay independent of any offline recovery",
    }