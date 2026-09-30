"""Collector rules: counters, deltas and terminal states must not be conflated."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.run_counters import (  # noqa: E402
    classify_terminal,
    optimizer_updates,
    parse_run_counters,
    session_delta,
)

REFERENCE = {
    "status": "completed", "completed_optimizer_updates": 2, "optimizer_updates": 2,
    "session_completed_updates": 2, "session_start_optimizer_updates": 0,
    "energy_updates_total": 2, "rollout_iterations": 2, "scientific_total_updates": 312,
    "stop_reason": None,
}
RESUMED = {
    "status": "completed", "completed_optimizer_updates": 2, "optimizer_updates": 2,
    "session_completed_updates": 1, "session_start_optimizer_updates": 1,
    "energy_updates_total": 2, "rollout_iterations": 2, "scientific_total_updates": 312,
    "stop_reason": None,
}


def test_reference_counters_parse():
    counters = parse_run_counters(REFERENCE)
    assert counters["completed_optimizer_updates"] == 2
    assert optimizer_updates(REFERENCE) == 2
    assert session_delta(REFERENCE) == 2


def test_resume_delta_uses_session_counter_not_cumulative_rollouts():
    assert RESUMED["rollout_iterations"] == 2
    assert session_delta(RESUMED) == 1
    verdict = classify_terminal(app_state="completed", child_exit=0, summary=RESUMED,
                                expected_start=1, expected_total=2, expected_session_delta=1)
    assert verdict["verdict"] == "pass", verdict
    assert verdict["session_delta"] == 1


def test_cumulative_rollout_iterations_alone_is_not_a_delta():
    summary = {"completed_optimizer_updates": 2, "optimizer_updates": 2, "rollout_iterations": 2}
    assert session_delta(summary) is None


def test_wrong_resumed_delta_fails():
    verdict = classify_terminal(app_state="completed", child_exit=0, summary=REFERENCE,
                                expected_start=1, expected_total=2, expected_session_delta=1)
    assert verdict["verdict"] == "fail"
    assert any(reason.startswith("start=") for reason in verdict["reasons"])


def test_missing_summary_fails():
    verdict = classify_terminal(app_state="completed", child_exit=0, summary=None,
                                expected_total=2)
    assert verdict["verdict"] == "fail"
    assert "summary_missing" in verdict["reasons"]


def test_null_summary_fails():
    verdict = classify_terminal(app_state="completed", child_exit=0, summary={"status": None},
                                expected_total=2)
    assert verdict["verdict"] == "fail"
    assert "total_absent" in verdict["reasons"]


def test_summary_without_counters_never_passes():
    verdict = classify_terminal(app_state="completed", child_exit=0, summary={},
                                expected_start=1, expected_total=2, expected_session_delta=1)
    assert verdict["verdict"] == "fail"
    assert {"start_absent", "total_absent", "session_delta_absent"} <= set(verdict["reasons"])


def test_child_exit_and_app_state_stay_independent():
    verdict = classify_terminal(app_state="failed", child_exit=0, summary=RESUMED,
                                expected_start=1, expected_total=2, expected_session_delta=1)
    assert verdict["verdict"] == "pass"
    assert verdict["app_state"] == "failed"
    assert "app_state and child_exit stay independent" in verdict["note"]


def test_failed_child_fails_even_with_complete_counters():
    verdict = classify_terminal(app_state="stopped", child_exit=1, summary=REFERENCE,
                                expected_start=0, expected_total=2, expected_session_delta=2)
    assert verdict["verdict"] == "fail"
    assert "child_exit=1" in verdict["reasons"]


def test_parity_failure_counter_mismatch_is_reported():
    summary = dict(REFERENCE, completed_optimizer_updates=1, optimizer_updates=1,
                   session_completed_updates=1)
    verdict = classify_terminal(app_state="completed", child_exit=0, summary=summary,
                                expected_start=0, expected_total=2, expected_session_delta=2)
    assert verdict["verdict"] == "fail"
    assert "total=1!=2" in verdict["reasons"]