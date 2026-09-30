"""Timing collector: real worker-log shapes must parse and classify."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.step_timing import parse_step_records, status_from_line, summarize  # noqa: E402

EVERY1_LOG = (
    "\x1b[36m(StudentRayActor pid=9)\x1b[0m [2026-09-17 10:01:00] [INFO] "
    "[on_policy_kd_trainer.py:logging:866] [Rank 0] epoch [1/2], step [1/312], "
    "rollout_time: 14.22, teacher_fwd_time: 3.03, mp_opd_valid_atom_count: 376.0, "
    "valid_student_tokens: 436.53, mp_opd_energy_updates_total: 1.0, optimizer_updates: 1.0, "
    "student_train_wall_time: 110.37, gpu_peak_memory_allocated_gib: 85.31, "
    "gpu_peak_memory_reserved_gib: 101.57, step_wall_time: 137.92\n"
    "[2026-09-17 10:03:00] [INFO] [Rank 0] epoch [1/2], step [2/312], rollout_time: 11.23, "
    "teacher_fwd_time: 2.01, mp_opd_energy_updates_total: 2.0, optimizer_updates: 1.0, "
    "student_train_wall_time: 107.21, gpu_peak_memory_allocated_gib: 80.14, "
    "gpu_peak_memory_reserved_gib: 103.76, step_wall_time: 128.95\n"
)

EVERY4_LOG = (
    "[Rank 0] step [1/312], rollout_time: 9.0, teacher_fwd_time: 2.0, "
    "mp_opd_energy_updates_total: 0.0, student_train_wall_time: 60.0, step_wall_time: 80.0, "
    "gpu_peak_memory_allocated_gib: 70.0\n"
    "[Rank 0] step [2/312], rollout_time: 9.5, teacher_fwd_time: 2.0, "
    "mp_opd_energy_updates_total: 0.0, student_train_wall_time: 62.0, step_wall_time: 82.0, "
    "gpu_peak_memory_allocated_gib: 71.0\n"
    "[Rank 0] step [3/312], rollout_time: 9.2, teacher_fwd_time: 2.1, "
    "mp_opd_energy_updates_total: 0.0, student_train_wall_time: 61.0, step_wall_time: 81.0, "
    "gpu_peak_memory_allocated_gib: 72.0\n"
    "[Rank 0] step [4/312], rollout_time: 10.0, teacher_fwd_time: 2.2, "
    "mp_opd_energy_updates_total: 1.0, student_train_wall_time: 150.0, step_wall_time: 175.0, "
    "gpu_peak_memory_allocated_gib: 88.0\n"
)


def test_parser_strips_ansi_and_reads_counters():
    records = parse_step_records(EVERY1_LOG)
    assert [row["step"] for row in records] == [1, 2]
    assert records[0]["horizon"] == 312
    assert records[0]["step_wall_time"] == 137.92
    assert records[0]["mp_opd_energy_updates_total"] == 1.0


def test_summary_splits_non_meta_from_energy_steps():
    summary = summarize(parse_step_records(EVERY4_LOG), 4)
    assert summary["non_meta_steps"] == [1, 2, 3]
    assert summary["energy_steps"] == [4]
    assert summary["non_meta_step_wall_mean"] == 81.0
    assert summary["energy_step_wall_mean"] == 175.0
    assert summary["peak_allocated_gib_max"] == 88.0


def test_resumed_step_can_be_excluded_from_the_steady_state_mean():
    summary = summarize(parse_step_records(EVERY4_LOG), 4, resumed_step=4)
    assert 4 not in summary["non_meta_steps"]
    assert summary["non_meta_step_wall_mean"] == 81.0


def test_empty_log_is_not_a_pass_signal():
    summary = summarize(parse_step_records(""), 4)
    assert summary["records"] == 0
    assert summary["non_meta_step_wall_mean"] is None


def test_status_marker_reader():
    assert status_from_line("noise\nSTEP_STATUS=completed\n", "STEP_STATUS=") == "completed"
    assert status_from_line("nothing", "STEP_STATUS=") is None
