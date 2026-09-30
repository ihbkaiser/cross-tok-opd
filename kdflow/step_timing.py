"""Per-update timing collector for the worker log (CPU-only, unit tested).

Separates recurring non-meta work from energy-update work and from session
overheads (startup, pause, save/export) so the every4 model
T ~= F + 312*C + 78*H + S can be fitted from evidence instead of guessed.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

ANSI = re.compile("\x1b\\[[0-9;]*m")
STEP = re.compile("step \\[(\\d+)/(\\d+)\\]")
NUMERIC = (
    "step_wall_time", "student_train_wall_time", "student_train_time", "rollout_time",
    "teacher_fwd_time", "weight_update_time", "gpu_peak_memory_allocated_gib",
    "gpu_peak_memory_reserved_gib", "gpu_memory_allocated_gib", "gpu_memory_reserved_gib",
    "valid_student_tokens", "valid_teacher_tokens", "mp_opd_valid_atom_count",
    "mp_opd_energy_updates_total", "optimizer_updates", "completed_optimizer_updates",
    "session_fit_seconds", "processed_valid_student_tokens", "eta_campaign_seconds",
)


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def parse_step_records(text: str) -> list:
    records: list = []
    for line in strip_ansi(text).splitlines():
        if "step [" not in line or "step_wall_time" not in line:
            continue
        match = STEP.search(line)
        if not match:
            continue
        record: dict = {"step": int(match.group(1)), "horizon": int(match.group(2))}
        for key in NUMERIC:
            found = re.search(key + "[:=] ([0-9.eE+-]+)", line)
            if found:
                record[key] = float(found.group(1))
        records.append(record)
    return records


def summarize(records: Iterable, energy_every: int, *, resumed_step: Optional[int] = None) -> dict:
    """Mean timing for non-meta steps versus energy-update steps."""
    rows = list(records)
    energy_steps = [row for row in rows
                    if int(row.get("mp_opd_energy_updates_total", 0)) > 0]
    energy_ids = {id(row) for row in energy_steps}
    non_meta = [row for row in rows if id(row) not in energy_ids]
    if resumed_step is not None:
        non_meta = [row for row in non_meta if row["step"] != resumed_step]

    def mean(rows_, key):
        values = [row[key] for row in rows_ if key in row]
        return round(sum(values) / len(values), 3) if values else None

    return {
        "energy_every": energy_every,
        "records": len(rows),
        "steps": [row["step"] for row in rows],
        "horizon": rows[0]["horizon"] if rows else None,
        "non_meta_steps": [row["step"] for row in non_meta],
        "energy_steps": [row["step"] for row in energy_steps],
        "non_meta_step_wall_mean": mean(non_meta, "step_wall_time"),
        "energy_step_wall_mean": mean(energy_steps, "step_wall_time"),
        "non_meta_student_wall_mean": mean(non_meta, "student_train_wall_time"),
        "energy_student_wall_mean": mean(energy_steps, "student_train_wall_time"),
        "rollout_mean": mean(rows, "rollout_time"),
        "teacher_mean": mean(rows, "teacher_fwd_time"),
        "weight_update_mean": mean(rows, "weight_update_time"),
        "peak_allocated_gib_max": max((row.get("gpu_peak_memory_allocated_gib", 0) for row in rows), default=None),
        "peak_reserved_gib_max": max((row.get("gpu_peak_memory_reserved_gib", 0) for row in rows), default=None),
    }


def status_from_line(text: str, prefix: str) -> Optional[str]:
    for line in strip_ansi(text).splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None
