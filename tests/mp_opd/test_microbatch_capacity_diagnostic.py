"""The bounded microbatch diagnostic must report exactly what it ran under."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "experiments/runai")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

SPEC = importlib.util.spec_from_file_location(
    "microbatch_diagnostic", ROOT / "experiments/runai/microbatch_diagnostic.py"
)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def test_diag_config_changes_only_the_microbatch_pair():
    import queue_full_alternating as F

    baseline = next(r for r in F.configurations() if r["id"] == f"ALT-main-s{F.TRAIN_SEEDS[0]}")
    config = D.diag_config(2, 4)
    assert (config["micro_B"], config["micro_M"]) == (2, 4)
    trimmed = lambda value: {k: v for k, v in value.items() if k not in ("micro_B", "micro_M")}
    assert trimmed(config) == trimmed(baseline)
    # The campaign recipe itself must stay pinned at micro1.
    assert all(run["micro_B"] == 1 and run["micro_M"] == 4 for run in F.configurations())


def test_summarise_reads_the_peak_and_the_full_meta_trace(tmp_path):
    run = tmp_path / "arm"
    (run / "checkpoint").mkdir(parents=True)
    (run / "checkpoint/run-summary.json").write_text(json.dumps({
        "optimizer_updates": 2, "energy_updates": 2, "status": "completed",
        "session_fit_seconds": 500.0,
    }))
    log = tmp_path / "arm.attempt-1.log"
    log.write_text(
        "x, gpu_peak_memory_allocated_gib: 155.1, gpu_peak_memory_reserved_gib: 179.2, y\n"
        + "FULL_META_MEMORY " + json.dumps({
            "phase": "inner_forward", "allocated_bytes": 10,
            "peak_allocated_bytes": int(200 * 2 ** 30),
        }) + "\n"
    )
    record = D.summarise(log, run)
    assert record["gpu_peak_memory_allocated_gib"] == 155.1
    assert record["gpu_peak_memory_reserved_gib"] == 179.2
    assert record["full_meta_trace_lines"] == 1
    assert record["full_meta_arithmetic_peak_gib"] == 200.0
    assert record["optimizer_updates"] == 2 and record["status"] == "completed"
    assert record["launch_config"] is False


def test_updates_above_the_diagnostic_bound_are_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["microbatch_diagnostic.py", "arm", "2", "4", "31"])
    try:
        D.main()
    except ValueError as error:
        assert "1-30" in str(error)
    else:
        raise AssertionError("31 updates must not be admitted as a diagnostic")
