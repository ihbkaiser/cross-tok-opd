"""Offline recovery/post-processing for existing Modal soft runs.

This module never launches a workload and never mutates raw result artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COLLECTOR_VERSION = "p1-wrapper-recovery-v1"


def expected_energy_updates(energy_every: int, updates: int) -> int:
    if energy_every <= 0:
        raise ValueError("energy_every must be positive")
    if updates < 0:
        raise ValueError("updates must be non-negative")
    return (updates + energy_every - 1) // energy_every


def recover_result(
    input_path: str | Path,
    *,
    variant: str | None = None,
    app_terminal_state: str = "unknown",
    expected_updates: int = 8,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Validate summaries in one raw result and return a derived receipt."""
    path = Path(input_path)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("raw result must be a JSON object")
    cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    raw_variant = str(raw.get("variant", ""))
    selected_variant = variant or raw_variant
    if not selected_variant:
        raise ValueError("variant is required when raw result has no variant")
    updates = int(cfg.get("updates", expected_updates))
    if updates != expected_updates:
        raise ValueError(f"raw config updates={updates} != expected={expected_updates}")
    energy_every = int(cfg.get("energy_every", 1))
    expected_energy = expected_energy_updates(energy_every, expected_updates)
    phase1 = raw.get("phase1_summary")
    final = raw.get("final_summary")
    phase1_exit = raw.get("phase1_exit")
    phase2_exit = raw.get("phase2_exit")
    training_checks = {
        "phase1_exit_zero": phase1_exit == 0,
        "phase2_exit_zero": phase2_exit == 0,
        "phase1_summary_present": bool(raw.get("phase1_summary_present")) and isinstance(phase1, dict),
        "final_summary_present": isinstance(final, dict),
        "final_status_completed": isinstance(final, dict) and final.get("status") == "completed",
        "optimizer_updates_match": isinstance(final, dict) and int(final.get("optimizer_updates", -1)) == expected_updates,
        "energy_updates_match": isinstance(final, dict) and int(final.get("energy_updates", -1)) == expected_energy,
    }
    training_completed = all(training_checks.values())
    artifact_checks = {"latest_checkpoint_hash_present": bool(raw.get("latest_checkpoint_sha256"))}
    validation_status = "pass" if training_completed and all(artifact_checks.values()) else "pending_artifact_comparator"
    if not training_completed:
        validation_status = "fail"
    return {
        "receipt_version": COLLECTOR_VERSION,
        "input_path": str(path),
        "raw_result_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "run_id": raw.get("run_id"),
        "variant": selected_variant,
        "source_commit": raw.get("source_commit"),
        "config": cfg,
        "training_status": "completed" if training_completed else "failed",
        "postprocess_status": "recovered",
        "validation_status": validation_status,
        "app_terminal_state": app_terminal_state,
        "raw_status": raw.get("status"),
        "raw_error": {"type": raw.get("error_type"), "message": raw.get("error"), "traceback": raw.get("traceback")},
        "expected": {"optimizer_updates": expected_updates, "energy_updates": expected_energy, "energy_every": energy_every},
        "checks": {**training_checks, **artifact_checks},
        "command": command or [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover existing Modal result JSON offline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--app-terminal-state", default="unknown")
    parser.add_argument("--expected-updates", type=int, default=8)
    args = parser.parse_args(argv)
    import sys
    receipt = recover_result(args.input, variant=args.variant, app_terminal_state=args.app_terminal_state,
                             expected_updates=args.expected_updates, command=["recover_soft_alternatives.py", *sys.argv[1:]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".pending")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
