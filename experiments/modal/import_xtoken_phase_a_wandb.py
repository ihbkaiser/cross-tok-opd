#!/usr/bin/env python3
"""Import a finalized TensorBoard metric export into one explicit W&B run.

This is intentionally labeled as a historical import. It does not claim that
the original training process used native W&B logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb


RESOURCE_PREFIX = "ray/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    required = {
        name: args.evidence_dir / name
        for name in (
            "finalization-result.json",
            "modal-verified-metrics.json",
            "prepare-result.json",
            "train-result.json",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing finalized evidence: {missing}")

    finalization = read_json(required["finalization-result.json"])
    metrics = read_json(required["modal-verified-metrics.json"])
    prepare = read_json(required["prepare-result.json"])
    train = read_json(required["train-result.json"])

    if finalization.get("status") != "completed":
        raise RuntimeError("finalization status is not completed")
    if finalization.get("optimizer_updates") != 10:
        raise RuntimeError("optimizer update gate is not exactly 10")
    if not metrics.get("metric_gate_pass"):
        raise RuntimeError("finalized TensorBoard metric gate did not pass")

    scalars = metrics.get("scalars")
    if not isinstance(scalars, dict) or not scalars:
        raise RuntimeError("no scalar metrics found")
    bad_values: list[str] = []
    for tag, events in scalars.items():
        if not isinstance(events, list) or not events:
            bad_values.append(f"{tag}:empty")
            continue
        for event in events:
            if not finite(event.get("value")) or not finite(event.get("wall_time")):
                bad_values.append(f"{tag}@{event.get('step')}")
    if bad_values:
        raise RuntimeError(f"non-finite/incomplete scalar evidence: {bad_values[:10]}")

    scientific = dict(finalization["scientific"])
    original_wandb = scientific.pop("wandb", False)
    config: dict[str, Any] = {
        **scientific,
        "original_training_wandb_enabled": original_wandb,
        "wandb_logging_mode": "historical_tensorboard_import",
        "source_metric_format": "TensorBoard event scalars",
        "source_run_id": finalization["run_id"],
        "source_status": finalization["status"],
        "source_training_process_exit_code": finalization["training_process_exit_code"],
        "source_training_stdout_sha256": finalization["training_stdout_sha256"],
        "source_event_file_sha256": finalization["event_file_sha256"],
        "source_config_identity": train["config_identity"],
        "source_prepare": {
            "raw_rows": prepare["raw_rows"],
            "complete_packs": prepare["complete_packs"],
            "data_verified": prepare["data_verified"],
            "models_verified": prepare["models_verified"],
        },
        "operational": finalization["operational"],
        "modal_training_app_id": "ap-ou4gVmixuSYzUtQaQRYnW3",
        "modal_finalizer_app_id": "ap-Na9OWoHO2xEBCTR5OVEfBv",
        "modal_exact_run_cost_usd": 1.14395234,
        "native_lock_sha256": "145d512cf6e56deec88eacfde4159ba97fd55496a26e26d5aec8d33b7ba357cb",
        "student_weight_sha256": "68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a",
        "importer_wandb_version": wandb.__version__,
        "metric_tag_count": len(scalars),
    }

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        group=args.group,
        job_type="historical_tensorboard_import",
        tags=["xtoken", "off-policy", "modal", "A10x2", "historical-import", "10-step"],
        config=config,
        resume="never",
        settings=wandb.Settings(silent=True),
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")

    wandb.define_metric("optimizer_step")
    wandb.define_metric("train/*", step_metric="optimizer_step")
    wandb.define_metric("timing/*", step_metric="optimizer_step")
    wandb.define_metric("resource_sample")
    wandb.define_metric("ray/*", step_metric="resource_sample")

    optimizer_events: dict[int, dict[str, float]] = defaultdict(dict)
    resource_events: dict[int, dict[str, float]] = defaultdict(dict)
    resource_meta: dict[int, dict[str, float]] = {}

    for tag, events in scalars.items():
        if tag.startswith(RESOURCE_PREFIX):
            for sample_index, event in enumerate(events):
                resource_events[sample_index][tag] = float(event["value"])
                resource_meta[sample_index] = {
                    "tensorboard_step": float(event["step"]),
                    "tensorboard_wall_time": float(event["wall_time"]),
                }
        else:
            for event in events:
                optimizer_events[int(event["step"])][tag] = float(event["value"])

    observed_steps = sorted(optimizer_events)
    if observed_steps != list(range(1, 11)):
        raise RuntimeError(f"optimizer metric steps are not exactly 1..10: {observed_steps}")

    for step in observed_steps:
        wandb.log({"optimizer_step": step, **optimizer_events[step]})
    for sample_index in sorted(resource_events):
        wandb.log(
            {
                "resource_sample": sample_index,
                **resource_meta[sample_index],
                **resource_events[sample_index],
            }
        )

    artifact = wandb.Artifact(
        name=f"{args.run_id}-finalized-evidence",
        type="experiment-evidence",
        description="Audited small finalized JSON evidence used for the historical TensorBoard import.",
        metadata={
            "source_run_id": finalization["run_id"],
            "optimizer_updates": 10,
            "metric_tag_count": len(scalars),
            "evidence_sha256": {name: sha256(path) for name, path in required.items()},
        },
    )
    for name, path in required.items():
        artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)

    run.summary["optimizer_updates"] = 10
    run.summary["metric_tag_count"] = len(scalars)
    run.summary["optimizer_metric_step_count"] = len(observed_steps)
    run.summary["resource_sample_count"] = len(resource_events)
    run.summary["all_scalar_values_finite"] = True
    run.summary["source_metric_gate_pass"] = True
    run_url = run.url
    run.finish(exit_code=0)

    receipt = {
        "entity": args.entity,
        "project": args.project,
        "run_id": args.run_id,
        "run_name": args.run_name,
        "run_url": run_url,
        "logging_mode": "historical_tensorboard_import",
        "metric_tag_count": len(scalars),
        "optimizer_steps": observed_steps,
        "resource_sample_count": len(resource_events),
        "evidence_sha256": {name: sha256(path) for name, path in required.items()},
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
