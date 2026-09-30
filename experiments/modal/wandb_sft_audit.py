"""Read-only W&B audit for the B200 SFT qualification run."""

from __future__ import annotations

import json
import math

import modal


APP_NAME = "vdt-simct-b200-sft-wandb-audit-no1ceboy-20260904-r6"
RUN_PATH = (
    "kieusontung8-hanoi-university-of-science-and-technology/"
    "vdt-simct-tunix-reproduction/simct-b200-sft-r6-9d5c801"
)

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install("wandb==0.29.0")
wandb_secret = modal.Secret.from_name("vdt-xtoken-wandb-no1ceboy")


@app.function(image=image, secrets=[wandb_secret], timeout=300, retries=0)
def audit() -> dict[str, object]:
    import wandb

    run = wandb.Api(timeout=60).run(RUN_PATH)
    rows: list[dict[str, object]] = []
    numeric_finite = True
    for row in run.scan_history(page_size=100):
        kept = {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and isinstance(value, (int, float, str, bool))
        }
        for value in kept.values():
            if isinstance(value, float) and not math.isfinite(value):
                numeric_finite = False
        if kept:
            rows.append(kept)

    system_rows = run.history(samples=500, stream="system", pandas=False)
    system_keys = sorted(
        {
            key
            for row in system_rows
            for key, value in row.items()
            if not key.startswith("_") and isinstance(value, (int, float))
        }
    )
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "history_rows": rows,
        "history_row_count": len(rows),
        "all_numeric_finite": numeric_finite,
        "system_row_count": len(system_rows),
        "system_keys": system_keys,
    }


@app.local_entrypoint()
def main() -> None:
    result = audit.remote()
    print("WANDB_AUDIT_JSON=" + json.dumps(result, sort_keys=True))
    if result["state"] != "finished" or not result["all_numeric_finite"]:
        raise SystemExit(1)
