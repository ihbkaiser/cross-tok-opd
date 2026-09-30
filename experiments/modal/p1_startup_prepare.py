from __future__ import annotations
import datetime as dt, json, os, subprocess, traceback
from pathlib import Path
import modal

LOCAL_ROOT = Path("/home/tung/simct-b200-portable")
IMAGE_REF = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME = "simct-qwen7b-gemma2-assets-20260916"
PREP_VOLUME = "simct-p1-startup-prep-20260916"
APP_NAME = "simct-p1-startup-prep-20260916"

image = (modal.Image.from_registry(IMAGE_REF).entrypoint([])
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/modal"), "/opt/repo/experiments/modal", copy=True))
app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=False)
prepvol = modal.Volume.from_name(PREP_VOLUME, create_if_missing=True)

@app.function(image=image, cpu=2, memory=8192, timeout=2400, retries=0,
              volumes={"/assets": assets, "/prep": prepvol})
def prepare_remote(commit: str, snapshot_id: str = "assets-20260916") -> dict:
    import sys
    sys.path.insert(0, "/opt/repo/experiments/modal")
    from provenance_prepare import prepare, write_receipt
    events = []
    roots = {"student": "/assets/student", "teacher": "/assets/teacher",
             "dataset": "/assets/prompts.parquet", "meta": "/assets/selected.parquet",
             "energy": "/assets/energy-select-4.pt"}
    try:
        receipt = prepare(roots=roots, source_root="/opt/repo/kdflow", source_commit=commit,
                          snapshot_id=snapshot_id, progress=events.append)
        stored = write_receipt("/prep/startup-provenance.json", receipt)
        prepvol.commit()
        return {"status": "ready", "receipt_sha256": stored["receipt_sha256"],
                "counts": stored["counts"], "events": events,
                "prepared_seconds": stored["prepared_seconds"],
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    except BaseException as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                "events": events, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}

@app.local_entrypoint()
def main():
    commit = subprocess.check_output(["git", "-C", str(LOCAL_ROOT), "rev-parse", "HEAD"], text=True).strip()
    result = prepare_remote.remote(commit)
    out = Path("/mnt/d/dev/codex/research_vdt/remote_artifacts/p1-startup-prep-20260916")
    out.mkdir(parents=True, exist_ok=True)
    (out / "prep.receipt.json").write_text(json.dumps({"source_commit": commit, "image": IMAGE_REF, "volume": PREP_VOLUME, **result}, indent=2) + "\n")
    print("P1_PREP_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") != "ready": raise SystemExit(1)
