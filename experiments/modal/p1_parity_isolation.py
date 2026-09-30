"""P1 parity isolation — offline first-divergence and sampler/serving audit.

CPU-only by default. The paid rollout-only A/A lives behind its own gate and is
never triggered implicitly. Reads r5 artifacts read-only; writes new receipts
under /runs/p1-parity-20260917/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import modal

LOCAL_ROOT = Path("/home/tung/simct-b200-portable")
REMOTE_ROOT = Path("/opt/repo")
IMAGE_REF = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME = "simct-qwen7b-gemma2-assets-20260916"
RUN_VOLUME = "simct-qwen7b-gemma2-runs-20260916-main"
APP_NAME = "simct-p1-parity-isolation-20260917"
OUT_DIR = "/runs/p1-parity-20260917"

RUN_TAG = "p1-hostmask-ab-20260916-r5"
CONTROL = f"/runs/{RUN_TAG}-control"
CANDIDATE = f"/runs/{RUN_TAG}-candidate"
RESUME_ATTEMPT3 = f"/runs/{RUN_TAG}-resume-control-attempt3"

image = (
    modal.Image.from_registry(IMAGE_REF)
    .entrypoint([])
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/runai"), "/opt/repo/experiments/runai", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "experiments/modal/vendor"),
                   "/opt/repo/experiments/modal/vendor", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "tests"), "/opt/repo/tests", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/mp_opd_phi_gemma_50.py"),
                    "/opt/repo/experiments/modal/mp_opd_phi_gemma_50.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/provenance_prepare.py"),
                    "/opt/repo/experiments/modal/provenance_prepare.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/_sampler_audit_probe.py"),
                    "/opt/repo/experiments/modal/_sampler_audit_probe.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/_rollout_aa_probe.py"),
                    "/opt/repo/experiments/modal/_rollout_aa_probe.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/_triton_audit_probe.py"),
                    "/opt/repo/experiments/modal/_triton_audit_probe.py", copy=True)
    .add_local_file(str(LOCAL_ROOT / "experiments/modal/p1_parity_isolation.py"),
                    "/opt/repo/experiments/modal/p1_parity_isolation.py", copy=True)
)
app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME, create_if_missing=False)
runs = modal.Volume.from_name(RUN_VOLUME, create_if_missing=False)
prepvol = modal.Volume.from_name("simct-p1-startup-prep-20260916", create_if_missing=False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value):
    """Make NaN/Inf safe for JSON without pretending they are numbers."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(sanitize(value), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def bounded(value, limit: int = 24):
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, str):
        return value[:limit]
    return value


def compare_sequence(left, right, *, numeric: bool = False) -> dict:
    report: dict = {
        "left_present": left is not None,
        "right_present": right is not None,
    }
    if left is None or right is None:
        report["equal"] = left is None and right is None
        return report
    report["length_left"] = len(left)
    report["length_right"] = len(right)
    shared = min(len(left), len(right))
    first_diff = None
    max_abs = 0.0
    nonfinite_left = nonfinite_right = 0
    for index in range(shared):
        x, y = left[index], right[index]
        if numeric:
            try:
                fx, fy = float(x), float(y)
            except (TypeError, ValueError):
                fx = fy = None
            if fx is not None:
                if not math.isfinite(fx):
                    nonfinite_left += 1
                if not math.isfinite(fy):
                    nonfinite_right += 1
                if math.isfinite(fx) and math.isfinite(fy):
                    max_abs = max(max_abs, abs(fx - fy))
        if x != y and first_diff is None:
            first_diff = index
    report["first_diff"] = first_diff
    report["common_prefix"] = shared if first_diff is None else first_diff
    report["equal"] = (
        first_diff is None and len(left) == len(right) and len(left) <= shared
    )
    if first_diff is not None:
        report["left_at_first_diff"] = bounded(left[first_diff], 8)
        report["right_at_first_diff"] = bounded(right[first_diff], 8)
    if numeric:
        report["max_abs_diff_on_prefix"] = max_abs
        report["nonfinite_left"] = nonfinite_left
        report["nonfinite_right"] = nonfinite_right
    return report


def compare_text(left, right) -> dict:
    if left is None or right is None:
        return {"left_present": left is not None, "right_present": right is not None,
                "equal": left == right}
    if left == right:
        return {"equal": True, "length_left": len(left), "length_right": len(right)}
    shared = min(len(left), len(right))
    first = next((i for i in range(shared) if left[i] != right[i]), shared)
    return {
        "equal": False,
        "length_left": len(left),
        "length_right": len(right),
        "first_diff": first,
        "common_prefix": first,
        "left_sha256": hashlib.sha256(left.encode()).hexdigest(),
        "right_sha256": hashlib.sha256(right.encode()).hexdigest(),
    }


def is_logprob_payload(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, (list, tuple)) and row for row in value)
    )


def logprob_series(value) -> list:
    series = []
    for row in value:
        head = row[0]
        series.append(float(head) if isinstance(head, (int, float)) else None)
    return series


def compare_meta(left_meta: dict, right_meta: dict) -> dict:
    left_keys, right_keys = set(left_meta), set(right_meta)
    report: dict = {
        "keys_only_left": sorted(left_keys - right_keys),
        "keys_only_right": sorted(right_keys - left_keys),
        "logprob_fields": {},
        "scalar_mismatch": [],
        "other_mismatch": [],
    }
    for key in sorted(left_keys & right_keys):
        a, b = left_meta[key], right_meta[key]
        if is_logprob_payload(a) and is_logprob_payload(b):
            report["logprob_fields"][key] = compare_sequence(
                logprob_series(a), logprob_series(b), numeric=True
            )
        elif isinstance(a, (int, float, str, bool)) or a is None:
            if a != b:
                report["scalar_mismatch"].append({"field": key, "left": a, "right": b})
        elif a != b:
            report["other_mismatch"].append({"field": key, "left_type": type(a).__name__,
                                             "right_type": type(b).__name__})
    return report


def row_report(left_row: dict, right_row: dict) -> dict:
    left_meta = left_row.get("meta_info") or {}
    right_meta = right_row.get("meta_info") or {}
    prompt_ids = compare_sequence(left_row.get("prompt_ids"), right_row.get("prompt_ids"))
    prompt_text = compare_text(left_row.get("prompt"), right_row.get("prompt"))
    output_ids = compare_sequence(left_row.get("output_ids"), right_row.get("output_ids"))
    output_text = compare_text(left_row.get("output"), right_row.get("output"))
    meta = compare_meta(left_meta, right_meta)
    logprob_bad = sorted(
        name for name, info in meta["logprob_fields"].items() if not info.get("equal")
    )
    weight_left = left_row.get("behavior_weight_version")
    weight_right = right_row.get("behavior_weight_version")

    kind = "identical"
    if not prompt_ids.get("equal") or not prompt_text.get("equal"):
        kind = "request_identity"
    elif not output_ids.get("equal"):
        kind = "output_token"
    elif logprob_bad:
        kind = "logprob_numeric"
    elif meta["scalar_mismatch"] or meta["keys_only_left"] or meta["keys_only_right"]:
        kind = "metadata"
    elif meta["other_mismatch"]:
        kind = "metadata_other"
    elif not output_text.get("equal"):
        kind = "detokenized_text"
    elif weight_left != weight_right:
        kind = "behavior_weight_version"

    return {
        "kind": kind,
        "prompt_ids": prompt_ids,
        "prompt_text": prompt_text,
        "output_ids": output_ids,
        "output_text": output_text,
        "meta": meta,
        "logprob_fields_mismatched": logprob_bad,
        "behavior_weight_version": {"left": weight_left, "right": weight_right},
        "meta_has_sampling_seed": ("sampling_seed" in left_meta) or ("sampling_seed" in right_meta),
        "meta_has_seed": ("seed" in left_meta) or ("seed" in right_meta),
    }


def read_rows(path: Path) -> list:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def permutation_report(left_rows: list, right_rows: list) -> dict:
    def fingerprint(ids):
        return hashlib.sha256(
            json.dumps(ids, sort_keys=True).encode()
        ).hexdigest()

    left_ids = [row.get("prompt_ids") for row in left_rows]
    right_ids = [row.get("prompt_ids") for row in right_rows]
    left_sorted = sorted(fingerprint(i) for i in left_ids)
    right_sorted = sorted(fingerprint(i) for i in right_ids)
    return {
        "ordered_stream_equal": left_ids == right_ids,
        "multiset_equal": left_sorted == right_sorted,
        "rows_left": len(left_ids),
        "rows_right": len(right_ids),
    }


def request_seed(train_seed: int, step: int, index: int) -> int:
    digest = hashlib.sha256(f"kdflow-rollout-v1:{train_seed}:{step}:{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2 ** 31)


def compare_pair(label: str, left_root: str, right_root: str, file_name: str) -> dict:
    left_path = Path(left_root) / "checkpoint/rollout_data" / file_name
    right_path = Path(right_root) / "checkpoint/rollout_data" / file_name
    report: dict = {
        "label": label,
        "left_root": left_root,
        "right_root": right_root,
        "file": file_name,
        "left_path": str(left_path),
        "right_path": str(right_path),
    }
    if not left_path.is_file() or not right_path.is_file():
        report["status"] = "missing_artifact"
        report["left_present"] = left_path.is_file()
        report["right_present"] = right_path.is_file()
        return report
    left_rows, right_rows = read_rows(left_path), read_rows(right_path)
    report.update({
        "status": "compared",
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
        "rows_left": len(left_rows),
        "rows_right": len(right_rows),
        "permutation": permutation_report(left_rows, right_rows),
    })
    step = int(Path(file_name).stem)
    rows = []
    kind_counts: dict = {}
    first_divergence = None
    shared = min(len(left_rows), len(right_rows))
    for index in range(shared):
        entry = row_report(left_rows[index], right_rows[index])
        kind_counts[entry["kind"]] = kind_counts.get(entry["kind"], 0) + 1
        if entry["kind"] != "identical":
            rows.append({
                "row_index": index,
                "kind": entry["kind"],
                "recomputed_sampling_seed": request_seed(42, step, index),
                "prompt_ids_len": entry["prompt_ids"].get("length_left"),
                "prompt_ids_first_diff": entry["prompt_ids"].get("first_diff"),
                "output_ids_len": entry["output_ids"].get("length_left"),
                "output_ids_first_diff": entry["output_ids"].get("first_diff"),
                "output_ids_prefix_equal": entry["output_ids"].get("common_prefix"),
                "logprob_fields_mismatched": entry["logprob_fields_mismatched"],
                "logprob_max_abs_diff": {
                    name: entry["meta"]["logprob_fields"][name].get("max_abs_diff_on_prefix")
                    for name in entry["logprob_fields_mismatched"]
                },
                "scalar_mismatch": entry["meta"]["scalar_mismatch"],
                "keys_only_left": entry["meta"]["keys_only_left"],
                "keys_only_right": entry["meta"]["keys_only_right"],
                "behavior_weight_version": entry["behavior_weight_version"],
                "prompt_text": entry["prompt_text"],
                "output_text_equal": entry["output_text"].get("equal"),
                "meta_has_sampling_seed": entry["meta_has_sampling_seed"],
            })
            if first_divergence is None:
                first_divergence = {
                    "row_index": index,
                    "kind": entry["kind"],
                    "recomputed_sampling_seed": request_seed(42, step, index),
                    "prompt_ids_first_diff": entry["prompt_ids"].get("first_diff"),
                    "output_ids_first_diff": entry["output_ids"].get("first_diff"),
                    "logprob_fields_mismatched": entry["logprob_fields_mismatched"],
                    "scalar_mismatch": entry["meta"]["scalar_mismatch"],
                    "keys_only_left": entry["meta"]["keys_only_left"],
                    "keys_only_right": entry["meta"]["keys_only_right"],
                    "prompt_ids_left_len": entry["prompt_ids"].get("length_left"),
                    "prompt_ids_right_len": entry["prompt_ids"].get("length_right"),
                    "output_ids_left_len": entry["output_ids"].get("length_left"),
                    "output_ids_right_len": entry["output_ids"].get("length_right"),
                    "output_ids_common_prefix": entry["output_ids"].get("common_prefix"),
                }
    report["kind_counts"] = kind_counts
    report["first_divergence"] = first_divergence
    report["divergent_rows"] = rows[:16]
    report["divergent_rows_truncated"] = max(0, len(rows) - 16)
    report["both_identical"] = first_divergence is None and len(left_rows) == len(right_rows)
    # Row 0 meta schema is enough to prove whether a request seed field exists at all.
    if left_rows:
        report["meta_keys_row0_left"] = sorted((left_rows[0].get("meta_info") or {}).keys())
    if right_rows:
        report["meta_keys_row0_right"] = sorted((right_rows[0].get("meta_info") or {}).keys())
    return report


# --------------------------------------------------------------------------- #
# CPU gate 1: first divergence
# --------------------------------------------------------------------------- #
PAIRS = (
    ("ab_step1", CONTROL, CANDIDATE, "1.jsonl"),
    ("ab_step2", CONTROL, CANDIDATE, "2.jsonl"),
    ("resume_vs_continuous_step2", CONTROL, RESUME_ATTEMPT3, "2.jsonl"),
)


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/runs": runs})
def first_divergence_remote() -> dict:
    result: dict = {"status": "completed", "pairs": [], "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        for label, left, right, name in PAIRS:
            print(f"DIVERGENCE_STAGE_START {label} pid={os.getpid()}", flush=True)
            result["pairs"].append(compare_pair(label, left, right, name))
            print(f"DIVERGENCE_STAGE_END {label} pid={os.getpid()}", flush=True)
        atomic_json(Path(OUT_DIR) / "FIRST_DIVERGENCE.json", result)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        runs.commit()
    return result


# --------------------------------------------------------------------------- #
# CPU gate 2: sampler / serving audit in the pinned image
# --------------------------------------------------------------------------- #
@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/runs": runs})
def sampler_audit_remote() -> dict:
    """Persist the audit to the runs volume so a dead client cannot lose it."""
    py = "/opt/venvs/simct-b200/bin/python"
    probe = "/opt/repo/experiments/modal/_sampler_audit_probe.py"
    proc = subprocess.run([py, probe], cwd=str(REMOTE_ROOT), text=True, capture_output=True)
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SAMPLER_AUDIT_JSON="):
            payload = json.loads(line.split("=", 1)[1])
    result = {
        "status": "completed" if proc.returncode == 0 and payload else "failed",
        "returncode": proc.returncode,
        "report": payload,
        "stderr_tail": (proc.stderr or "")[-4000:],
        "executable": py,
    }
    atomic_json(Path(OUT_DIR) / "sampler-audit.json", result)
    runs.commit()
    print("SAMPLER_AUDIT_STATUS=" + result["status"], flush=True)
    print("SAMPLER_AUDIT_JSON=" + json.dumps(payload, sort_keys=True)[:20000], flush=True)
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def manifest_supplement_remote(predecessor_sha: str) -> dict:
    """64-row request manifest + asset identity lifted from the existing prep receipt."""
    import sys

    sys.path.insert(0, "/opt/repo")
    from kdflow.trajectory import bounded_sampling_params
    from kdflow.training_checkpoint import seeded_sampling

    source = Path("/runs/p1-hostmask-ab-20260916-r5-control/checkpoint/rollout_data/1.jsonl")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    prompt_ids = [record["prompt_ids"] for record in records]
    params = seeded_sampling(
        bounded_sampling_params(
            prompt_ids,
            {"max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95},
            4096,
        ),
        seed=42, step=1, count=len(prompt_ids),
    )
    prep_path = Path("/prep/startup-provenance.json")
    prep = json.loads(prep_path.read_text()) if prep_path.is_file() else {}
    result = {
        "status": "completed",
        "predecessor_sha256": predecessor_sha,
        "source": str(source),
        "source_sha256": sha256(source),
        "rows": len(prompt_ids),
        "manifest": [
            {
                "original_index": index,
                "prompt_len": len(ids),
                "prompt_sha256": hashlib.sha256(
                    ",".join(str(item) for item in ids).encode()).hexdigest(),
                "sampling_seed": params[index]["sampling_seed"],
                "max_new_tokens": params[index]["max_new_tokens"],
                "temperature": params[index]["temperature"],
                "top_p": params[index]["top_p"],
            }
            for index, ids in enumerate(prompt_ids)
        ],
        "prep_receipt": prep,
        "prep_receipt_sha256": sha256(prep_path) if prep_path.is_file() else None,
    }
    atomic_json(Path(OUT_DIR) / "reference-manifest-supplement.json", result)
    runs.commit()
    print("MANIFEST_SUPPLEMENT_ROWS=" + str(result["rows"]), flush=True)
    return result


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/assets": assets, "/runs": runs})
def triton_audit_remote() -> dict:
    """CPU-only audit of installed Triton backend + Gemma2 semantics."""
    py = "/opt/venvs/simct-b200/bin/python"
    probe = "/opt/repo/experiments/modal/_triton_audit_probe.py"
    proc = subprocess.run([py, probe], cwd=str(REMOTE_ROOT), text=True, capture_output=True)
    payload = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("TRITON_AUDIT_JSON="):
            payload = json.loads(line.split("=", 1)[1])
    result = {"status": "completed" if proc.returncode == 0 and payload else "failed",
              "returncode": proc.returncode, "report": payload,
              "stderr_tail": (proc.stderr or "")[-3000:]}
    atomic_json(Path(OUT_DIR) / "triton-audit.json", result)
    runs.commit()
    print("TRITON_AUDIT_STATUS=" + result["status"], flush=True)
    return result



# --------------------------------------------------------------------------- #
# GPU gate 4: reference2 + resume1 under the deterministic Triton contract
# --------------------------------------------------------------------------- #
TRAIN_TAG = "p1-triton-training-20260917"
REFERENCE_ROOT = f"/runs/{TRAIN_TAG}-reference"
RESUME_ROOT = f"/runs/{TRAIN_TAG}-resume"


EVERY4_TAG = "p1-triton-every4-20260917"
EVERY4_CONTINUOUS = f"/runs/{EVERY4_TAG}-continuous"
EVERY4_PAUSED = f"/runs/{EVERY4_TAG}-paused"


def training_environment(run_root: str, commit: str, prepared_sha: str,
                         *, energy_every: str = "1") -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update({
        "PATH": "/opt/venvs/simct-b200/bin:/usr/local/cuda-13.0/bin:" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": "/opt/repo/experiments/modal/vendor:/opt/repo",
        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "KDFLOW_TRUST_REMOTE_CODE": "0",
        "RAY_USAGE_STATS_ENABLED": "0", "NCCL_CUMEM_HOST_ENABLE": "0", "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": "/usr/local/cuda-13.0",
        "CPATH": "/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include",
        "LD_LIBRARY_PATH": "/tmp/runtime-host-libs:/opt/simct-portable-libs:/usr/local/cuda-13.0/lib64",
        "MP_RUNTIME_DIR": "/tmp/runtime", "MP_RAY_TMP": "/tmp/ray",
        "XDG_CACHE_HOME": "/tmp/cache", "TRITON_CACHE_DIR": "/tmp/cache/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/cache/torch", "FLASHINFER_WORKSPACE_BASE": "/tmp/cache",
        "KDFLOW_ROLLOUT_PORT_BASE": "15000", "KDFLOW_ROUTER_PORT_BASE": "16000",
        "KDFLOW_ROUTER_PROMETHEUS_PORT": "20000",
        "MP_SHARED_ROOT": "/assets", "MP_STUDENT_PATH": "/assets/student",
        "MP_TEACHER_PATH": "/assets/teacher", "MP_DATASET_PATH": "/assets/prompts.parquet",
        "MP_META_PATH": "/assets/selected.parquet", "MP_ENERGY_CHECKPOINT": "/assets/energy-select-4.pt",
        "MP_ALGORITHM": "mp_opd", "MP_ALTERNATING": "1", "MP_OFFLOAD_ADAM_MOMENTS": "1",
        "MP_MICRO_TRAIN_BATCH_SIZE": "1", "MP_META_MICRO_BATCH_SIZE": "4",
        "MP_SEED": "42", "MP_PARTITION_SEED": "43",
        "MP_MAX_SPAN_LENGTH": "2", "MP_FIXED_SPAN_LENGTH": "2",
        "MP_ATTN_IMPLEMENTATION": "eager", "MP_CHECKPOINT_STEPS": "1,2",
        "MP_SOURCE_COMMIT": commit, "MP_SOURCE_DIRTY": "",
        "MP_QUALIFICATION_POLICY": "p1-triton-training-pair",
        "MP_QUALIFICATION_STATUS": "diagnostic",
        "MP_ENERGY_LR": "0.001", "MP_ENERGY_EVERY": energy_every,
        "MP_RUN_ROOT": run_root, "MP_OPD_TIMING": "1", "MP_OPD_HOST_MASK": "0",
        "MP_PREPARED_RECEIPT": "/prep/startup-provenance.json",
        "MP_PREPARED_RECEIPT_SHA256": prepared_sha,
        "MP_SNAPSHOT_ID": "assets-20260916",
        "WANDB_MODE": "offline", "WANDB_DISABLED": "true",
        # Deterministic Triton serving contract, pinned for reference AND resume.
        "MP_ROLLOUT_DETERMINISTIC": "1", "MP_ROLLOUT_SEED": "42",
        "MP_ROLLOUT_ATTENTION_BACKEND": "triton", "MP_ROLLOUT_DISABLE_RADIX_CACHE": "1",
    })
    return env


def _worker_dirs() -> None:
    for directory in ("/tmp/runtime/runtime-host-libs", "/tmp/ray", "/tmp/cache/triton",
                      "/tmp/cache/torch"):
        Path(directory).mkdir(parents=True, exist_ok=True)


def _run_worker(run_dir: Path, env: dict, log_path: Path, timeout: int) -> int:
    cmd = ["bash", "/opt/repo/experiments/runai/python-b200-host.sh",
           "/opt/repo/experiments/runai/run_single_gpu.py", "soft", "2", str(run_dir)]
    started = time.time()
    with log_path.open("w") as sink:
        sink.write(f"WORKER_START run_dir={run_dir} pid=pending resume={env.get('MP_RESUME', '0')}\n")
        sink.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    with log_path.open("a") as sink:
        sink.write(f"WORKER_END rc={rc} seconds={time.time() - started:.3f}\n")
    return rc


def _checkpoint_digests(root: Path) -> dict:
    pointer = root / "checkpoints/latest.json"
    if not pointer.is_file():
        return {}
    latest = json.loads(pointer.read_text())
    manifest = root / "checkpoints" / latest["directory"] / "manifest.json"
    return {
        "latest": latest,
        "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "directory": latest["directory"],
        "rollout_files": sorted(p.name for p in (root / "checkpoint/rollout_data").glob("*.jsonl")),
    }


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=5400, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def training_reference_remote(commit: str, prepared_sha: str) -> dict:
    """Fresh 2-update reference under the deterministic Triton serving contract."""
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    from kdflow.run_counters import classify_terminal

    _worker_dirs()
    run_dir = Path(REFERENCE_ROOT)
    result: dict = {"schema": "simct-p1-training-reference-v1", "arm": "reference",
                    "source_commit": commit, "run_dir": str(run_dir), "status": "starting"}
    try:
        if run_dir.exists():
            raise RuntimeError(f"reference run dir already exists: {run_dir}")
        env = training_environment(str(run_dir), commit, prepared_sha)
        log_path = Path(OUT_DIR) / "training-reference.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rc = _run_worker(run_dir, env, log_path, timeout=4200)
        result["child_exit"] = rc
        summary_path = run_dir / "checkpoint" / "run-summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        result["summary_present"] = summary is not None
        result["verdict"] = classify_terminal(app_state="completed", child_exit=rc, summary=summary,
                                              expected_start=0, expected_total=2,
                                              expected_session_delta=2)
        launch = run_dir / "launch-config.json"
        if launch.is_file():
            manifest = json.loads(launch.read_text())
            result["serving"] = manifest.get("serving")
            result["launch_config_sha256"] = sha256(launch)
        result["checkpoints"] = _checkpoint_digests(run_dir)
        result["log_path"] = str(log_path)
        result["status"] = "completed" if result["verdict"]["verdict"] == "pass" else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(OUT_DIR) / "training-reference.receipt.json", result)
        runs.commit()
    print("TRAINING_REFERENCE_STATUS=" + str(result.get("status")), flush=True)
    print("TRAINING_REFERENCE_VERDICT=" + json.dumps(result.get("verdict"), default=str), flush=True)
    return result


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=5400, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def training_resume_remote(commit: str, prepared_sha: str) -> dict:
    """Resume from the reference step1 transaction in a fresh process, one update."""
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    from kdflow.run_counters import classify_terminal

    _worker_dirs()
    source = Path(REFERENCE_ROOT)
    run_dir = Path(RESUME_ROOT)
    result: dict = {"schema": "simct-p1-training-resume-v1", "arm": "resume",
                    "source_commit": commit, "run_dir": str(run_dir),
                    "source_transaction": str(source / "checkpoints"), "status": "starting"}
    try:
        if run_dir.exists():
            raise RuntimeError(f"resume run dir already exists: {run_dir}")
        steps = sorted((source / "checkpoints").glob("step00000001-*"))
        if len(steps) != 1:
            raise RuntimeError(f"expected exactly one step1 transaction, found {len(steps)}")
        destination = run_dir / "checkpoints" / steps[0].name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(steps[0], destination)
        atomic_json(run_dir / "checkpoints/latest.json", {
            "directory": destination.name,
            "manifest_sha256": sha256(destination / "manifest.json"),
        })
        previous = json.loads((source / "launch-config.json").read_text())
        options = previous.get("options", {})
        for key in ("save_path", "ckpt_path"):
            if key in options:
                options[key] = str(options[key]).replace(Path(REFERENCE_ROOT).name,
                                                         Path(RESUME_ROOT).name)
        atomic_json(run_dir / "launch-config.json", previous)
        result["restored_manifest_sha256"] = sha256(destination / "manifest.json")
        env = training_environment(str(run_dir), commit, prepared_sha)
        env.update({"MP_RESUME": "1", "MP_PAUSE_AFTER_UPDATES": "0"})
        log_path = Path(OUT_DIR) / "training-resume.log"
        rc = _run_worker(run_dir, env, log_path, timeout=3600)
        result["child_exit"] = rc
        summary_path = run_dir / "checkpoint" / "run-summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        result["summary_present"] = summary is not None
        result["verdict"] = classify_terminal(app_state="completed", child_exit=rc, summary=summary,
                                              expected_start=1, expected_total=2,
                                              expected_session_delta=1)
        result["checkpoints"] = _checkpoint_digests(run_dir)
        result["log_path"] = str(log_path)
        result["status"] = "completed" if result["verdict"]["verdict"] == "pass" else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(OUT_DIR) / "training-resume.receipt.json", result)
        runs.commit()
    print("TRAINING_RESUME_STATUS=" + str(result.get("status")), flush=True)
    print("TRAINING_RESUME_VERDICT=" + json.dumps(result.get("verdict"), default=str), flush=True)
    return result


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/runs": runs})
def checkpoint_inspect_remote(step: int) -> dict:
    """CPU-only transactional verification of one reference checkpoint step."""
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    from kdflow.training_checkpoint import inspect

    root = Path(REFERENCE_ROOT) / "checkpoints"
    out: dict = {"status": "starting", "step": step, "root": str(root)}
    try:
        folders = sorted(root.glob(f"step{step:08d}-*"))
        if len(folders) != 1:
            raise ValueError(f"expected one step{step} transaction, found {len(folders)}")
        manifest_path = folders[0] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        print(f"CHECKPOINT_STAGE_START step={step} pid={os.getpid()}", flush=True)
        checked, value = inspect(root, manifest["contract"], manifest["world_size"])
        out.update({
            "status": "verified",
            "directory": folders[0].name,
            "manifest_sha256": sha256(manifest_path),
            "world_size": value["world_size"],
            "inspected_directory": str(checked),
            "files": sorted(manifest.get("files", {}).keys()),
            "contract_schema": manifest.get("contract", {}).get("schema"),
        })
        print(f"CHECKPOINT_STAGE_END step={step} pid={os.getpid()}", flush=True)
    except BaseException as exc:
        out.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                   traceback=traceback.format_exc()[-2000:])
    finally:
        atomic_json(Path(OUT_DIR) / f"checkpoint-inspect-step{step}.json", out)
        runs.commit()
    print("CHECKPOINT_INSPECT_STATUS=" + str(out.get("status")), flush=True)
    return out


@app.function(image=image, cpu=16, memory=65536, timeout=2400, retries=0, volumes={"/runs": runs})
def training_compare_remote() -> dict:
    """Compare reference step2 with resumed step2: state tensors AND trajectories.

    The 1+1 diagnostic legitimately has different rollout file sets (the resume
    session only produces its update-2 rollout), so the production file-set gate
    is not reused here; state and policy evidence are compared directly.
    """
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    import numpy as np
    import torch
    from kdflow.training_checkpoint import inspect

    reference = Path(REFERENCE_ROOT)
    resumed = Path(RESUME_ROOT)
    result: dict = {"schema": "simct-p1-training-compare-v1", "status": "starting",
                    "reference": str(reference), "resumed": str(resumed)}

    def load(root: Path):
        pointer = json.loads((root / "checkpoints/latest.json").read_text())
        folder = root / "checkpoints" / pointer["directory"]
        manifest = json.loads((folder / "manifest.json").read_text())
        folder, manifest = inspect(root / "checkpoints", manifest["contract"], manifest["world_size"])
        return (folder, manifest,
                torch.load(folder / "rank0.pt", map_location="cpu", weights_only=False),
                torch.load(folder / "driver.pt", map_location="cpu", weights_only=False))

    def equal(x, y, path, problems):
        if torch.is_tensor(x):
            if not torch.equal(x, y):
                problems.append(path)
        elif isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                problems.append(path)
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                problems.append(path + "/keys")
                return
            for key in x:
                equal(x[key], y[key], f"{path}/{key}", problems)
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                problems.append(path + "/length")
                return
            for index, (u, v) in enumerate(zip(x, y)):
                equal(u, v, f"{path}/{index}", problems)
        elif x != y:
            problems.append(path)

    def trajectory(path: Path) -> list:
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            info = row.pop("meta_info", {})
            row["behavior_logprobs"] = {k: v for k, v in info.items() if "logprob" in k}
            row["finish_reason"] = info.get("finish_reason")
            rows.append(row)
        return rows

    try:
        print(f"COMPARE_STAGE_START reference_resumed_step2 pid={os.getpid()}", flush=True)
        ref_folder, ref_manifest, ref_actor, ref_driver = load(reference)
        res_folder, res_manifest, res_actor, res_driver = load(resumed)
        result["reference_checkpoint"] = {"directory": ref_folder.name,
                                          "manifest_sha256": sha256(ref_folder / "manifest.json")}
        result["resumed_checkpoint"] = {"directory": res_folder.name,
                                        "manifest_sha256": sha256(res_folder / "manifest.json")}
        problems: list = []
        for driver in (ref_driver, res_driver):
            driver.pop("resource_sample_index", None)
        equal(ref_actor, res_actor, "actor", problems)
        equal(ref_driver, res_driver, "driver", problems)
        result["state_status"] = "exact_match" if not problems else "mismatch"
        result["state_problems"] = problems[:20]
        result["state_problem_count"] = len(problems)
        result["state_keys_compared"] = sorted(ref_actor.keys())[:8]

        left = reference / "checkpoint/rollout_data/2.jsonl"
        right = resumed / "checkpoint/rollout_data/2.jsonl"
        traj_problems: list = []
        left_rows, right_rows = trajectory(left), trajectory(right)
        if len(left_rows) != len(right_rows):
            traj_problems.append(f"row_count:{len(left_rows)}!={len(right_rows)}")
        for index, (a, b) in enumerate(zip(left_rows, right_rows)):
            if a.get("prompt_ids") != b.get("prompt_ids"):
                traj_problems.append(f"row{index}/prompt_ids")
            if a.get("output_ids") != b.get("output_ids"):
                traj_problems.append(f"row{index}/output_ids")
                continue
            if a.get("behavior_logprobs") != b.get("behavior_logprobs"):
                traj_problems.append(f"row{index}/behavior_logprobs")
            if a.get("finish_reason") != b.get("finish_reason"):
                traj_problems.append(f"row{index}/finish_reason")
        result["trajectory_status"] = "exact_match" if not traj_problems else "mismatch"
        result["trajectory_problems"] = traj_problems[:20]
        result["trajectory_problem_count"] = len(traj_problems)
        result["trajectory_rows"] = len(left_rows)
        result["reference_counters"] = json.loads((reference / "checkpoint/run-summary.json").read_text())
        result["resumed_counters"] = json.loads((resumed / "checkpoint/run-summary.json").read_text())
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        atomic_json(Path(OUT_DIR) / "training-compare.receipt.json", result)
        runs.commit()
    print("TRAINING_COMPARE_STATUS=" + str(result.get("status")), flush=True)
    print("TRAINING_COMPARE_STATE=" + str(result.get("state_status")), flush=True)
    print("TRAINING_COMPARE_TRAJECTORY=" + str(result.get("trajectory_status")), flush=True)
    return result


# --------------------------------------------------------------------------- #
# GPU gate 5: EVERY4 qualification package (continuous4 vs pause3+resume1)
# --------------------------------------------------------------------------- #
def _every4_worker(run_dir, commit, prepared_sha, *, updates: int, pause_after: int,
                   resume: bool, log_name: str) -> tuple:
    env = training_environment(str(run_dir), commit, prepared_sha, energy_every="4")
    env["MP_PAUSE_AFTER_UPDATES"] = str(pause_after)
    if resume:
        env["MP_RESUME"] = "1"
    log_path = Path(OUT_DIR) / log_name
    started = time.time()
    cmd = ["bash", "/opt/repo/experiments/runai/python-b200-host.sh",
           "/opt/repo/experiments/runai/run_single_gpu.py", "soft", str(updates), str(run_dir)]
    with log_path.open("w") as sink:
        sink.write(f"WORKER_START arm={log_name} resume={resume} pause_after={pause_after}\n")
        sink.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(REMOTE_ROOT), env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, text=True, timeout=4200)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    with log_path.open("a") as sink:
        sink.write(f"WORKER_END rc={rc} seconds={time.time() - started:.3f}\n")
    return rc, log_path


def _every4_receipt(arm: str, run_dir: Path, rc: int, log_path: Path,
                    *, expected_start: int, expected_total: int,
                    expected_delta: int, expected_energy: int) -> dict:
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    from kdflow.run_counters import classify_terminal
    from kdflow.step_timing import parse_step_records, summarize

    summary_path = run_dir / "checkpoint" / "run-summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    verdict = classify_terminal(app_state="completed", child_exit=rc, summary=summary,
                                expected_start=expected_start, expected_total=expected_total,
                                expected_session_delta=expected_delta)
    if summary is not None:
        energy = summary.get("energy_updates_total", summary.get("energy_updates"))
        if energy is None or int(energy) != expected_energy:
            verdict["reasons"].append(f"energy={energy}!={expected_energy}")
            verdict["verdict"] = "fail"
    text = log_path.read_text() if log_path.is_file() else ""
    timing = summarize(parse_step_records(text), 4,
                       resumed_step=(expected_delta if expected_start else None))
    launch = run_dir / "launch-config.json"
    return {
        "arm": arm, "run_dir": str(run_dir), "child_exit": rc,
        "summary_present": summary is not None, "verdict": verdict,
        "energy_updates_total": (summary or {}).get("energy_updates_total"),
        "timing": timing,
        "checkpoints": _checkpoint_digests(run_dir),
        "serving": (json.loads(launch.read_text()).get("serving") if launch.is_file() else None),
        "launch_config_sha256": sha256(launch) if launch.is_file() else None,
        "log_path": str(log_path),
    }


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=5400, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def every4_continuous_remote(commit: str, prepared_sha: str) -> dict:
    """fresh -> update1..4 with energy_every=4, no pause; energy updates == 1."""
    _worker_dirs()
    run_dir = Path(EVERY4_CONTINUOUS)
    result: dict = {"schema": "simct-p1-every4-continuous-v1", "status": "starting",
                    "run_dir": str(run_dir), "source_commit": commit}
    try:
        if run_dir.exists():
            raise RuntimeError(f"run dir already exists: {run_dir}")
        rc, log_path = _every4_worker(run_dir, commit, prepared_sha, updates=4, pause_after=0,
                                      resume=False, log_name="every4-continuous.log")
        result.update(_every4_receipt("every4-continuous", run_dir, rc, log_path,
                                      expected_start=0, expected_total=4,
                                      expected_delta=4, expected_energy=1))
        result["status"] = "completed" if result["verdict"]["verdict"] == "pass" else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(OUT_DIR) / "every4-continuous.receipt.json", result)
        runs.commit()
    print("EVERY4_CONTINUOUS_STATUS=" + str(result.get("status")), flush=True)
    return result


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=5400, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def every4_pause_remote(commit: str, prepared_sha: str) -> dict:
    """fresh -> update1..3 then checkpoint_pause; energy updates == 0."""
    _worker_dirs()
    run_dir = Path(EVERY4_PAUSED)
    result: dict = {"schema": "simct-p1-every4-pause-v1", "status": "starting",
                    "run_dir": str(run_dir), "source_commit": commit}
    try:
        if run_dir.exists():
            raise RuntimeError(f"run dir already exists: {run_dir}")
        rc, log_path = _every4_worker(run_dir, commit, prepared_sha, updates=4, pause_after=3,
                                      resume=False, log_name="every4-pause.log")
        result.update(_every4_receipt("every4-paused", run_dir, rc, log_path,
                                      expected_start=0, expected_total=3,
                                      expected_delta=3, expected_energy=0))
        result["status"] = "completed" if result["verdict"]["verdict"] == "pass" else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(OUT_DIR) / "every4-pause.receipt.json", result)
        runs.commit()
    print("EVERY4_PAUSE_STATUS=" + str(result.get("status")), flush=True)
    return result


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=5400, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs, "/prep": prepvol})
def every4_resume_remote(commit: str, prepared_sha: str) -> dict:
    """resume the paused root from its step3 transaction; student delta == 1."""
    _worker_dirs()
    run_dir = Path(EVERY4_PAUSED)
    result: dict = {"schema": "simct-p1-every4-resume-v1", "status": "starting",
                    "run_dir": str(run_dir), "source_commit": commit}
    try:
        steps = sorted((run_dir / "checkpoints").glob("step00000003-*"))
        if len(steps) != 1:
            raise RuntimeError(f"expected one paused step3 transaction, found {len(steps)}")
        result["restored_manifest_sha256"] = sha256(steps[0] / "manifest.json")
        result["restored_directory"] = steps[0].name
        rc, log_path = _every4_worker(run_dir, commit, prepared_sha, updates=4, pause_after=0,
                                      resume=True, log_name="every4-resume.log")
        result.update(_every4_receipt("every4-resume", run_dir, rc, log_path,
                                      expected_start=3, expected_total=4,
                                      expected_delta=1, expected_energy=1))
        result["status"] = "completed" if result["verdict"]["verdict"] == "pass" else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(Path(OUT_DIR) / "every4-resume.receipt.json", result)
        runs.commit()
    print("EVERY4_RESUME_STATUS=" + str(result.get("status")), flush=True)
    return result


@app.function(image=image, cpu=16, memory=65536, timeout=2400, retries=0, volumes={"/runs": runs})
def every4_compare_remote() -> dict:
    """Compare both EVERY4 paths: step3 (pause) and step4 (resume) versus continuous."""
    import sys

    if "/opt/repo" not in sys.path:
        sys.path.insert(0, "/opt/repo")
    import numpy as np
    import torch
    from kdflow.training_checkpoint import inspect

    continuous = Path(EVERY4_CONTINUOUS)
    paused = Path(EVERY4_PAUSED)
    result: dict = {"schema": "simct-p1-every4-compare-v1", "status": "starting",
                    "continuous": str(continuous), "paused": str(paused)}

    def load_step(root: Path, step: int):
        folders = sorted((root / "checkpoints").glob(f"step{step:08d}-*"))
        if len(folders) != 1:
            raise ValueError(f"expected one step{step} transaction in {root}, found {len(folders)}")
        folder, manifest = folders[0], json.loads((folders[0] / "manifest.json").read_text())
        folder, manifest = inspect(root / "checkpoints", manifest["contract"], manifest["world_size"])
        return (folder, manifest,
                torch.load(folder / "rank0.pt", map_location="cpu", weights_only=False),
                torch.load(folder / "driver.pt", map_location="cpu", weights_only=False))

    def equal(x, y, path, problems):
        if torch.is_tensor(x):
            if not torch.equal(x, y):
                problems.append(path)
        elif isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                problems.append(path)
        elif isinstance(x, dict):
            if x.keys() != y.keys():
                problems.append(path + "/keys")
                return
            for key in x:
                equal(x[key], y[key], f"{path}/{key}", problems)
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y):
                problems.append(path + "/length")
                return
            for index, (u, v) in enumerate(zip(x, y)):
                equal(u, v, f"{path}/{index}", problems)
        elif x != y:
            problems.append(path)

    def trajectory(path: Path) -> list:
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            info = row.pop("meta_info", {})
            row["behavior_logprobs"] = {k: v for k, v in info.items() if "logprob" in k}
            row["finish_reason"] = info.get("finish_reason")
            rows.append(row)
        return rows

    def compare(step: int, rollout_names: list) -> dict:
        left = load_step(continuous, step)
        right = load_step(paused, step)
        problems: list = []
        for driver in (left[3], right[3]):
            driver.pop("resource_sample_index", None)
        equal(left[2], right[2], "actor", problems)
        equal(left[3], right[3], "driver", problems)
        traj_problems: list = []
        for name in rollout_names:
            left_rows = trajectory(continuous / "checkpoint/rollout_data" / name)
            right_rows = trajectory(paused / "checkpoint/rollout_data" / name)
            if len(left_rows) != len(right_rows):
                traj_problems.append(f"{name}/row_count:{len(left_rows)}!={len(right_rows)}")
                continue
            for index, (a, b) in enumerate(zip(left_rows, right_rows)):
                for field in ("prompt_ids", "output_ids", "behavior_logprobs", "finish_reason"):
                    if a.get(field) != b.get(field):
                        traj_problems.append(f"{name}/row{index}/{field}")
        return {
            "step": step,
            "rollout_files": rollout_names,
            "state_status": "exact_match" if not problems else "mismatch",
            "state_problems": problems[:20],
            "state_problem_count": len(problems),
            "trajectory_status": "exact_match" if not traj_problems else "mismatch",
            "trajectory_problems": traj_problems[:20],
            "trajectory_problem_count": len(traj_problems),
            "continuous_manifest_sha256": sha256(left[0] / "manifest.json"),
            "paused_manifest_sha256": sha256(right[0] / "manifest.json"),
        }

    try:
        print(f"EVERY4_COMPARE_START pid={os.getpid()}", flush=True)
        def has_step(root: Path, step: int) -> bool:
            return len(sorted((root / "checkpoints").glob(f"step{step:08d}-*"))) == 1

        if has_step(continuous, 3) and has_step(paused, 3):
            result["step3_pause_vs_continuous"] = compare(3, ["1.jsonl", "2.jsonl", "3.jsonl"])
        else:
            result["step3_pause_vs_continuous"] = {
                "status": "not_available",
                "reason": "specimen did not persist a step3 transaction in both lineages; "
                          "the pinned MP_CHECKPOINT_STEPS saves 1,2 plus the final/latest transaction",
            }
        if has_step(continuous, 4) and has_step(paused, 4):
            result["step4_resume_vs_continuous"] = compare(4, ["4.jsonl"])
        else:
            result["step4_resume_vs_continuous"] = {
                "status": "not_run",
                "reason": "resume segment has not produced a step4 transaction yet",
            }
        result["continuous_counters"] = json.loads(
            (continuous / "checkpoint/run-summary.json").read_text())
        result["paused_counters"] = json.loads((paused / "checkpoint/run-summary.json").read_text())
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-3000:])
    finally:
        atomic_json(Path(OUT_DIR) / "every4-compare.receipt.json", result)
        runs.commit()
    print("EVERY4_COMPARE_STATUS=" + str(result.get("status")), flush=True)
    return result


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0,
              volumes={"/runs": runs})
def pinned_tests_remote(paths: list) -> dict:
    """Run CPU-eligible regression with the pinned interpreter inside the pinned image."""
    py = "/opt/venvs/simct-b200/bin/python"
    env = dict(os.environ, PYTHONPATH="/opt/repo/experiments/modal/vendor:/opt/repo")
    proc = subprocess.run(
        [py, "-m", "pytest", "-q", *paths, "--disable-warnings", "-p", "no:cacheprovider"],
        cwd=str(REMOTE_ROOT), env=env, text=True, capture_output=True,
    )
    result = {
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "executable": py,
        "tests": list(paths),
        "output_tail": (proc.stdout + proc.stderr)[-9000:],
    }
    atomic_json(Path(OUT_DIR) / "pinned-tests.json", result)
    runs.commit()
    print("PINNED_TESTS_STATUS=" + result["status"], flush=True)
    print(result["output_tail"][-5000:], flush=True)
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0)
def list_dir_remote(path: str) -> dict:
    """Diagnostic: prove what the deployed image actually contains."""
    root = Path(path)
    if not root.exists():
        return {"path": path, "exists": False}
    entries = sorted(p.as_posix() for p in root.rglob("*"))
    return {"path": path, "exists": True, "count": len(entries), "entries": entries[:400],
            "tests_mp_opd": sorted(
                p.as_posix() for p in (Path("/opt/repo/tests/mp_opd").rglob("*")
                                       if Path("/opt/repo/tests/mp_opd").exists() else [])
            )[:80]}


@app.local_entrypoint()
def main() -> None:
    gate = os.environ.get("P1_GATE", "divergence")
    if gate == "spawn":
        # Fire-and-forget: the client only has to live for the spawn call, so a
        # dropped heartbeat cannot lose the result. Receipts land on the volume.
        target = os.environ.get("P1_SPAWN_FN", "sampler_audit")
        if target == "every4_continuous":
            call = every4_continuous_remote.spawn(os.environ.get("P1_COMMIT", ""),
                                                  os.environ.get("P1_PREPARED_SHA", ""))
        elif target == "every4_pause":
            call = every4_pause_remote.spawn(os.environ.get("P1_COMMIT", ""),
                                             os.environ.get("P1_PREPARED_SHA", ""))
        elif target == "every4_resume":
            call = every4_resume_remote.spawn(os.environ.get("P1_COMMIT", ""),
                                              os.environ.get("P1_PREPARED_SHA", ""))
        elif target == "every4_compare":
            call = every4_compare_remote.spawn()
        elif target == "checkpoint_inspect":
            call = checkpoint_inspect_remote.spawn(int(os.environ.get("P1_STEP", "1")))
        elif target == "training_reference":
            call = training_reference_remote.spawn(os.environ.get("P1_COMMIT", ""),
                                                   os.environ.get("P1_PREPARED_SHA", ""))
        elif target == "training_resume":
            call = training_resume_remote.spawn(os.environ.get("P1_COMMIT", ""),
                                                os.environ.get("P1_PREPARED_SHA", ""))
        elif target == "training_compare":
            call = training_compare_remote.spawn()
        elif target == "manifest_supplement":
            call = manifest_supplement_remote.spawn(os.environ.get("P1_PREDECESSOR_SHA", ""))
        elif target == "triton_audit":
            call = triton_audit_remote.spawn()
        elif target == "pinned_tests":
            paths = [item for item in os.environ.get(
                "P1_TEST_PATHS",
                "/opt/repo/tests/mp_opd/test_rollout_deterministic_args.py,/opt/repo/tests/test_runai_contract.py",
            ).split(",") if item]
            call = pinned_tests_remote.spawn(paths)
        elif target == "sampler_audit":
            call = sampler_audit_remote.spawn()
        elif target == "divergence":
            call = first_divergence_remote.spawn()
        elif target == "rollout_aa":
            call = rollout_aa_remote.spawn(os.environ.get("P1_AA_MODE", "baseline"))
        else:
            raise SystemExit("unknown P1_SPAWN_FN=" + target)
        print("SPAWNED_CALL_ID=" + str(call.object_id), flush=True)
        print("SPAWNED_FN=" + target, flush=True)
        return
    out_dir = Path("/mnt/d/dev/codex/research_vdt/remote_artifacts/p1-parity-20260917")
    out_dir.mkdir(parents=True, exist_ok=True)
    if gate == "divergence":
        result = first_divergence_remote.remote()
        atomic_json(out_dir / "first-divergence.receipt.json", result)
        print("FIRST_DIVERGENCE_STATUS=" + str(result.get("status")), flush=True)
        for pair in result.get("pairs", []):
            print(f"PAIR {pair['label']} status={pair.get('status')} first={pair.get('first_divergence')}", flush=True)
        return
    if gate == "rollout_aa":
        mode = os.environ.get("P1_AA_MODE", "baseline")
        result = rollout_aa_remote.remote(mode)
        atomic_json(out_dir / f"rollout-aa-{mode}.receipt.json", result)
        print("ROLLOUT_AA_GATE_STATUS=" + str(result.get("status")), flush=True)
        print("ROLLOUT_AA_GATE_MARKER=" + json.dumps(sanitize(result.get("marker")), sort_keys=True)[:4000], flush=True)
        return
    if gate == "checkpoint_inspect":
        result = checkpoint_inspect_remote.remote(int(os.environ.get("P1_STEP", "1")))
        atomic_json(out_dir / f"checkpoint-inspect-step{os.environ.get('P1_STEP', '1')}.receipt.json", result)
        print("CHECKPOINT_INSPECT_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "every4_continuous":
        result = every4_continuous_remote.remote(os.environ.get("P1_COMMIT", ""),
                                                 os.environ.get("P1_PREPARED_SHA", ""))
        atomic_json(out_dir / "every4-continuous.receipt.json", result)
        print("EVERY4_CONTINUOUS_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "every4_pause":
        result = every4_pause_remote.remote(os.environ.get("P1_COMMIT", ""),
                                            os.environ.get("P1_PREPARED_SHA", ""))
        atomic_json(out_dir / "every4-pause.receipt.json", result)
        print("EVERY4_PAUSE_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "every4_resume":
        result = every4_resume_remote.remote(os.environ.get("P1_COMMIT", ""),
                                             os.environ.get("P1_PREPARED_SHA", ""))
        atomic_json(out_dir / "every4-resume.receipt.json", result)
        print("EVERY4_RESUME_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "every4_compare":
        result = every4_compare_remote.remote()
        atomic_json(out_dir / "every4-compare.receipt.json", result)
        print("EVERY4_COMPARE_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "training_reference":
        result = training_reference_remote.remote(os.environ.get("P1_COMMIT", ""),
                                                  os.environ.get("P1_PREPARED_SHA", ""))
        atomic_json(out_dir / "training-reference.receipt.json", result)
        print("TRAINING_REFERENCE_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "training_resume":
        result = training_resume_remote.remote(os.environ.get("P1_COMMIT", ""),
                                               os.environ.get("P1_PREPARED_SHA", ""))
        atomic_json(out_dir / "training-resume.receipt.json", result)
        print("TRAINING_RESUME_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "training_compare":
        result = training_compare_remote.remote()
        atomic_json(out_dir / "training-compare.receipt.json", result)
        print("TRAINING_COMPARE_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "manifest_supplement":
        result = manifest_supplement_remote.remote(os.environ.get("P1_PREDECESSOR_SHA", ""))
        atomic_json(out_dir / "reference-manifest-supplement.receipt.json", result)
        print("MANIFEST_SUPPLEMENT_ROWS=" + str(result.get("rows")), flush=True)
        return
    if gate == "triton_audit":
        result = triton_audit_remote.remote()
        atomic_json(out_dir / "triton-audit.receipt.json", result)
        print("TRITON_AUDIT_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "pinned_tests":
        paths = [item for item in os.environ.get(
            "P1_TEST_PATHS",
            "/opt/repo/tests/mp_opd/test_rollout_deterministic_args.py,"
            "/opt/repo/tests/mp_opd/test_logprob_probe.py,"
            "/opt/repo/tests/mp_opd/test_run_counters.py,"
            "/opt/repo/tests/test_runai_contract.py",
        ).split(",") if item]
        result = pinned_tests_remote.remote(paths)
        atomic_json(out_dir / "pinned-tests.receipt.json", result)
        print("PINNED_TESTS_GATE=" + str(result.get("status")), flush=True)
        return
    if gate == "sampler_audit":
        result = sampler_audit_remote.remote()
        atomic_json(out_dir / "sampler-audit.receipt.json", result)
        print("SAMPLER_AUDIT_STATUS=" + str(result.get("status")), flush=True)
        return
    raise SystemExit("unknown P1_GATE=" + gate)
# --------------------------------------------------------------------------- #
# GPU gate 3: rollout-only A/A through the production RolloutActorGroup path
# --------------------------------------------------------------------------- #
DRIVER = "/opt/repo/experiments/modal/_rollout_aa_probe.py"


def driver_environment(mode: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update({
        "PATH": "/opt/venvs/simct-b200/bin:/usr/local/cuda-13.0/bin:" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": "/opt/repo/experiments/modal/vendor:/opt/repo",
        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "KDFLOW_TRUST_REMOTE_CODE": "0",
        "RAY_USAGE_STATS_ENABLED": "0", "NCCL_CUMEM_HOST_ENABLE": "0", "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_HOME": "/usr/local/cuda-13.0",
        "CPATH": "/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include",
        "LD_LIBRARY_PATH": "/tmp/runtime-host-libs:/opt/simct-portable-libs:/usr/local/cuda-13.0/lib64",
        "MP_RUNTIME_DIR": "/tmp/runtime", "MP_RAY_TMP": "/tmp/ray",
        "XDG_CACHE_HOME": "/tmp/cache", "TRITON_CACHE_DIR": "/tmp/cache/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/cache/torch", "FLASHINFER_WORKSPACE_BASE": "/tmp/cache",
        "KDFLOW_ROLLOUT_PORT_BASE": "15000", "KDFLOW_ROUTER_PORT_BASE": "16000",
        "MP_SHARED_ROOT": "/assets", "MP_STUDENT_PATH": "/assets/student",
        "P1_AA_MODE": mode,
        "KDFLOW_PAYLOAD_DUMP": f"{OUT_DIR}/wire-payload-{mode}.jsonl",
        "P1_AA_ROWS": os.environ.get("P1_AA_ROWS", "64"),
        "P1_AA_REPLAYS": os.environ.get("P1_AA_REPLAYS", "2"),
        "P1_AA_RESTART": os.environ.get("P1_AA_RESTART", "1"),
        "WANDB_MODE": "offline", "WANDB_DISABLED": "true",
    })
    return env


@app.function(image=image, gpu="B200", cpu=12, memory=65536, ephemeral_disk=524288,
              timeout=3600, retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": assets, "/runs": runs})
def rollout_aa_remote(mode: str) -> dict:
    """One arm of the rollout-only A/A. No teacher, no full-meta, no optimizer."""
    for directory in ("/tmp/runtime/runtime-host-libs", "/tmp/ray", "/tmp/cache/triton",
                      "/tmp/cache/torch"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"rollout-aa-{mode}.log"
    result: dict = {"status": "starting", "mode": mode, "driver": DRIVER,
                    "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        env = driver_environment(mode)
        started = time.time()
        with log_path.open("w") as sink:
            sink.write(f"ROLLOUT_AA_STAGE_START mode={mode} rows={env['P1_AA_ROWS']} "
                       f"replays={env['P1_AA_REPLAYS']} restart={env['P1_AA_RESTART']}\n")
            sink.flush()
            proc = subprocess.run(["/opt/venvs/simct-b200/bin/python", DRIVER],
                                  cwd=str(REMOTE_ROOT), env=env, stdout=sink,
                                  stderr=subprocess.STDOUT, text=True, timeout=3300)
        result["driver_seconds"] = time.time() - started
        result["returncode"] = proc.returncode
        tail = log_path.read_text()[-20000:]
        marker = None
        for line in tail.splitlines():
            if line.startswith("ROLLOUT_AA_JSON="):
                marker = json.loads(line.split("=", 1)[1])
        result["marker"] = marker
        result["log_path"] = str(log_path)
        result["log_tail"] = tail[-6000:]
        payload_path = Path(OUT_DIR) / f"wire-payload-{mode}.jsonl"
        if payload_path.is_file():
            samples = [json.loads(line) for line in payload_path.read_text().splitlines() if line.strip()]
            result["wire_payloads"] = samples[:4]
            result["wire_payload_count"] = len(samples)
        v2_path = Path(OUT_DIR) / f"rollout-aa-{mode}-v2.json"
        if v2_path.is_file():
            v2 = json.loads(v2_path.read_text())
            result["stage_summaries"] = [
                {"stage": stage.get("stage"), "comparison": stage.get("comparison")}
                for stage in v2.get("stages", [])
            ]
            result["probe_verdicts"] = [
                {"tag": probe.get("tag"), "verdict": (probe.get("verdict") or {}).get("status"),
                 "reason": (probe.get("verdict") or {}).get("reason"),
                 "entries": probe.get("entries"), "fingerprint": probe.get("fingerprint")}
                for probe in v2.get("probes", [])
            ]
            result["backend_consistency"] = v2.get("backend_consistency")
            result["stopped_at"] = v2.get("stopped_at_first_failed_stage")
        result["status"] = "completed" if proc.returncode == 0 and marker else "failed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(out_dir / f"rollout-aa-{mode}.receipt.json", result)
        runs.commit()
    print("ROLLOUT_AA_STATUS=" + str(result.get("status")), flush=True)
    print("ROLLOUT_AA_LOG=" + str(result.get("log_path")), flush=True)
    return result