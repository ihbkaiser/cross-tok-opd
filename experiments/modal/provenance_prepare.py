"""CPU-side startup provenance preparation and strict receipt validation."""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from pathlib import Path

SCHEMA = "simct-p1-startup-provenance-v1"
MODEL_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}

class PreparationError(ValueError):
    pass

def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _files(root: Path, *, model_view: bool = False):
    root = root.resolve()
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and (not model_view or p.suffix in MODEL_SUFFIXES)]

def _entries(root: Path, *, model_view: bool = False, progress=None, label=""):
    root = root.resolve(); files = _files(root, model_view=model_view)
    if not files: raise PreparationError(f"empty provenance input: {root}")
    result = {}; total = 0
    for i, path in enumerate(files, 1):
        size = path.stat().st_size
        result[path.name if root.is_file() else str(path.relative_to(root))] = {
            "bytes": size, "sha256": _sha256_bytes(path)}
        total += size
        if progress:
            progress({"event": "read", "label": label, "index": i,
                      "files": len(files), "path": str(path), "bytes": size,
                      "total_bytes": total})
    return result

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _view_digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()

def prepare(*, roots: dict, source_root: str | None = None,
            source_commit: str | None = None, snapshot_id: str | None = None,
            progress=None, interpreter: str | None = None):
    started = time.time(); root_map = {str(k): str(v) for k, v in roots.items()}
    launcher_models = {}; pipeline_inputs = {}; counts = {"files": 0, "bytes": 0}
    for role, raw in root_map.items():
        entries = _entries(Path(raw), model_view=role in {"student", "teacher"},
                           progress=progress, label=role)
        if role in {"student", "teacher"}: launcher_models[role] = entries
        pipeline_inputs[raw] = entries
        counts["files"] += len(entries); counts["bytes"] += sum(x["bytes"] for x in entries.values())
    source = {}
    if source_root:
        source = {k: v for k, v in _entries(Path(source_root), progress=progress, label="source").items()
                  if k.endswith(".py")}
        if not source: raise PreparationError("empty source .py provenance input")
    return {"schema": SCHEMA, "status": "ready", "source_commit": source_commit,
            "snapshot_id": snapshot_id, "roots": root_map,
            "launcher_models": launcher_models, "pipeline_inputs": pipeline_inputs,
            "source": source, "launcher_models_sha256": _view_digest(launcher_models),
            "pipeline_inputs_sha256": _view_digest(pipeline_inputs), "source_sha256": _view_digest(source),
            "counts": counts, "prepared_seconds": time.time() - started,
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "interpreter": interpreter or sys.executable,
                        "source_root": str(Path(source_root).resolve()) if source_root else None}}

def write_receipt(path, receipt):
    path = Path(path).resolve()
    for raw in receipt.get("roots", {}).values():
        root = Path(raw).resolve()
        if path == root or root in path.parents: raise PreparationError("receipt sidecar must be outside hashed roots")
    body = dict(receipt); body.pop("receipt_sha256", None)
    body["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n"); os.replace(tmp, path)
    return body

def load_receipt(path, *, expected_sha256=None, expected_source_commit=None,
                 expected_snapshot_id=None, require_ready=True, verify_files=True):
    try: receipt = json.loads(Path(path).read_text())
    except Exception as exc: raise PreparationError(f"invalid provenance receipt: {path}") from exc
    if receipt.get("schema") != SCHEMA or (require_ready and receipt.get("status") != "ready"):
        raise PreparationError("partial or unsupported provenance receipt")
    body = dict(receipt); got = body.pop("receipt_sha256", None)
    if not got or got != hashlib.sha256(_canonical(body)).hexdigest(): raise PreparationError("provenance receipt checksum mismatch")
    if expected_sha256 and got != expected_sha256: raise PreparationError("provenance receipt digest mismatch")
    if expected_source_commit is not None and receipt.get("source_commit") != expected_source_commit: raise PreparationError("provenance source snapshot mismatch")
    if expected_snapshot_id is not None and receipt.get("snapshot_id") != expected_snapshot_id: raise PreparationError("provenance snapshot mismatch")
    if not isinstance(receipt.get("roots"), dict) or not receipt["roots"]: raise PreparationError("missing provenance roots")
    for key in ("launcher_models", "pipeline_inputs", "source"):
        digest_key = key + "_sha256"
        if receipt.get(digest_key) != _view_digest(receipt.get(key, {})): raise PreparationError(f"{key} coverage digest mismatch")
    if verify_files:
        for role, raw in receipt["roots"].items():
            expected = receipt["launcher_models"].get(role) if role in {"student", "teacher"} else receipt["pipeline_inputs"].get(raw)
            actual = _entries(Path(raw), model_view=role in {"student", "teacher"})
            if expected != actual: raise PreparationError(f"provenance content changed for {role}")
        if receipt.get("source"):
            actual = {k: v for k, v in _entries(Path(receipt["runtime"].get("source_root", ""))).items() if k.endswith(".py")} if receipt["runtime"].get("source_root") else None
            if actual is not None and actual != receipt["source"]: raise PreparationError("source provenance content changed")
    return receipt

def prepared_entries(receipt, role, raw_path):
    if role in {"student", "teacher"}:
        values = receipt.get("launcher_models", {}).get(role)
    else:
        values = receipt.get("pipeline_inputs", {}).get(str(raw_path))
        if values is None:
            values = receipt.get("pipeline_inputs", {}).get(str(Path(raw_path).resolve()))
    if values is None: raise PreparationError(f"missing provenance coverage for {role or raw_path}")
    return values
