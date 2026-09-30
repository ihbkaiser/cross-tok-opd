"""Transactional, algorithm-neutral training checkpoints (not HF weight exports).

Only trusted local checkpoints may be loaded: the payload includes Python RNG
and optimizer state. A manifest is published after every rank finishes writing.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import importlib.metadata
import math

import numpy as np
import torch

SCHEMA = "kdflow-training-v1"


def normalize_config(value):
    """Encode disabled cadence infinities as strict JSON, never nonstandard NaN."""
    if isinstance(value,dict):return {k:normalize_config(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize_config(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        if math.isnan(value):raise ValueError('NaN configuration')
        return 'Infinity' if value>0 else '-Infinity'
    return value


def pipeline_contract(args, dataset_size):
    from dataclasses import asdict
    config = asdict(args)
    # JSON-normalize tuples before comparing against a decoded manifest.
    config = normalize_config(config)
    for key in ("load_checkpoint", "pause_after_updates", "save_path", "ckpt_path", "local_rank"):
        config["train"].pop(key, None)
    config.pop("log", None)
    prepared = None
    receipt_path = os.environ.get("MP_PREPARED_RECEIPT")
    if receipt_path:
        from experiments.modal.provenance_prepare import load_receipt, prepared_entries, PreparationError
        try:
            prepared = load_receipt(receipt_path,
                expected_sha256=os.environ.get("MP_PREPARED_RECEIPT_SHA256") or None,
                expected_source_commit=os.environ.get("MP_SOURCE_COMMIT") or None,
                expected_snapshot_id=os.environ.get("MP_SNAPSHOT_ID") or None, verify_files=False)
        except PreparationError as exc:
            raise ValueError("Prepared provenance cache miss; refusing pipeline rehash: " + str(exc)) from exc
    source = Path(__file__).parent
    files = ({str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))}
             if prepared is None else prepared["source"])
    paths = [args.data.train_dataset_path, args.model.student_name_or_path,
             args.model.teacher_name_or_path]
    for name in ("mp_opd_energy_checkpoint", "mp_opd_meta_path"):
        value = getattr(args.kd, name, None)
        if value:
            paths.append(value)
    data = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise ValueError("Reliable resume requires pinned local inputs: " + raw)
        if prepared is not None:
            entries = prepared_entries(prepared, "dataset" if raw == args.data.train_dataset_path else ("student" if raw == args.model.student_name_or_path else ("teacher" if raw == args.model.teacher_name_or_path else "energy")), raw)
            data[raw] = {key: value["sha256"] for key, value in entries.items()}
            continue
        selected = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        if not selected:
            raise ValueError("Empty provenance input: " + raw)
        data[raw] = {str(p.relative_to(path)) if path.is_dir() else path.name: digest(p) for p in selected}
    versions = {}
    for name in ("torch", "transformers", "sglang", "ray", "torchdata"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"config": config, "dataset_size": dataset_size, "inputs": data,
            "source": files, "versions": versions, "rollout_rng": "request-seed-v1"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG topology differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    with temp.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    sync_dir(path.parent)


def sync_dir(path):
    if os.name == "posix":
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_payload(path, payload):
    path = Path(path)
    # Unique transaction directory plus exclusive files; never overwrite a
    # checkpoint a reader or an evaluator could already be using.
    with path.open("xb") as f:
        torch.save(payload, f)
        f.flush(); os.fsync(f.fileno())
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def begin(root, step):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / (f"step{step:08d}-" + uuid.uuid4().hex)
    dest.mkdir()
    return dest


def publish(root, directory, *, step, world_size, contract, client, rank_files):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if directory.parent != root or len(rank_files) != world_size:
        raise ValueError("Invalid checkpoint transaction")
    files = {f"rank{i}.pt": info for i, info in enumerate(rank_files)}
    for name, info in files.items():
        p = directory / name
        if p.stat().st_size != info["bytes"] or digest(p) != info["sha256"]:
            raise ValueError("Incomplete rank checkpoint: " + name)
    files["driver.pt"] = write_payload(directory / "driver.pt", client)
    manifest = {"schema": SCHEMA, "step": step, "world_size": world_size,
                "contract": contract, "files": files}
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(root / "latest.json", {"directory": directory.name,
                "manifest_sha256": digest(directory / "manifest.json")})
    return directory


def inspect(root, contract, world_size):
    root = Path(root).resolve()
    pointer = json.loads((root / "latest.json").read_text())
    directory = (root / pointer["directory"]).resolve()
    if directory.parent != root:
        raise ValueError("Checkpoint path escapes root")
    if digest(directory / "manifest.json") != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest checksum mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["world_size"] != world_size:
        raise ValueError("Unsupported checkpoint schema/topology")
    if manifest["contract"] != contract:
        raise ValueError("Resume contract changed (config/data/source/runtime)")
    expected = {"driver.pt"} | {f"rank{i}.pt" for i in range(world_size)}
    if set(manifest["files"]) != expected:
        raise ValueError("Incomplete checkpoint manifest")
    for name, info in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError("Checkpoint payload checksum mismatch: " + name)
    return directory, manifest


def prune_complete(root, keep=2):
    """Retain newest complete recovery snapshots; never touch partial evidence/HF exports."""
    import shutil
    root = Path(root).resolve()
    if keep < 2:
        raise ValueError("Keep at least two recovery checkpoints")
    latest = json.loads((root / "latest.json").read_text())["directory"]
    complete = []
    for p in root.glob("step*-*/manifest.json"):
        if p.parent.is_symlink() or p.parent.resolve().parent != root:
            continue
        value = json.loads(p.read_text())
        if value.get("schema") == SCHEMA:
            complete.append((value["step"], p.stat().st_mtime_ns, p.parent))
    ordered = sorted(complete, reverse=True)
    retained = {p for _, _, p in ordered[:keep]} | {root / latest}
    for _, _, p in ordered:
        if p not in retained:
            # Only remove payloads generated by this module's transaction format.
            names = {f"rank{i}.pt" for i in range(json.loads((p/'manifest.json').read_text())["world_size"])}
            if {f.name for f in p.iterdir()} != names | {"driver.pt", "manifest.json"}:
                continue
            shutil.rmtree(p)


def seeded_sampling(params, *, seed, step, count):
    """Stateless request RNG; does not consume driver/student RNG streams."""
    rows = params if isinstance(params, list) else [params] * count
    if len(rows) != count:
        raise ValueError("Sampling cardinality mismatch")
    return [dict(p, sampling_seed=int.from_bytes(hashlib.sha256(
        f"kdflow-rollout-v1:{seed}:{step}:{i}".encode()).digest()[:4], "big") % (2**31))
        for i, p in enumerate(rows)]


def state_bundle(student, algorithm, optimizer):
    """Name every trainable module/optimizer, including algorithm projectors.

    Algorithms with additional mutable non-module state implement
    training_state_dict/load_training_state_dict; no config-name special cases.
    """
    bundle = torch.nn.Module()
    bundle.add_module("student", student)
    optimizers = [optimizer]
    excluded = {id(student)}
    for key in ("student", "teacher_lm_head"):
        value = getattr(algorithm, key, None)
        if isinstance(value, torch.nn.Module):
            excluded.update(id(m) for m in value.modules())
    for name, value in sorted(vars(algorithm).items()):
        if isinstance(value, torch.nn.Module) and id(value) not in excluded:
            bundle.add_module("algorithm_" + name, value)
            excluded.update(id(m) for m in value.modules())
        elif isinstance(value, torch.optim.Optimizer) and value not in optimizers:
            optimizers.append(value)
    known = {id(p) for p in bundle.parameters()}
    owned = [id(p) for o in optimizers for g in o.param_groups for p in g["params"]]
    if set(owned) - known or len(set(owned)) != len(owned):
        raise ValueError("Checkpoint optimizer ownership is missing or ambiguous")
    return bundle, optimizers


def snapshot_training(student, algorithm, optimizer, scheduler, *, accumulation_step):
    if accumulation_step != 0 or any(p.grad is not None for p in student.parameters()):
        raise ValueError("Training checkpoint requires a completed optimizer boundary")
    from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
    bundle, optimizers = state_bundle(student, algorithm, optimizer)
    names = {id(p): n for n, p in bundle.named_parameters()}
    lazy = [(o, p) for o in optimizers for g in o.param_groups for p in g["params"] if not o.state.get(p)]
    # DCP initializes an untouched optimizer using a dummy step. Preserve laziness
    # explicitly: otherwise every4 resumes with Adam's time index advanced by 1.
    try:
        model_state, optim_state = get_state_dict(bundle, optimizers,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    finally:
        for o, p in lazy:
            o.state.pop(p, None)
            p.grad = None
    lazy_names = [names[id(p)] for _, p in lazy]
    # Retain DCP's synthetic entries for its strict FQN loader, but explicitly
    # remove them from the live optimizer again after load, using lazy_names.
    return {"model": model_state, "optimizers": optim_state,
            "scheduler": scheduler.state_dict(), "rng": capture_rng(),
            "algorithm": algorithm.training_state_dict()
                if hasattr(algorithm, "training_state_dict") else {},
            "module_names": list(bundle._modules), "optimizer_count": len(optimizers),
            "lazy_optimizer_params": lazy_names}


def restore_training(payload, student, algorithm, optimizer, scheduler):
    from torch.distributed.checkpoint.state_dict import set_state_dict, StateDictOptions
    bundle, optimizers = state_bundle(student, algorithm, optimizer)
    if payload["module_names"] != list(bundle._modules) or payload["optimizer_count"] != len(optimizers):
        raise ValueError("Training component layout differs")
    # All ranks load the rank-zero full tensors, with their own rank RNG below.
    set_state_dict(bundle, optimizers, model_state_dict=payload["model"],
                   optim_state_dict=payload["optimizers"],
                   options=StateDictOptions(full_state_dict=True, strict=True))
    scheduler.load_state_dict(payload["scheduler"])
    lazy = set(payload["lazy_optimizer_params"])
    names = {id(p): n for n,p in bundle.named_parameters()}
    for o in optimizers:
        for group in o.param_groups:
            for p in group["params"]:
                if names[id(p)] in lazy:
                    o.state.pop(p, None)
    if hasattr(algorithm, "load_training_state_dict"):
        algorithm.load_training_state_dict(payload["algorithm"])
    elif payload["algorithm"]:
        raise ValueError("Algorithm cannot restore its saved training state")
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    restore_rng(payload["rng"])
