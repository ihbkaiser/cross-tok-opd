"""Sampler/serving audit probe — runs inside the pinned image with the pinned interpreter.

CPU only: never initializes a model or a CUDA context. Reports installed SGLang
source identities, whether the request-level seed key survives into
SamplingParams, and the sampling code path that consumes it. Normalized
ServerArgs are dumped by the GPU-side driver, where accelerator detection works.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import json
import sys
from pathlib import Path

NEEDLES = (
    "enable_deterministic",
    "sampling_seed",
    "random_seed",
    "multinomial",
    "sampling_backend",
    "deterministic",
)
SOURCE_FILES = (
    "sampling/sampling_batch_info.py",
    "sampling/sampling_params.py",
    "layers/sampler.py",
    "server_args.py",
    "managers/schedule_batch.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(path: Path, needles=NEEDLES, window=4) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    hits = set()
    defs: dict = {}
    current = "<module>"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def "):
            current = stripped.split("(")[0].replace("def ", "").replace("class ", "").replace("async ", "")
        if any(needle in line for needle in needles):
            hits.update(range(max(0, index - window), min(len(lines), index + window + 1)))
            defs.setdefault(current, []).append(index + 1)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "lines": len(lines),
        "occurrences_by_symbol": defs,
        "excerpt": [f"{index + 1}: {lines[index]}" for index in sorted(hits)],
    }


def main() -> int:
    report: dict = {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        report["torch_error"] = repr(exc)

    import sglang

    report["sglang_version"] = md.version("sglang")
    package_root = Path(sglang.__file__).resolve().parent
    report["sglang_package_root"] = str(package_root)
    report["sglang_srt"] = str(package_root / "srt")

    report["sources"] = {}
    for relative in SOURCE_FILES:
        path = package_root / "srt" / relative
        report["sources"][relative] = excerpt(path) if path.is_file() else {"missing": str(path)}

    # Does the production request payload survive into SamplingParams?
    from sglang.srt.sampling.sampling_params import SamplingParams

    report["sampling_params_class"] = {
        "name": SamplingParams.__name__,
        "module": SamplingParams.__module__,
        "bases": [base.__name__ for base in SamplingParams.__mro__[1:4]],
        "is_dataclass": bool(getattr(SamplingParams, "__dataclass_fields__", None)),
        "has_model_fields": bool(getattr(SamplingParams, "model_fields", None)),
    }
    try:
        report["sampling_params_init_params"] = sorted(
            name for name, _ in inspect.signature(SamplingParams.__init__).parameters.items()
            if name != "self"
        )
    except Exception as exc:
        report["sampling_params_init_error"] = repr(exc)
    seed_attrs = sorted(
        name for name in dir(SamplingParams) if "seed" in name.lower() and not name.startswith("__")
    )
    report["sampling_params_seed_attrs"] = seed_attrs

    production_payload = {"max_new_tokens": 64, "temperature": 0.6, "top_p": 0.95}
    for label, payload in (
        ("production_keys_only", dict(production_payload)),
        ("with_sampling_seed", {**production_payload, "sampling_seed": 12345}),
        ("with_seed", {**production_payload, "seed": 12345}),
    ):
        try:
            built = SamplingParams(**payload)
            report.setdefault("sampling_params_acceptance", {})[label] = {
                "accepted": True,
                "sampling_seed_attr": str(getattr(built, "sampling_seed", "<absent>")),
                "seed_attr": str(getattr(built, "seed", "<absent>")),
                "max_new_tokens": str(getattr(built, "max_new_tokens", "<absent>")),
            }
        except Exception as exc:
            report.setdefault("sampling_params_acceptance", {})[label] = {
                "accepted": False, "error": repr(exc),
            }

    # Normalized ServerArgs need accelerator detection; report failure instead of dying.
    try:
        from sglang.srt.server_args import ServerArgs

        launch = {
            "model_path": "/assets/student",
            "trust_remote_code": True,
            "host": "127.0.0.1",
            "port": 15000,
            "nccl_port": 15001,
            "dist_init_addr": "127.0.0.1:15002",
            "tp_size": 1,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "node_rank": 0,
            "nnodes": 1,
            "enable_memory_saver": True,
            "enable_weights_cpu_backup": True,
            "skip_server_warmup": True,
            "log_level": "warning",
            "log_level_http": "warning",
        }
        report["launch_dict"] = launch
        report["server_args_fields"] = sorted(
            name for name, _ in inspect.signature(ServerArgs.__init__).parameters.items()
            if name != "self"
        )[:80]
        normalized = ServerArgs(**launch)
        report["server_args_normalized"] = {
            name: str(getattr(normalized, name, "<absent>"))
            for name in (
                "enable_deterministic_inference", "random_seed", "sampling_backend",
                "attention_backend", "prefill_attention_backend", "decode_attention_backend",
                "chunked_prefill_size", "disable_radix_cache", "cuda_graph_max_bs",
                "disable_cuda_graph", "disable_piecewise_cuda_graph", "dtype",
                "device",
            )
        }
    except Exception as exc:
        report["server_args_error"] = repr(exc)

    print("SAMPLER_AUDIT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
