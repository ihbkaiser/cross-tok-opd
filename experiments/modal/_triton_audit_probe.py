"""Installed-source audit for the Triton deterministic candidate (CPU only).

Answers two questions before spending GPU time: what the pinned SGLang ships for
the Triton attention backend, and what Gemma2 semantics (sliding window) the
student config requires. Absence of evidence is reported, never assumed away.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NEEDLES = (
    "sliding_window", "triton", "trtllm_mha", "flashinfer", "is_sm100",
    "DETERMINISTIC_ATTENTION_BACKEND_CHOICES", "attention_backend",
)
FILES = (
    "srt/layers/attention/triton_backend.py",
    "srt/layers/attention/attention_registry.py",
    "srt/layers/attention/flashattention_backend.py",
    "srt/server_args.py",
    "srt/model_executor/model_runner.py",
    "srt/models/gemma2.py",
    "srt/configs/model_config.py",
)
STUDENT_CONFIG = Path("/assets/student/config.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt(path: Path, window: int = 3) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    hits: set[int] = set()
    symbols: dict = {}
    current = "<module>"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "async def ")):
            current = stripped.split("(")[0].split()[-1]
        if any(needle in line for needle in NEEDLES):
            hits.update(range(max(0, index - window), min(len(lines), index + window + 1)))
            symbols.setdefault(current, []).append(index + 1)
    return {
        "path": str(path), "sha256": sha256(path), "lines": len(lines),
        "occurrences_by_symbol": symbols,
        "excerpt": [f"{index + 1}: {lines[index]}" for index in sorted(hits)],
    }


def main() -> int:
    import sglang

    package = Path(sglang.__file__).resolve().parent
    report: dict = {
        "sglang_package": str(package),
        "executable": sys.executable,
        "python": sys.version,
        "sources": {},
        "gemma2_student_config": None,
        "deterministic_backend_choices": None,
        "triton_module_present": None,
    }

    for relative in FILES:
        path = package / relative
        report["sources"][relative] = excerpt(path) if path.is_file() else {"missing": str(path)}

    report["triton_module_present"] = (package / "srt/layers/attention/triton_backend.py").is_file()
    try:
        from sglang.srt.server_args import DETERMINISTIC_ATTENTION_BACKEND_CHOICES

        report["deterministic_backend_choices"] = list(DETERMINISTIC_ATTENTION_BACKEND_CHOICES)
    except Exception as exc:
        report["deterministic_backend_choices_error"] = repr(exc)
    try:
        from sglang.srt.utils import is_sm100_supported

        report["is_sm100_supported_importable"] = True
    except Exception as exc:
        report["is_sm100_supported_importable"] = repr(exc)
    if STUDENT_CONFIG.is_file():
        config = json.loads(STUDENT_CONFIG.read_text())
        report["gemma2_student_config"] = {
            "sha256": sha256(STUDENT_CONFIG),
            "architectures": config.get("architectures"),
            "sliding_window": config.get("sliding_window"),
            "sliding_window_size": config.get("sliding_window_size"),
            "layer_types": (config.get("layer_types") or [])[:12],
            "attn_logit_softcapping": config.get("attn_logit_softcapping"),
            "query_pre_attn_scalar": config.get("query_pre_attn_scalar"),
        }
    print("TRITON_AUDIT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
