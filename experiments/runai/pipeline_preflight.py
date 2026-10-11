"""Read-only node preflight; never starts a trainer or installs dependencies."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/modal/vendor")]
os.environ.setdefault("KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT", "1")

EXPECTED = {"torch": "2.11.0+cu130", "transformers": "5.6.0", "sglang": "0.5.11", "ray": "2.58.0"}
IMAGE = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--algorithm", choices=("mp_opd", "span_ctkd"), default="mp_opd")
    p.add_argument("--chunk-source", choices=("run", "xtoken"), default="run")
    p.add_argument("--check-gpu", action="store_true", help="Small CUDA context check on the already selected GPU")
    a = p.parse_args()
    result = {"python": sys.executable, "expected_runtime_image": IMAGE, "checks": [], "errors": []}
    try:
        result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        if dirty: result["errors"].append("Tracked source is dirty; use the committed source bundle")
        versions = {name: importlib.metadata.version(name) for name in EXPECTED}
        result["versions"] = versions
        for name, expected in EXPECTED.items():
            if versions[name] != expected:
                result["errors"].append(f"Unqualified {name}={versions[name]}; qualified version={expected}. Select python-b200.sh, then recheck")
        for path in (a.student, a.teacher, a.dataset):
            if not path.exists(): raise FileNotFoundError(str(path))
        from kdflow.utils.tokenizer_loader import load_tokenizer
        from tokenizers import Tokenizer
        result["tokenizers"] = {}
        tokens = []
        for name, path in (("student", a.student), ("teacher", a.teacher)):
            tok = load_tokenizer(str(path), local_files_only=True, trust_remote_code=False)
            raw = Tokenizer.from_file(str(path / "tokenizer.json"))
            for text in ("(n", "Let's", "def factorial(n):\n", "\n\t café 日本語"):
                if tok.encode(text, add_special_tokens=False) != raw.encode(text, add_special_tokens=False).ids:
                    raise RuntimeError(f"{name} tokenizer does not preserve serialized IDs")
            messages = [{"role": "user", "content": "Write factorial(n)."}]
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            ids = ids["input_ids"] if hasattr(ids, "keys") else ids
            if list(ids) != raw.encode(rendered, add_special_tokens=False).ids:
                raise RuntimeError(f"{name} rendered prompt ID mismatch")
            result["tokenizers"][name] = tok.kdflow_tokenizer_provenance
            tokens.append(tok)
        result["checks"].append("serialized tokenizer fixtures and rendered chat prompt IDs")
        if a.algorithm == "span_ctkd":
            from kdflow.algorithms._semantic_overlap import semantic_overlap
            _, _, result["semantic_overlap"] = semantic_overlap(*tokens)
            result["checks"].append("semantic common vocabulary and mapping digest")
        elif a.chunk_source == "xtoken":
            from xtoken_upstream_token_aligner import TokenAligner  # noqa: F401
            result["checks"].append("vendored native chunker import")
        import pyarrow.parquet as pq
        columns = pq.read_schema(a.dataset).names
        if not {"messages", "label", "source"}.issubset(columns):
            raise ValueError("Expected prompt parquet columns messages/label/source")
        result["dataset"] = {"path": str(a.dataset), "bytes": a.dataset.stat().st_size,
            "sha256": hashlib.sha256(a.dataset.read_bytes()).hexdigest(), "columns": columns}
        if a.check_gpu:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Expose exactly one intended GPU via CUDA_VISIBLE_DEVICES")
            result["gpu"] = torch.cuda.get_device_name(0)
            if "B200" not in result["gpu"]:
                result["errors"].append("This fix qualification used B200; this GPU needs its own canary")
        result["checks"].append("local assets and prompt schema")
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["status"] = "PASS" if not result["errors"] else "BLOCKED"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__": sys.exit(main())
