"""Apply the narrow LlamaFactory 0.9.5 Python 3.12 JSON-path fix.

The 0.9.5 parser passes a pathlib.Path directly to json.load(). Python's
stdlib json.load() expects a readable file object. Keep this workaround
fail-closed so a future upstream change cannot be patched accidentally.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sysconfig


OLD = "dict_config = OmegaConf.create(json.load(Path(sys.argv[1]).absolute()))"
NEW = (
    "dict_config = OmegaConf.create("
    "json.load(Path(sys.argv[1]).absolute().open(encoding=\"utf-8\")))"
)


def patch_text(source: str) -> str:
    if source.count(OLD) != 1:
        raise RuntimeError("expected exactly one LlamaFactory 0.9.5 JSON-path pattern")
    patched = source.replace(OLD, NEW)
    if patched.count(NEW) != 1 or OLD in patched:
        raise RuntimeError("LlamaFactory JSON-path patch verification failed")
    return patched


def default_target() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "llamafactory/hparams/parser.py"


def patch_file(target: Path) -> None:
    source = target.read_text(encoding="utf-8")
    patched = patch_text(source)
    pending = target.with_suffix(target.suffix + ".pending")
    pending.write_text(patched, encoding="utf-8")
    os.chmod(pending, target.stat().st_mode)
    pending.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=default_target())
    args = parser.parse_args()
    patch_file(args.target)
    print("LLAMAFACTORY_JSON_PATH_PATCH=pass")


if __name__ == "__main__":
    main()
