#!/usr/bin/env python3
"""Isolated, source-pinned X-Token setup. No model downloads or trainer launches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_logged import run_logged, workload_environment

EXPERIMENT = Path(__file__).resolve().parents[1]
SIMCT = EXPERIMENT.parents[1]
PIN = json.loads((EXPERIMENT / "upstream.json").read_text())
ACTIONS = ("prepare", "check", "install-base", "test-base", "gpu-smoke", "capture")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, env=workload_environment()
    ).strip()


def validate_root(value: str) -> Path:
    raw = Path(value).expanduser()
    root = raw.resolve()
    if not raw.is_absolute():
        raise ValueError("--root must be absolute")
    if len(root.parts) < 3 or root == Path.home().resolve():
        raise ValueError("--root must be a dedicated task directory, not / or home")
    if root == SIMCT or root.is_relative_to(SIMCT) or SIMCT.is_relative_to(root):
        raise ValueError("--root must be outside the SimCT worktree")
    for name in ("NeMo-RL", "artifacts", "uv-cache", "hf-cache"):
        if (root / name).is_symlink():
            raise ValueError(f"managed path must not be a symlink: {name}")
    return root


def verify_checkout(repo: Path) -> dict:
    if not repo.is_dir():
        raise ValueError("upstream checkout missing; run prepare first")
    if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo.resolve():
        raise ValueError("expected a separate upstream Git checkout")
    if git(repo, "remote", "get-url", "origin") != PIN["repository"]:
        raise ValueError("upstream origin differs from the pinned public repository")
    head = git(repo, "rev-parse", "HEAD")
    if head != PIN["commit"]:
        raise ValueError("upstream HEAD differs from pin; refusing reset/checkout")
    if git(repo, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("upstream checkout is dirty; refusing to overwrite changes")
    lock_hash = hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()
    if lock_hash != PIN["uv_lock_sha256"]:
        raise ValueError("upstream uv.lock hash differs from pin")
    submodules = {}
    # Preserve the first status character: space=initialized at expected commit.
    lines = subprocess.check_output(
        ["git", "-C", str(repo), "submodule", "status", "--recursive"],
        text=True, env=workload_environment(),
    ).splitlines()
    for line in lines:
        match = re.match(r"^ ([0-9a-f]{40}) (\S+)(?: .*)?$", line)
        if not match:
            raise ValueError("submodule missing, conflicted, or at a different commit")
        submodules[match[2]] = match[1]
    if submodules != PIN["submodules"]:
        raise ValueError("recursive submodule identities differ from pin")
    for path in submodules:
        if git(repo / path, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("upstream submodule is dirty")
    return {"head": head, "uv_lock_sha256": lock_hash, "submodules": submodules}


def resolve_uv(value: str) -> str:
    executable = shutil.which(value)
    if not executable:
        raise ValueError("uv not found; install the version recorded in upstream.json")
    executable = str(Path(executable).resolve())
    version = subprocess.check_output([executable, "--version"], text=True).split()
    if version[:2] != ["uv", PIN["uv_version"]]:
        raise ValueError(f"workload requires uv {PIN['uv_version']}; no auto-upgrade performed")
    return executable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="absolute task directory outside SimCT")
    parser.add_argument("--uv", default="uv", help="workload uv executable (pinned version)")
    parser.add_argument("--timeout", type=int, default=1800, help="per-action timeout in seconds")
    parser.add_argument("action", choices=ACTIONS)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        root = validate_root(args.root)
        repo = root / "NeMo-RL"
        env = workload_environment()
        for key in list(env):
            if key.startswith("UV_") or key in (
                "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH", "PYTHONHOME"
            ):
                env.pop(key)
        env.update(
            XTOKEN_ROOT=str(root), XTOKEN_REPO=str(repo),
            UV_NO_CACHE="false", UV_CACHE_DIR=str(root / "uv-cache"),
            UV_LINK_MODE="hardlink", HF_HOME=str(root / "hf-cache"),
            PYTHONUNBUFFERED="1", WANDB_MODE="disabled", DO_NOT_TRACK="1",
            HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="4",
        )
        if args.action == "prepare" and not repo.exists():
            root.mkdir(parents=True, exist_ok=True)
            command = ["bash", str(EXPERIMENT / "scripts" / "prepare.sh"),
                       PIN["repository"], PIN["commit"]]
            rc, _ = run_logged(command, cwd=root, root=root / "artifacts",
                               name="prepare", timeout=args.timeout, env=env)
            if rc:
                return rc
        identity = verify_checkout(repo)
        if args.action in ("check", "prepare"):
            print(json.dumps({"source_verified": True, **identity}, indent=2))
            return 0
        env["XTOKEN_UV_BIN"] = resolve_uv(args.uv)
        command = ["bash", str(EXPERIMENT / "scripts" / (args.action.replace("-", "_") + ".sh"))]
        if args.action in ("gpu-smoke", "capture"):
            script = "gpu_smoke.py" if args.action == "gpu-smoke" else "capture_environment.py"
            command = [env["XTOKEN_UV_BIN"], "run", "--no-sync", "python",
                       str(EXPERIMENT / "scripts" / script)]
        rc, out = run_logged(command, cwd=repo, root=root / "artifacts",
                             name=args.action, timeout=args.timeout, env=env)
        (out / "source-before.json").write_text(json.dumps(identity, indent=2) + "\n")
        # Record verification separately; it never turns a failed workload into success.
        try:
            after = {"source_verified": True, **verify_checkout(repo)}
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            after = {"source_verified": False, "error_type": type(exc).__name__}
            rc = rc or 1
        (out / "source-after.json").write_text(json.dumps(after, indent=2) + "\n")
        return rc
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        # Never print subprocess command lines/response bodies from a failed preflight.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"X-Token gate failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
