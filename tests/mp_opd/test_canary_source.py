"""Offline real-Git checks for source identity before paid Modal submission."""
import ast
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT/"experiments/modal/guarded_production_modal.py").read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_local_source")
namespace = {"subprocess": subprocess}
exec(compile(ast.Module(body=[function], type_ignores=[]), "<local-source-check>", "exec"), namespace)
validate = namespace["validate_local_source"]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    repo = tmp_path/"repo"; repo.mkdir()
    git(repo, "init", "-b", "vdt/ops/b200-portable")
    (repo/"source.py").write_text("pass\n")
    git(repo, "-c", "core.autocrlf=true", "add", "source.py")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    bundle = tmp_path/"source.bundle"
    git(repo, "bundle", "create", str(bundle), "vdt/ops/b200-portable")
    return repo, bundle, git(repo, "rev-parse", "HEAD")


def test_bad_sha_and_dirty_tracked_source_are_rejected_before_modal(source):
    repo, bundle, commit = source
    validate(repo, bundle, commit)
    with pytest.raises(ValueError, match="local HEAD"):
        validate(repo, bundle, "0"*40)
    (repo/"source.py").write_text("changed\n")
    with pytest.raises(ValueError, match="tracked source"):
        validate(repo, bundle, commit)


def test_clean_head_with_an_older_bundle_is_rejected(source):
    repo, bundle, old = source
    (repo/"source.py").write_text("print(1)\n")
    git(repo, "add", "source.py")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "new fixture")
    with pytest.raises(ValueError, match="bundle ref"):
        validate(repo, bundle, git(repo, "rev-parse", "HEAD"))


def test_windows_checkout_eol_policy_preserves_real_edit_detection(source):
    repo, bundle, commit = source
    (repo/"source.py").write_bytes(b"pass\r\n")
    with pytest.raises(ValueError, match="tracked source"):
        validate(repo, bundle, commit)
    validate(repo, bundle, commit, windows_checkout=True)
    (repo/"source.py").write_bytes(b"print(2)\r\n")
    with pytest.raises(ValueError, match="tracked source"):
        validate(repo, bundle, commit, windows_checkout=True)
