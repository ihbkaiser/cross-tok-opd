from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.environments.patch_llamafactory_json_path import NEW, OLD, patch_text


def test_patch_replaces_exact_bug_pattern() -> None:
    source = f"before\n    {OLD}\nafter\n"
    patched = patch_text(source)
    assert OLD not in patched
    assert patched.count(NEW) == 1


def test_patch_rejects_unknown_source() -> None:
    try:
        patch_text("upstream changed")
    except RuntimeError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("patch must fail closed")
