from pathlib import Path

import pytest

from kdflow.wandb_schema import (
    IMPLEMENTATION_VALIDATION_JOB_TYPE,
    build_wandb_tags,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_wandb_tags_has_stable_namespaced_order():
    tags = build_wandb_tags(
        method="mp-opd",
        regime="on-policy",
        objective="canonical-path-credit",
        variant="atomic",
        platform="modal",
        accelerator="b200x1",
        budget="100-update",
        stage=IMPLEMENTATION_VALIDATION_JOB_TYPE,
        student="llama-3.2-1b",
        teacher="qwen3-4b",
    )

    assert tags.split(",") == [
        "schema:experiment-v1",
        "framework:kdflow",
        "method:mp-opd",
        "regime:on-policy",
        "objective:canonical-path-credit",
        "variant:atomic",
        "platform:modal",
        "accelerator:b200x1",
        "budget:100-update",
        "stage:implementation-validation",
        "student:llama-3.2-1b",
        "teacher:qwen3-4b",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [("method", "MP OPD"), ("method", "mp:opd"), ("accelerator", "B200x1")],
)
def test_build_wandb_tags_rejects_noncanonical_slugs(field, value):
    kwargs = {
        "method": "mp-opd",
        "regime": "on-policy",
        "objective": "canonical-path-credit",
        "platform": "modal",
        "accelerator": "b200x1",
        "budget": "100-update",
        "stage": IMPLEMENTATION_VALIDATION_JOB_TYPE,
        "student": "llama-3.2-1b",
        "teacher": "qwen3-4b",
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_wandb_tags(**kwargs)


def test_path_diagnostic_passes_shared_schema_to_cli():
    import ast
    source = (ROOT / "experiments/modal/mp_opd_path_modal.py").read_text()
    tree = ast.parse(source)
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_impl")
    command = next(n.value for n in ast.walk(train) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "command" for t in n.targets))
    for flag, expected in [("--wandb_tags", "WANDB_TAGS"), ("--wandb_job_type", "IMPLEMENTATION_VALIDATION_JOB_TYPE")]:
        index = next(i for i, value in enumerate(command.elts) if isinstance(value, ast.Constant) and value.value == flag)
        assert isinstance(command.elts[index + 1], ast.Name)
        assert command.elts[index + 1].id == expected
