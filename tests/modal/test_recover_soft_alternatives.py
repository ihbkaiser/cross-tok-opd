import json
from pathlib import Path

import pytest

from experiments.modal.recover_soft_alternatives import expected_energy_updates, recover_result


def _raw(tmp_path: Path, variant: str, *, completed=True, checkpoint=True):
    every = 1 if variant == "main" else 4
    final = {"status": "completed", "optimizer_updates": 8,
             "energy_updates": expected_energy_updates(every, 8)}
    if not completed:
        final["status"] = "failed"
    result = {
        "run_id": f"run-{variant}", "variant": variant, "source_commit": "abc",
        "status": "completed" if completed else "stopped",
        "phase1_exit": 0, "phase2_exit": 0,
        "phase1_summary_present": True,
        "phase1_summary": {"status": "paused", "optimizer_updates": 4},
        "final_summary": final,
        "config": {"updates": 8, "energy_every": every},
    }
    if checkpoint:
        result["latest_checkpoint_sha256"] = "00" * 32
    path = tmp_path / f"{variant}.json"
    path.write_text(json.dumps(result))
    return path


def test_expected_energy_is_derived_from_cadence():
    assert expected_energy_updates(1, 8) == 8
    assert expected_energy_updates(4, 8) == 2
    assert expected_energy_updates(4, 9) == 3
    with pytest.raises(ValueError):
        expected_energy_updates(0, 8)


@pytest.mark.parametrize("variant,state", [("main", "stopped"), ("every4", "completed")])
def test_main_and_every4_recovery_preserves_terminal_and_training(tmp_path, variant, state):
    path = _raw(tmp_path, variant)
    receipt = recover_result(path, app_terminal_state=state, command=["offline"])
    assert receipt["training_status"] == "completed"
    assert receipt["postprocess_status"] == "recovered"
    assert receipt["app_terminal_state"] == state
    assert receipt["raw_result_sha256"]
    assert receipt["validation_status"] == "pass"


def test_missing_artifact_is_pending_not_false_pass(tmp_path):
    receipt = recover_result(_raw(tmp_path, "main", checkpoint=False), app_terminal_state="stopped")
    assert receipt["training_status"] == "completed"
    assert receipt["validation_status"] == "pending_artifact_comparator"


def test_child_failure_and_summary_mismatch_fail(tmp_path):
    path = _raw(tmp_path, "main")
    raw = json.loads(path.read_text())
    raw["phase2_exit"] = 17
    path.write_text(json.dumps(raw))
    receipt = recover_result(path)
    assert receipt["training_status"] == "failed"
    assert receipt["validation_status"] == "fail"

    raw["phase2_exit"] = 0
    raw["final_summary"]["optimizer_updates"] = 7
    path.write_text(json.dumps(raw))
    receipt = recover_result(path)
    assert receipt["training_status"] == "failed"
    assert receipt["checks"]["optimizer_updates_match"] is False


def test_root_wrapper_error_is_retained(tmp_path):
    path = _raw(tmp_path, "main")
    raw = json.loads(path.read_text())
    raw.update({"status": "stopped", "error_type": "NameError", "error": "name 'every' is not defined"})
    path.write_text(json.dumps(raw))
    receipt = recover_result(path, app_terminal_state="stopped")
    assert receipt["training_status"] == "completed"
    assert receipt["raw_error"]["type"] == "NameError"
    assert receipt["raw_error"]["message"] == "name 'every' is not defined"


def test_runai_launcher_wires_opt_in_host_mask_without_changing_default():
    import ast
    source = Path(__file__).parents[2] / "experiments" / "runai" / "run_single_gpu.py"
    tree = ast.parse(source.read_text())
    text = source.read_text()
    assert 'mp_opd_host_mask=os.environ.get("MP_OPD_HOST_MASK", "0") == "1"' in text
    # The candidate is opt-in; the default path remains false.
    assert 'host_mask: bool = False' in (Path(__file__).parents[2] / "kdflow" / "algorithms" / "_mp_opd_semimarkov.py").read_text()
