"""Energy cadence: step 4 fires, steps 1-3 do not, resume must not skip or double."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.energy_cadence import (  # noqa: E402
    energy_update_due,
    expected_energy_updates,
    schedule,
)

SOURCE = (ROOT / "kdflow/algorithms/mp_opd.py").read_text()


def test_every4_fires_on_the_fourth_update_only():
    assert schedule(4, 4) == [False, False, False, True]
    assert [energy_update_due(index, 4) for index in (0, 1, 2)] == [False, False, False]
    assert energy_update_due(3, 4) is True


def test_resume_at_step3_does_not_skip_or_double_step4():
    assert expected_energy_updates(3, 4) == 0
    assert energy_update_due(3, 4) is True
    assert expected_energy_updates(4, 4) == 1
    assert schedule(4, 4).count(True) == 1


def test_every1_matches_historical_counts_and_horizon_is_unchanged():
    assert schedule(4, 1) == [True, True, True, True]
    assert expected_energy_updates(312, 1) == 312
    assert expected_energy_updates(312, 4) == 78


def test_eight_update_package_has_exactly_two_energy_events():
    continuous = schedule(4, 4)
    paused_then_resumed = schedule(4, 4)
    assert continuous.count(True) == 1
    assert paused_then_resumed.count(True) == 1
    assert continuous.count(True) + paused_then_resumed.count(True) == 2


def test_invalid_cadence_is_rejected():
    for bad in (0, -1):
        try:
            energy_update_due(0, bad)
        except ValueError:
            continue
        raise AssertionError("energy_every=" + str(bad) + " must be rejected")


def test_trainer_uses_the_shared_rule_instead_of_an_inline_copy():
    assert "from kdflow.energy_cadence import energy_update_due" in SOURCE
    assert "if not energy_update_due(self.student_updates, args.mp_opd_energy_every):" in SOURCE
    assert "% args.mp_opd_energy_every" not in SOURCE
