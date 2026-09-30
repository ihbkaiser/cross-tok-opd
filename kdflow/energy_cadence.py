"""Energy-update cadence, extracted so the rule is testable without a GPU.

The trainer calls this rule with the number of student updates already completed
before the current one, so energy_every=4 fires on updates 4, 8, 12, ...
"""
from __future__ import annotations


def energy_update_due(student_updates_before: int, energy_every: int) -> bool:
    if energy_every < 1:
        raise ValueError("energy_every must be >= 1")
    if student_updates_before < 0:
        raise ValueError("student_updates_before must be >= 0")
    return (student_updates_before + 1) % energy_every == 0


def expected_energy_updates(updates: int, energy_every: int) -> int:
    if energy_every < 1:
        raise ValueError("energy_every must be >= 1")
    if updates < 0:
        raise ValueError("updates must be >= 0")
    return updates // energy_every


def schedule(updates: int, energy_every: int) -> list:
    """Per-update flag: does this update carry an energy update?"""
    return [energy_update_due(index, energy_every) for index in range(updates)]
