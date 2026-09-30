"""Cooperative rollout admission; no signals, imports or model side effects."""
import math
import os


def stop_before_rollout(now, previous_step_seconds=0, environ=None):
    env = os.environ if environ is None else environ
    value = env.get("MP_TRAIN_STOP_AT")
    if not value:
        return False
    deadline = float(value)
    reserve = float(env.get("MP_CHECKPOINT_RESERVE_SECONDS", "300"))
    if not math.isfinite(deadline) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Invalid cooperative training deadline/reserve")
    # Conservative admission, not a guarantee against arbitrarily slow updates.
    return now + reserve + max(120.0, 1.5 * previous_step_seconds) >= deadline
