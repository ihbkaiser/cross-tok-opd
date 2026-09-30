import pytest
from kdflow.deadline import stop_before_rollout


def test_deadline_is_opt_in_and_reserves_checkpoint_time():
    assert not stop_before_rollout(1000, environ={})
    env = {"MP_TRAIN_STOP_AT": "2000", "MP_CHECKPOINT_RESERVE_SECONDS": "300"}
    assert not stop_before_rollout(1000, 100, env)
    assert stop_before_rollout(1600, 100, env)
    assert stop_before_rollout(1400, 300, env)
    with pytest.raises(ValueError):
        stop_before_rollout(1, environ={"MP_TRAIN_STOP_AT": "nan"})
