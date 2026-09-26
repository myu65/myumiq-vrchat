import numpy as np
import pytest

from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.tracker_action import ACTION_SHAPE, integrate_tracker_action
from myumiq_vrchat.whole_body import vector


@pytest.mark.parametrize("posture", ["standing", "sitting_floor", "lying"])
def test_zero_action_preserves_full_body_without_gravity(posture):
    body = posture_target(posture)
    result = integrate_tracker_action(body, np.zeros(ACTION_SHAPE), 0.05)
    assert vector(result) == pytest.approx(vector(body))
    assert all(xy == (0.0, 0.0) for xy in result.left.controls.sticks)


def test_coordinated_motion_is_bounded_and_rotations_normalized():
    body = posture_target("standing")
    result = integrate_tracker_action(body, np.ones(ACTION_SHAPE), 0.05)
    before, after = vector(body), vector(result)
    assert np.linalg.norm(after[:, :3] - before[:, :3], axis=1) == pytest.approx(np.full(11, 0.03))
    assert np.linalg.norm(after[:, 3:], axis=1) == pytest.approx(np.ones(11))
    with pytest.raises(ValueError):
        integrate_tracker_action(body, np.full(ACTION_SHAPE, np.nan), 0.05)
    with pytest.raises(ValueError):
        integrate_tracker_action(body, np.zeros(ACTION_SHAPE), 1.0)
