# ruff: noqa: E402 -- optional actor dependency
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
from test_articulated_controller import controller, low_hand_state

from myumiq_vrchat.body import simulated_body
from myumiq_vrchat.buffered_actor import BufferedActor
from myumiq_vrchat.motion_buffer import MotionBuffer, MotionKnot, blend_state
from myumiq_vrchat.whole_body import target_from_vector, vector


def prepared(monkeypatch, tmp_path):
    base, states, jobs = controller(monkeypatch, tmp_path)
    base.state = states[0]
    pose = base.rig.forward(states[0])
    base.expected = pose
    base.actor.action[0] = 0.2
    buffered = BufferedActor(base)
    return buffered, pose, jobs


def test_buffer_interpolates_skeleton_preserves_residual_and_holds_after_end(monkeypatch, tmp_path):
    control, pose, _ = prepared(monkeypatch, tmp_path)
    rig, start = control.rig, control.state
    action = np.zeros(rig.action_size)
    action[0] = 0.5
    following, _, rates, _ = rig.advance(start, action, 0.05)
    buffer = MotionBuffer(rig, start, pose)
    buffer.extend([MotionKnot(1, start, np.zeros(66)), MotionKnot(1.05, following, rates)])
    middle = buffer.sample(1.025)
    np.testing.assert_allclose(middle.state.root, (start.root + following.root) / 2)
    np.testing.assert_allclose(vector(buffer.pose(start)), vector(pose), atol=1e-12)
    assert buffer.sample(100) is buffer.knots[-1]
    for fraction in np.linspace(0, 1, 21):
        rig.validate_limits(blend_state(rig, start, following, fraction))


def test_actor_continues_through_readback_delay_and_bounded_worker_stall(monkeypatch, tmp_path):
    control, pose, jobs = prepared(monkeypatch, tmp_path)
    control.observe(simulated_body(pose, 1), 1)
    held, obs, rates = control.step(simulated_body(pose, 1), pose, 0.01)
    assert obs is rates is None and len(jobs) == 1
    jobs[0][0].set_result(jobs[0][1]())
    history = [pose]
    for i in range(1, 40):
        now = 1 + i * 0.01
        # Device packets lag two motor frames; this no longer gates every step.
        body = simulated_body(history[max(0, len(history) - 3)], now)
        control.observe(body, now)
        value, _, _ = control.step(body, pose, 0.01)
        history.append(value)
    assert control.error is None and control.started and control.horizons == 1
    assert len(jobs) == 2  # second worker deliberately never returns
    assert vector(history[-1])[0, 0] > vector(history[15])[0, 0]
    at_end = vector(history[-1])
    body = simulated_body(history[-1], 1.45)
    control.observe(body, 1.45)
    held, _, _ = control.step(body, pose, 0.01)
    assert np.max(np.abs(vector(held) - at_end)) < 0.01
    control.observe(simulated_body(held, 2), 2)
    control.step(simulated_body(held, 2), pose, 0.01)
    assert control.error == "motion horizon producer timed out"
    assert control.replay_observation(None) is None


def test_cancelled_horizon_never_resumes_or_overlaps_worker(monkeypatch, tmp_path):
    control, pose, jobs = prepared(monkeypatch, tmp_path)
    control.observe(simulated_body(pose, 1), 1)
    control.step(simulated_body(pose, 1), pose, 0.01)
    control.end_goal()
    jobs[0][0].set_result(jobs[0][1]())
    control.observe(simulated_body(pose, 1.01), 1.01)
    held, obs, _ = control.step(simulated_body(pose, 1.01), pose, 0.01)
    assert held == pose and obs is None and control.horizons == 0


def test_feedback_loss_and_divergence_stop_the_buffer(monkeypatch, tmp_path):
    control, pose, jobs = prepared(monkeypatch, tmp_path)
    control.observe(simulated_body(pose, 1), 1)
    control.step(simulated_body(pose, 1), pose, 0.01)
    jobs[0][0].set_result(jobs[0][1]())
    control.observe(simulated_body(pose, 1.05), 1.05)
    control.step(simulated_body(pose, 1.05), pose, 0.01)
    assert not control.observe(simulated_body(pose, 1), 1.6)
    assert "fresh device feedback" in control.error


@pytest.mark.parametrize("stage", ["planning", "interpolation"])
def test_buffer_rejects_hand_floor_crossing_with_feet_above_floor(monkeypatch, tmp_path, stage):
    control, pose, jobs = prepared(monkeypatch, tmp_path)
    control.base.state = low_hand_state(control.rig, control.state)
    pose = control.rig.forward(control.state)
    control.base.expected = pose
    control.base.actor.action[:] = 0
    control.base.actor.action[2] = -1
    control.observe(simulated_body(pose, 1), 1)
    held, _, _ = control.step(simulated_body(pose, 1), pose, 0.01)
    if stage == "planning":
        with pytest.raises(ValueError, match="tracking floor"):
            jobs[0][1]()
    else:
        # Interpolated samples need their own check, even if planner knots passed.
        points = vector(pose)
        points[3, 2] = -0.01
        monkeypatch.setattr(control.buffer, "pose", lambda state: target_from_vector(points))
        held, _, _ = control.step(simulated_body(pose, 1), pose, 0.01)
        assert held == pose and "tracking floor" in control.error
