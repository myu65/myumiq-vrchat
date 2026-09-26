# ruff: noqa: E402 -- optional learning dependencies before adapter imports
from concurrent.futures import Future

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_articulated_body import fixture

import myumiq_vrchat.articulated_controller as module
from myumiq_vrchat.articulated_body import JointState
from myumiq_vrchat.body import simulated_body
from myumiq_vrchat.tracker_basis import RateBasis
from myumiq_vrchat.whole_body import target_from_vector, vector


def controller(monkeypatch, tmp_path):
    rig, states = fixture()
    jobs = []

    class Actor:
        manifest = {"sha256": "b" * 64, "reference_floor": 0.0}

        def __init__(self, path):
            self.rig = rig
            self.decoder = RateBasis(
                action_size=rig.action_size,
                variance_fraction=1.0,
                rows=tuple(tuple(float(x) for x in row) for row in np.eye(rig.action_size)),
            )
            self.action = np.zeros(rig.action_size)

        def predict(self, obs):
            return self.action.copy(), None

    def submit(fn):
        future = Future()
        jobs.append((future, fn))
        return future

    monkeypatch.setattr(module, "ArticulatedActor", Actor)
    return (
        module.ArticulatedController(tmp_path / "actor.pt", reference_floor=0.0, submit=submit),
        states,
        jobs,
    )


def test_fitting_holds_measured_pose_and_zero_action_does_not_publish_the_fit(
    monkeypatch, tmp_path
):
    control, states, jobs = controller(monkeypatch, tmp_path)
    pose = vector(control.rig.forward(states[1]))
    pose[:, 0] += 0.001  # an observed residual, not an instruction to snap to FK
    current = target_from_vector(pose)
    body = simulated_body(current, 1.0)
    assert not control.observe(body, 1.0)
    held, obs, rates = control.step(body, current, 0.02)
    np.testing.assert_allclose(vector(held), pose, atol=1e-12)
    assert obs is rates is None
    jobs[0][0].set_result((states[1], {"joint_state_source": "fitted_tracker_estimate"}))
    assert control.observe(body, 1.1)
    held, obs, rates = control.step(body, current, 0.02)
    np.testing.assert_allclose(vector(held), pose, atol=1e-12)
    np.testing.assert_allclose(rates, 0, atol=1e-12)
    assert obs.shape == (262,)
    control.actor.action[0] = 0.1
    following, _, rates = control.step(body, control.rig.forward(states[0]), 0.02)
    next_body = simulated_body(following, 1.2)
    assert (
        control.replay_observation(next_body, current, rates, 0.02, control.last_metadata)
        is not None
    )
    bad = vector(following)
    bad[0, 0] += 0.1
    wrong_body = simulated_body(target_from_vector(bad), 1.3)
    assert (
        control.replay_observation(wrong_body, current, rates, 0.02, control.last_metadata) is None
    )
    assert not control.observe(wrong_body, 1.3)
    assert len(jobs) == 2


def test_cancelled_fit_cannot_activate_or_create_concurrent_fitting(monkeypatch, tmp_path):
    control, states, jobs = controller(monkeypatch, tmp_path)
    body = simulated_body(control.rig.forward(states[0]), 1.0)
    control.observe(body, 1.0)
    cancel = control.pending[3]
    control.reset()
    assert cancel.is_set()
    control.observe(body, 1.1)
    assert len(jobs) == 1 and not control.ready
    jobs[0][0].set_result((states[0], {}))  # even a successful old result is stale
    control.observe(body, 1.2)
    assert len(jobs) == 2 and not control.ready
    control.end_goal()
    assert control.pending[3].is_set()


def test_cancelled_old_fit_timeout_cannot_fail_a_replacement_goal(monkeypatch, tmp_path):
    control, states, jobs = controller(monkeypatch, tmp_path)
    body = simulated_body(control.rig.forward(states[0]), 1.0)
    control.observe(body, 1.0)
    control.reset()
    # A slow cancellation still owns its worker, but no longer owns this goal.
    control.observe(body, 10.0)
    assert control.error is None and len(jobs) == 1
    assert not control.status()["fit_generation_active"]
    jobs[0][0].set_exception(RuntimeError("cancelled"))
    control.observe(body, 10.1)
    assert control.error is None and len(jobs) == 2
    assert control.status()["fit_generation_active"]
    control.observe(body, 18.2)
    assert control.error == "articulated fitting timed out"


def test_floor_rejection_keeps_current_pose_and_reports_unavailable(monkeypatch, tmp_path):
    control, states, jobs = controller(monkeypatch, tmp_path)
    state = JointState(states[0].root + [0, 0, -0.095], states[0].rotations)
    current = control.rig.forward(state)
    body = simulated_body(current, 1.0)
    control.observe(body, 1.0)
    jobs[0][0].set_result((state, {}))
    assert control.observe(body, 1.1)
    control.actor.action[2] = -1.0
    held, obs, rates = control.step(body, current, 0.1)
    assert obs is rates is None
    np.testing.assert_allclose(vector(held), vector(current), atol=1e-12)
    assert not control.ready and "reference floor" in control.status()["error"]
    np.testing.assert_array_equal(control.previous, np.zeros(66))
    rejected = control.status()["rejected_step"]
    assert rejected["proposed_minimum_foot_height_m"] < rejected["reference_floor_m"]
    assert rejected["observed_minimum_foot_height_m"] >= rejected["reference_floor_m"]
    assert rejected["sent"] is False and rejected["integration_dt_s"] == 0.1
    control.new_goal()
    assert control.status()["rejected_step"] is None


@pytest.mark.parametrize("arrival", ["confirmed", "timeout", "cancelled"])
@pytest.mark.parametrize("magnitude", [1.0, 0.025, 0.0005])
def test_delayed_device_feedback_keeps_one_command_without_reverting_or_refitting(
    monkeypatch, tmp_path, arrival, magnitude
):
    control, states, jobs = controller(monkeypatch, tmp_path)
    current = control.rig.forward(states[0])
    body = simulated_body(current, 1.0)
    control.observe(body, 1.0)
    jobs[0][0].set_result((states[0], {}))
    assert control.observe(body, 1.1)
    control.actor.action[0] = magnitude
    target, _, rates = control.step(body, current, 0.02)
    predicted = control.state
    metadata = control.last_metadata.copy()
    assert control.feedback_pending and control.steps_executed == 1
    assert not control.observe(body, 1.14)
    repeated, obs, action = control.step(body, current, 0.02)
    assert repeated == target and repeated != current
    assert obs is action is None and control.state is predicted and len(jobs) == 1
    assert control.replay_observation(body, current, rates, 0.02, metadata) is None
    if arrival == "confirmed":
        following = simulated_body(target, 1.18)
        assert control.observe(following, 1.18)
        assert control.replay_observation(following, current, rates, 0.02, metadata) is not None
        assert len(jobs) == 1 and control.steps_executed == 1
    elif arrival == "timeout":
        assert not control.observe(body, 1.5)
        held, obs, action = control.step(body, current, 0.02)
        assert held == current and len(jobs) == 2 and obs is action is None
    else:
        control.end_goal()
        held, obs, action = control.step(body, current, 0.02)
        assert held == current and obs is action is None and not control.feedback_pending


def test_confirmed_cadence_controls_rate_without_catching_up_pauses_or_deadlines(
    monkeypatch, tmp_path
):
    control, states, jobs = controller(monkeypatch, tmp_path)
    current = control.rig.forward(states[0])
    body = simulated_body(current, 1.0)
    control.observe(body, 1.0)
    jobs[0][0].set_result((states[0], {}))
    assert control.observe(body, 1.1)
    control.actor.action[0] = 1.0
    first, _, _ = control.step(body, current, 0.02)
    assert control.last_metadata["integration_dt"] == pytest.approx(0.02)
    # Four polling intervals pass; exactly one pending action is still emitted.
    assert not control.observe(body, 1.16)
    assert control.step(body, current, 0.02)[0] == first
    body = simulated_body(first, 1.18)
    assert control.observe(body, 1.18)
    second, _, rates = control.step(body, current, 0.02)
    assert control.last_metadata["integration_dt"] == pytest.approx(0.08)
    assert second.pelvis.position[0] - first.pelvis.position[0] == pytest.approx(
        4 * (first.pelvis.position[0] - current.pelvis.position[0]), rel=0.002
    )
    body = simulated_body(second, 1.38)
    assert control.observe(body, 1.38)
    third, _, _ = control.step(body, current, 0.02, remaining_s=0.03)
    assert control.last_metadata["integration_dt"] == pytest.approx(0.03)
    body = simulated_body(third, 1.60)
    assert control.observe(body, 1.60)
    fourth, _, _ = control.step(body, current, 0.02)
    assert control.last_metadata["integration_dt"] == pytest.approx(0.1)
    body = simulated_body(fourth, 1.68)
    assert control.observe(body, 1.68)
    control.end_goal()
    control.new_goal()
    body = simulated_body(fourth, 5.0)
    assert control.observe(body, 5.0)
    control.step(body, current, 0.02)
    assert control.last_metadata["integration_dt"] == pytest.approx(0.02)


def test_subresolution_commands_do_not_advance_only_the_latent_body(monkeypatch, tmp_path):
    control, states, jobs = controller(monkeypatch, tmp_path)
    current = control.rig.forward(states[0])
    body = simulated_body(current, 1.0)
    control.observe(body, 1.0)
    jobs[0][0].set_result((states[0], {}))
    assert control.observe(body, 1.1)
    control.actor.action[0] = 0.00001
    before = control.state
    for step in range(100):
        control.observe(body, 1.12 + step * 0.02)
        target, _, rates = control.step(body, current, 0.02)
        assert target == current
        assert control.state is before
        assert not control.feedback_pending
        np.testing.assert_array_equal(rates, np.zeros(66))


def test_large_tracker_update_cannot_acknowledge_a_stale_small_tracker():
    rig, states = fixture()
    before = rig.forward(states[0])
    p = vector(before)
    p[0, 0] += 0.001
    p[10, 0] += 0.000006
    expected = target_from_vector(p)
    partial = p.copy()
    partial[10] = vector(before)[10]
    assert not module.feedback_matches(expected, target_from_vector(partial), before)
    assert module.feedback_matches(expected, expected, before)
    # Float32 device readback roundoff remains admissible.
    rounded = p.copy()
    rounded[:, :3] = rounded[:, :3].astype(np.float32)
    assert module.feedback_matches(expected, target_from_vector(rounded), before)
