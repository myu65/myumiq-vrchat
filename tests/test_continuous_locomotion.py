from types import SimpleNamespace

import pytest
from test_exploration import snapshot

from myumiq_vrchat.actuation import LocomotionCommand
from myumiq_vrchat.exploration import ContinuousExplorer
from myumiq_vrchat.locomotion import LocomotionController


def test_continuous_goal_has_no_pulse_gaps_despite_slow_actor_confirmation():
    explorer, controller = ContinuousExplorer(), LocomotionController()
    speeds = []
    for index in range(181):
        now = index / 60
        frame_at = int(now * 2) / 2  # 2 Hz detector, 60 Hz output.
        lease = explorer.tick(
            1,
            now,
            snapshot(frame_at, int(frame_at * 40)),
            True,
            motion_ready=False,
            direction="forward",
        )
        assert lease is not None and lease[1] <= now + 0.151
        state = controller.step(LocomotionCommand(forward=0.45), now, permitted=True, key=1)
        speeds.append(state.smoothed_forward_speed)
    assert all(value == pytest.approx(0.45) for value in speeds[40:])
    assert explorer.cycles == 3  # Image observations continue while inputs are maintained.
    assert controller.state.gait_speed_scale == pytest.approx(1.0)


def test_expiry_stop_preemption_and_clock_stall_release_without_a_smoothing_tail():
    controller = LocomotionController()
    demand = LocomotionCommand(forward=0.45, turn=0.3)
    for index in range(31):
        controller.step(demand, index / 30, permitted=True, key=1)
    assert controller.state.smoothed_forward_speed == 0.45
    stopped = controller.step(demand, 1.01, permitted=False, key=1)
    assert stopped.command() == LocomotionCommand()
    for index in range(1, 31):
        controller.step(demand, 1.01 + index / 30, permitted=True, key=1)
    assert controller.step(demand, 2.02, permitted=True, key=2).command() == LocomotionCommand()
    assert controller.step(demand, 10.0, permitted=True, key=2).command() == LocomotionCommand()


def test_direction_change_decelerates_through_zero_and_updates_at_30hz():
    controller = LocomotionController()
    changes, previous = 0, None
    for index in range(61):
        now = index / 60
        state = controller.step(LocomotionCommand(forward=0.45), now, permitted=True, key=1)
        if state.command() != previous:
            changes += 1
        previous = state.command()
    assert changes <= 31
    after = controller.step(LocomotionCommand(forward=-0.45), 1.04, permitted=True, key=1)
    assert 0 < after.smoothed_forward_speed < 0.45
    for index in range(1, 31):
        after = controller.step(
            LocomotionCommand(forward=-0.45), 1.04 + index / 30, permitted=True, key=1
        )
    assert after.smoothed_forward_speed == -0.45


def test_30hz_deadlines_do_not_drift_to_25hz_on_the_100hz_console_clock():
    controller = LocomotionController(acceleration=0.01)
    changes, previous = 0, 0.0
    for index in range(101):
        state = controller.step(LocomotionCommand(forward=0.45), index / 100, permitted=True, key=1)
        if state.smoothed_forward_speed != previous:
            changes += 1
        previous = state.smoothed_forward_speed
    assert changes == 30


def test_lost_camera_or_home_releases_continuous_lease():
    explorer = ContinuousExplorer()
    assert explorer.tick(1, 0.0, snapshot(0.0, 10), True, direction="forward")
    assert explorer.tick(1, 0.8, snapshot(0.0, 10), True, direction="forward") is None
    assert explorer.key == 1  # Same goal may resume on fresh feedback.
    assert explorer.tick(1, 0.9, snapshot(0.9, 80), True, direction="forward")
    assert explorer.tick(1, 1.0, snapshot(1.0, 80), False, direction="forward") is None
    assert explorer.key is None


def test_actor_readback_wait_does_not_cut_controller_input(tmp_path):
    from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig, Intent
    from myumiq_vrchat.body import simulated_body
    from myumiq_vrchat.cognition import LLMConfig
    from myumiq_vrchat.exploration import ExplorationSettings
    from myumiq_vrchat.postures import posture_target

    config = AutonomousConfig(
        llm=LLMConfig(base_url="http://localhost:1", model="fixture"),
        memory=tmp_path / "memory.json",
        exploration=ExplorationSettings(enabled=True),
    )
    pose = posture_target("standing")
    owner = AutonomousBody(config, tmp_path, pose, None)
    owner.enable(True)
    owner._choose(0.0, Intent(skill="MOVE_FORWARD", duration_s=8), "fixture")
    actor = SimpleNamespace(
        playback=SimpleNamespace(phase=0.2),
        timing=lambda *args: {"phase": "running"},
        locomotion_state=None,
    )
    owner.learned_motor = actor

    def pending_actor(*args):
        owner.learning_metadata = None  # Exactly the former input-cut condition.
        return pose

    owner._base_step = pending_actor
    for index in range(61):
        now = index / 60
        owner.exploration_lease = (owner.generation, now + 0.15, "forward")
        target = owner.step(simulated_body(pose, now), now, 1 / 60)
    assert target.left.controls.sticks[1][1] == pytest.approx(0.45)
    assert actor.locomotion_state.smoothed_forward_speed == pytest.approx(0.45)
    assert owner.locomotion.state.gait_phase == 0.2
    assert target.pelvis == pose.pelvis  # Controller demand is never integrated into tracking XY.
