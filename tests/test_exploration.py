import base64
import json

import cv2
import numpy as np
import pytest

from myumiq_vrchat.body import Controls, WorldState
from myumiq_vrchat.body_console import Command
from myumiq_vrchat.exploration import Explorer, PrivateHomeGate, drive_target
from myumiq_vrchat.postures import posture_target


@pytest.fixture
def immediate_gate(monkeypatch):
    from concurrent.futures import Future

    def background(function):
        result = Future()
        result.set_result(function())
        return result

    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", background)


def snapshot(now, value):
    ok, image = cv2.imencode(".jpg", np.full((80, 120, 3), value, dtype=np.uint8))
    assert ok
    return WorldState(), base64.b64encode(image).decode(), now


def test_official_compatible_thumbstick_channel_and_no_other_buttons():
    rest = posture_target("standing")
    moving = drive_target(rest, "forward")
    assert moving.left.controls.sticks[1] == (0.0, 0.45)
    assert moving.left.controls.sticks[0] == (0.0, 0.0)
    assert moving.right.controls == Controls()
    assert not any(moving.left.controls.buttons)
    assert drive_target(rest, "right").right.controls.sticks[1] == (0.85, 0.0)


def test_pulse_release_freshness_and_observed_outcome():
    explorer = Explorer()
    assert explorer.tick(1, 0, snapshot(0, 20), True) is None
    lease = explorer.tick(1, 1.1, snapshot(1.1, 20), True)
    assert lease[2] == "right" and lease[1] <= 1.25
    assert explorer.tick(1, 1.5, snapshot(1.5, 20), True) is None
    explorer.tick(1, 2.6, snapshot(2.6, 90), True)
    assert explorer.outcomes[-1]["outcome"] == "changed_view"
    assert explorer.lease[2] == "forward"
    assert explorer.tick(1, 2.7, snapshot(1.0, 90), True) is None
    assert explorer.lease is None
    assert explorer.tick(1, 3, snapshot(3, 90), False) is None


def test_no_visual_response_stops_retrying():
    explorer = Explorer()
    for now in np.arange(0, 12, 0.1):
        explorer.tick(1, float(now), snapshot(float(now), 60), True)
    assert explorer.phase == "blocked" and explorer.lease is None
    assert explorer.unchanged == 3


@pytest.mark.parametrize("direction", ["forward", "left", "right"])
def test_directed_motion_never_changes_to_exploration_direction(direction):
    explorer = Explorer()
    emitted = []
    for now in np.arange(0, 8, 0.1):
        lease = explorer.tick(
            1, float(now), snapshot(float(now), int(now * 25)), True, direction=direction
        )
        if lease:
            emitted.append(lease[2])
            assert lease[1] <= now + 0.151
    assert emitted and set(emitted) == {direction}
    assert explorer.tick(1, 9.0, snapshot(9.0, 80), False, direction=direction) is None


def test_camera_gap_preserves_observation_but_never_replays_interrupted_pulse():
    explorer = Explorer()
    explorer.tick(7, 0.0, snapshot(0.0, 20), True)
    explorer.tick(7, 0.8, None, True)
    assert explorer.phase == "observe" and explorer.deadline == 1.0
    assert explorer.tick(7, 1.1, snapshot(1.1, 20), True) is not None
    explorer.tick(7, 1.2, None, True)
    assert explorer.phase == "settle" and explorer.lease is None
    assert explorer.tick(7, 1.3, snapshot(1.3, 90), True) is None
    explorer.tick(7, 2.3, snapshot(2.3, 90), True)
    assert explorer.outcomes[-1]["outcome"] == "changed_view"
    assert explorer.cycles == 1
    explorer.tick(7, 2.4, snapshot(2.4, 90), False)
    assert explorer.key is None and explorer.phase == "idle"


def test_exploration_waits_for_confirmed_gait_without_restarting_observation():
    explorer = Explorer()
    explorer.tick(7, 0.0, snapshot(0.0, 20), True, motion_ready=False)
    assert explorer.tick(7, 1.1, snapshot(1.1, 20), True, motion_ready=False) is None
    assert explorer.deadline == 1.0
    assert explorer.tick(7, 1.2, snapshot(1.2, 20), True, motion_ready=True) is not None


@pytest.mark.usefixtures("immediate_gate")
def test_runtime_does_not_confuse_temporary_camera_loss_with_permission_loss(tmp_path):
    from types import SimpleNamespace

    from myumiq_vrchat.capabilities import CapabilityRegistry
    from myumiq_vrchat.exploration import ExplorationSettings
    from myumiq_vrchat.exploration_runtime import ExplorationRuntime

    owner = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(exploration=ExplorationSettings(enabled=True, mode="pulse_test")),
        learned_motor=None,
        world=WorldState(timestamp=0.0),
        enabled=True,
        generation=7,
        choice=(7, 0.0, SimpleNamespace(skill="EXPLORE_HOME")),
        action_timing=lambda now: {"phase": "running"},
        health={},
    )
    vision = SimpleNamespace(decision_snapshot=lambda: snapshot(owner.world.timestamp, 20))
    runner = SimpleNamespace(
        owner=owner,
        registry=CapabilityRegistry(),
        services=SimpleNamespace(vision=vision),
        shared=SimpleNamespace(working={"attention": "environment"}),
    )
    runtime = ExplorationRuntime(runner)
    runtime.gate.valid = lambda: True
    runtime.tick(0.0)
    owner.action_timing = lambda now: {"phase": "waiting_observation"}
    runtime.tick(0.8)
    assert not runner.registry.get("EXPLORE_HOME").available
    assert runtime.explorer.phase == "observe" and runtime.explorer.deadline == 1.0
    assert owner.exploration_lease is None
    owner.world = WorldState(timestamp=1.1)
    owner.action_timing = lambda now: {"phase": "running"}
    runtime.tick(1.1)
    assert owner.exploration_lease is not None
    owner.enabled = False
    runtime.tick(1.2)
    assert runtime.explorer.phase == "idle" and owner.exploration_lease is None


def test_private_home_gate_rejects_world_transition(tmp_path):
    log = tmp_path / "vr.log"
    (tmp_path / "startup.json").write_text(
        json.dumps({"private_home": True, "vrchat_log": str(log)})
    )
    gate = PrivateHomeGate(tmp_path)
    log.write_text(
        "[Behaviour] Joining wrld_test~private(user)\n[Behaviour] Entering Room: VRChat Home\n"
    )
    assert gate.valid()
    with log.open("a") as stream:
        stream.write("[Behaviour] Joining wrld_other:123\n")
    assert not gate.valid()


def test_owned_friends_home_requires_explicit_setting_and_same_startup_instance(tmp_path):
    log = tmp_path / "vr.log"
    instance = "wrld_test:123~friends(owned_user)"
    (tmp_path / "startup.json").write_text(
        json.dumps(
            {
                "private_home": False,
                "controlled_home": True,
                "instance": instance,
                "vrchat_log": str(log),
            }
        )
    )
    log.write_text(f"[Behaviour] Joining {instance}\n[Behaviour] Entering Room: VRChat Home\n")
    assert not PrivateHomeGate(tmp_path).valid()
    assert PrivateHomeGate(tmp_path, allow_controlled_home=True).valid()
    with log.open("a") as stream:
        stream.write("[Behaviour] Joining wrld_test:456~friends(owned_user)\n")
    assert not PrivateHomeGate(tmp_path, allow_controlled_home=True).valid()


def test_drive_duration_and_arguments_are_bounded():
    import pytest

    with pytest.raises(ValueError):
        Command(session="x", issued=1.0, kind="drive", direction="forward", duration_s=2.0)
    with pytest.raises(ValueError):
        Command(session="x", issued=1.0, kind="manual", direction="forward")


def test_friends_plus_home_requires_session_authorization_and_exact_instance(tmp_path):
    log = tmp_path / "vr.log"
    instance = "wrld_test:123~hidden(owned_user)~region(jp)"
    startup = {
        "private_home": False,
        "controlled_home": True,
        "instance": instance,
        "vrchat_log": str(log),
    }

    def save():
        (tmp_path / "startup.json").write_text(json.dumps(startup))

    def join(value, room="VRChat Home"):
        log.write_text(f"[Behaviour] Joining {value}\n[Behaviour] Entering Room: {room}\n")

    gate = PrivateHomeGate(tmp_path, allow_controlled_home=True)
    join(instance)
    save()
    assert not gate.valid()
    startup["allow_friends_plus_home"] = True
    save()
    assert gate.valid()
    assert not PrivateHomeGate(tmp_path).valid()
    for value in (
        instance.replace(":123", ":456"),
        instance.replace("owned_user", "other"),
        instance.replace("wrld_test", "wrld_other"),
        "wrld_test:123",
    ):
        join(value)
        assert not gate.valid()
    join(instance, "Other World")
    assert not gate.valid()
    join(instance)
    startup["controlled_home"] = False
    save()
    assert not gate.valid()


@pytest.mark.usefixtures("immediate_gate")
def test_exploration_capability_requires_a_fresh_camera_frame(tmp_path):
    from types import SimpleNamespace

    from myumiq_vrchat.capabilities import CapabilityRegistry
    from myumiq_vrchat.exploration import ExplorationSettings
    from myumiq_vrchat.exploration_runtime import ExplorationRuntime

    owner = SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(exploration=ExplorationSettings(enabled=True)),
        learned_motor=None,
        world=WorldState(),
        enabled=True,
        generation=1,
        choice=(1, 0, SimpleNamespace(skill="WAIT")),
        action_timing=lambda now: {"phase": "expired"},
        health={},
    )
    runner = SimpleNamespace(
        owner=owner,
        registry=CapabilityRegistry(),
        services=SimpleNamespace(vision=object()),
        shared=SimpleNamespace(working={"attention": "environment"}),
    )
    runtime = ExplorationRuntime(runner)
    runtime.gate.valid = lambda: True
    for now, stamp, expected in (
        (1.0, None, False),
        (2.0, 2.0, True),
        (3.0, 2.0, False),
        (4.0, 5.0, False),
        (5.0, 5.0, True),
    ):
        owner.world = WorldState(timestamp=stamp)
        runtime.tick(now)
        assert runner.registry.get("EXPLORE_HOME").available is expected
        assert owner.exploration_status["visual_ready"] is expected
        assert owner.exploration_lease is None  # WAIT never moves on frame recovery.


@pytest.mark.parametrize("skill", ["EXPLORE_HOME", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"])
def test_motor_releases_expired_and_preempted_navigation(tmp_path, skill):
    import pytest

    from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig, Intent
    from myumiq_vrchat.body import simulated_body
    from myumiq_vrchat.cognition import LLMConfig
    from myumiq_vrchat.exploration import ExplorationSettings

    owner = AutonomousBody(
        AutonomousConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            memory=tmp_path / "memory.json",
            exploration=ExplorationSettings(enabled=True),
        ),
        tmp_path,
        posture_target("standing"),
        None,
    )
    owner.enable(True)
    owner._choose(10.0, Intent(skill=skill, duration_s=10), "operator_goal_plan")
    owner.exploration_lease = (owner.generation, 10.15, "forward")
    seated = posture_target("sitting_floor")
    body = simulated_body(seated, 10.0)
    owner.step(body, 10.0, 0.02)
    moving = owner.step(body, 10.1, 0.02)
    assert moving.left.controls.sticks[1][1] > 0.0
    for part in ("head", "pelvis", "left_knee", "right_knee", "left_foot", "right_foot"):
        assert moving.pose_for(part).position == pytest.approx(seated.pose_for(part).position)
        assert moving.pose_for(part).orientation == pytest.approx(seated.pose_for(part).orientation)
    held = owner.step(body, 10.2, 0.02)
    assert held.left.controls == Controls()
    assert held.head == moving.head and held.pelvis == moving.pelvis
    owner.exploration_lease = (owner.generation, 11.0, "forward")
    owner.generation += 1  # Speech or manual preemption, before slow worker catches up.
    assert owner.step(body, 10.3, 0.02).left.controls == Controls()
