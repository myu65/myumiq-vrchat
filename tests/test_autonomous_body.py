import json
import time

import pytest

from myumiq_vrchat.autonomous_body import (
    AutonomousBody,
    AutonomousConfig,
    Intent,
    fallback_intent,
)
from myumiq_vrchat.autonomy import Drives
from myumiq_vrchat.body import WorldObject, WorldState, simulated_body
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.postures import posture_target


def make_agent(tmp_path):
    config = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test", timeout_s=1),
        memory=tmp_path / "memory.json",
        retry_s=2,
    )
    return AutonomousBody(config, tmp_path, posture_target("standing"), None)


def test_intents_reject_joint_control_and_invalid_arguments():
    for payload in (
        {"skill": "WAVE"},
        {"skill": "SIT", "hand": "right"},
        {"skill": "WAIT", "position": [0, 0, 0]},
        {"skill": "WALK_IN_PLACE", "duration_s": 100},
    ):
        with pytest.raises(ValueError):
            Intent(**payload)


def test_visual_geometry_is_not_reach_ground_truth():
    world = WorldState(objects=(WorldObject(name="p", position=(1, 0, 1), source="vision"),))
    with pytest.raises(ValueError):
        Intent(skill="REACH", hand="right", target="p").validate_world(world)
    Intent(skill="LOOK_AT", target="p").validate_world(world)


@pytest.mark.parametrize("missing", [False, True])
def test_gaze_holds_observed_heading_during_feedback_gap(tmp_path, missing):
    from myumiq_vrchat.body import Pose

    agent = make_agent(tmp_path)
    agent.enable(True)
    agent.choice = (1, 10, Intent(skill="LOOK_AT", target="p", duration_s=5), "local_llm")
    turned = agent.rest.model_copy(
        update={
            "head": Pose(
                position=agent.rest.head.position, orientation=(0.9800665778, 0, 0, 0.1986693308)
            )
        }
    )
    body = simulated_body(turned, 10)
    obj = WorldObject(
        name="p",
        position=(1, 0, 1.6),
        source="vision",
        image_position=(0.2, 0),
        last_seen=9,
        confidence=0.8,
    )
    agent.world = WorldState(objects=() if missing else (obj,))
    held = agent.step(body, 10.1, 0.01)
    assert held.head.orientation == pytest.approx(body.head.pose.orientation)
    assert agent.health["motor_skill"]["state"] == "pending"
    agent.world = WorldState(objects=(obj.model_copy(update={"last_seen": 10}),))
    agent.step(body, 10.2, 0.01)
    assert "motor_skill" not in agent.health
    expired = agent.step(body, 16, 0.01)
    assert expired.head.orientation == pytest.approx(held.head.orientation)
    assert agent.intent_metadata["body_state"] == "holding_observed_posture"
    agent._choose(16, Intent(skill="RETURN_TO_REST", duration_s=3), "operator")
    reset = agent.step(body, 16.1, 0.01)
    assert reset.head.orientation != held.head.orientation
    agent.enable(False)
    assert agent.step(body, 17, 0.01).head.orientation == pytest.approx(held.head.orientation)


def test_posture_is_bounded_and_expired_intent_holds_pose(tmp_path):
    agent = make_agent(tmp_path)
    agent.enable(True)
    agent.choice = (1, 10, Intent(skill="SIT", duration_s=5), "local_llm")
    body = simulated_body(agent.rest, 10)
    target = agent.step(body, 10.1, 0.01)
    assert target.is_full_body
    assert abs(target.head.position[2] - agent.rest.head.position[2]) <= 0.0061
    expired = agent.step(simulated_body(target, 20), 20, 0.01)
    assert expired == target


def test_hold_latches_current_pose_instead_of_echoing_delayed_previous_outputs(tmp_path):
    agent = make_agent(tmp_path)
    agent.enable(True)
    agent._choose(10.0, Intent(skill="WAIT", duration_s=3), "operator")
    current = posture_target("crouching")
    assert agent.step(simulated_body(current, 10.0), 10.0, 0.02) == current
    assert agent.step(simulated_body(agent.rest, 10.02), 10.02, 0.02) == current
    assert agent.step(simulated_body(current, 14.0), 14.0, 0.02) == current
    agent._choose(15.0, Intent(skill="WAIT", duration_s=3), "operator")
    assert agent.step(simulated_body(agent.rest, 15.0), 15.0, 0.02) == agent.rest


@pytest.mark.parametrize("skill", ["WALK_IN_PLACE", "EXPLORE_HOME", "RETURN_TO_REST"])
def test_all_expired_intentions_hold_full_body_without_input_lease(tmp_path, skill):
    from myumiq_vrchat.body import Controls
    from myumiq_vrchat.exploration import ExplorationSettings

    agent = make_agent(tmp_path)
    agent.config = agent.config.model_copy(
        update={"exploration": ExplorationSettings(enabled=True)}
    )
    agent.enable(True)
    agent._choose(10, Intent(skill=skill, duration_s=1), "operator")
    agent.exploration_lease = (agent.generation, 30, "forward")
    lying = posture_target("lying")
    held = agent.step(simulated_body(lying, 12), 12, 0.02)
    assert held == lying
    assert held.left.controls == held.right.controls == Controls()
    assert agent.goal.tasks[0].kind == "hold"


def test_stationary_actions_preserve_seated_body_and_wait(tmp_path):
    agent = make_agent(tmp_path)
    agent.enable(True)
    seated = simulated_body(posture_target("sitting_floor"), 10)
    for intent in (Intent(skill="SIT", duration_s=3), Intent(skill="WAIT", duration_s=3)):
        agent._choose(1, intent, "local_llm_plan")
        held = agent.step(seated, 10, 0.02)
        assert held.pelvis == seated.pelvis.pose
        assert held.head == seated.head.pose
    agent._choose(10, Intent(skill="WAVE", hand="right", duration_s=3), "local_llm_plan")
    wave = agent.step(seated, 10.5, 0.02)
    assert wave.pelvis == seated.pelvis.pose
    assert wave.head == seated.head.pose
    assert wave.left.pose == seated.left.pose
    assert wave.right.pose != seated.right.pose
    agent._choose(11, Intent(skill="WAIT", duration_s=3), "local_llm_plan")
    waiting = agent.step(simulated_body(wave, 11), 11.1, 0.02)
    assert waiting == wave


def test_llm_outage_keeps_drives_running_and_records_explicit_fallback(tmp_path):
    agent = make_agent(tmp_path)
    agent.start()
    agent.enable(True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and agent.choice[3] != "drive_fallback":
            time.sleep(0.05)
        assert agent.choice[3] == "drive_fallback"
        assert agent.health["llm"]["state"] == "degraded"
        assert agent.health["voice"]["state"] == "pending_configuration"
        assert agent.error is None
    finally:
        agent.close()
    # The choice is published before its audit write; join the writer first.
    decisions = (tmp_path / "decisions.jsonl").read_text("utf-8").splitlines()
    assert any(json.loads(row)["source"] == "drive_fallback" for row in decisions)
    assert (tmp_path / "memory.json").exists()


def test_fallback_is_drive_conditioned_and_varied():
    assert fallback_intent(Drives(fatigue=0.9), WorldState(), [], True).skill == "SIT"
    assert fallback_intent(Drives(), WorldState(), [], True).skill == "WAVE"
    assert (
        fallback_intent(Drives(), WorldState(), [{"skill": "WAVE"}], True).skill == "WALK_IN_PLACE"
    )


def test_manual_override_invalidates_pending_generation(tmp_path):
    agent = make_agent(tmp_path)
    agent.enable(True)
    old = agent.generation
    agent.enable(False)
    assert not agent.enabled and agent.generation > old
    assert agent.choice[2].skill == "WAIT"


def test_broken_optional_configs_do_not_abort_executive(tmp_path):
    agent = make_agent(tmp_path)
    agent.config = agent.config.model_copy(
        update={
            "vision": tmp_path / "missing-vision.json",
            "voice": tmp_path / "missing-voice.json",
        }
    )
    agent.start()
    agent.enable(True)
    try:
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and agent.choice[3] != "drive_fallback":
            time.sleep(0.05)
        assert agent.enabled and agent.error is None
        assert agent.health["vision"]["state"] == "pending"
        assert agent.health["voice"]["state"] == "pending"
        assert agent.choice[3] == "drive_fallback"
    finally:
        agent.close()


def test_speech_onset_preempts_body_before_transcription(tmp_path, monkeypatch):
    from myumiq_vrchat.autonomous_services import Services
    from myumiq_vrchat.events import RuntimeEvent

    sent = []

    def poll(self, now, enabled=True):
        if enabled and not sent:
            sent.append(True)
            return WorldState(), [RuntimeEvent("speech_started", now)]
        return WorldState(), []

    monkeypatch.setattr(Services, "poll", poll)
    agent = make_agent(tmp_path)
    agent.enable(True)
    agent.choice = (
        agent.generation,
        time.perf_counter(),
        Intent(skill="WAVE", hand="right"),
        "local_llm",
    )
    agent.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and agent.choice[3] != "speech_onset":
            time.sleep(0.02)
        assert agent.choice[2].skill == "WAIT"
        assert agent.choice[3] == "speech_onset"
    finally:
        agent.close()
