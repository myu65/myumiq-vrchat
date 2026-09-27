# ruff: noqa: E402 -- optional learning dependencies before runtime imports
import hashlib
import json
import math
import time
from concurrent.futures import Future

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_articulated_controller import controller
from test_embodied_decision import Services, result

from myumiq_vrchat.articulated_tasks import ArticulatedIntentMotor, ArticulatedTasks, anchored_goal
from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig, Intent
from myumiq_vrchat.body import qmul, rotate, simulated_body
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.decision import BackendConfig
from myumiq_vrchat.decision_runtime import DecisionSettings
from myumiq_vrchat.purpose_runtime import PurposeRunner, PurposeSettings
from myumiq_vrchat.purposes import Purpose, SkillRequest
from myumiq_vrchat.whole_body import target_from_vector, vector


def test_acquisition_installs_and_restores_motion_without_replacing_body(
    timed_motor, tmp_path, monkeypatch
):
    from myumiq_vrchat.capability_learning import LearningSettings
    from myumiq_vrchat.motion_prior import FiniteImitation

    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    source = tmp_path / "source.gltf"
    source.write_text("test source")
    cfg = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "legacy.json",
        purpose=PurposeSettings(
            state=tmp_path / "purpose.json",
            learning=LearningSettings(
                motion_source=source, motion_license=source, source_url="fixture"
            ),
        ),
    )
    owner = AutonomousBody(cfg, tmp_path, current, None)
    owner.learned_motor = motor
    runner = PurposeRunner(owner, Services())
    jobs = []

    def background(fn):
        future = Future()
        jobs.append(future)
        return future

    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", background)
    goal = Purpose(
        description="learn walk",
        reason="missing",
        success_description="candidate available",
        steps=(SkillRequest(capability="WALK_IN_PLACE"),),
    )
    old_choice = owner.choice
    old_condition = runner.condition(Intent(skill="WALK_IN_PLACE"))
    try:
        assert runner.queue_learning(goal) == ["WALK_IN_PLACE"]
        assert runner.queue_learning(goal) == ["WALK_IN_PLACE"]
        assert len(runner.tasks) == 1
        runner._learning()
        assert len(jobs) == 1
        checkpoint = tmp_path / "learning" / "new" / "policy.json"
        checkpoint.parent.mkdir(parents=True)
        frames = [(float(t), current) for t in np.linspace(0, 1, 41)]
        motion = FiniteImitation.fit(frames, 1.0, "fixture", count=12)
        motion.save(checkpoint)
        report = {
            "policy": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluation_scope": "fixture",
        }
        jobs[0].set_result((motion, report))
        runner._learning()
        motor.configure_registry(runner.registry)
        assert runner.registry.get("WALK_IN_PLACE").available
        assert runner.tasks[0]["status"] == "trained" and owner.model is None
        assert "WALK_IN_PLACE" in motor.motions and owner.choice == old_choice
        new_condition = runner.condition(Intent(skill="WALK_IN_PLACE"))
        assert new_condition != old_condition
        assert json.loads(new_condition)["motion_sha256"] == report["sha256"]
        runner.close()
        motor.motions.clear()
        runner = PurposeRunner(owner, Services())
        assert runner.registry.get("WALK_IN_PLACE").available
        assert "WALK_IN_PLACE" in motor.motions
        runner.close()
        checkpoint.write_text("{}")
        motor.motions.clear()
        runner = PurposeRunner(owner, Services())
        assert not runner.registry.get("WALK_IN_PLACE").available
        assert runner.tasks[0]["status"] == "invalid_artifact"
    finally:
        runner.close()
        owner.close()  # also valid before its worker was started


@pytest.mark.parametrize("periodic", [False, True])
def test_motion_completion_uses_its_task_criterion(timed_motor, periodic):
    from types import SimpleNamespace

    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    other = vector(current)
    other[0, 0] += 0.2  # arbitrary stopping phase is outside the static endpoint tolerance
    motor.pose_goal = target_from_vector(other)
    motor.key = 3
    motion = {"endpoint_required": not periodic, "sequence_observed": True}
    motor.playback = SimpleNamespace(completed=True, evidence=lambda: motion)
    result = motor.goal_evidence(simulated_body(current, 10.0), 10.0, 3)
    assert result["success"] is periodic
    assert result["scope"] == "simulated_tracker_motion"
    assert result["endpoint_within_tolerance"] is False
    motor.playback.completed = False
    assert motor.goal_evidence(simulated_body(current, 10.0), 10.0, 3)["success"] is False


def test_anchor_preserves_current_heading_xy_and_source_height(monkeypatch, tmp_path):
    control, states, _ = controller(monkeypatch, tmp_path)
    template = control.rig.forward(states[0])
    points = vector(template)
    q = (math.cos(0.7), 0.0, 0.0, math.sin(0.7))
    for row in points:
        row[:3] = np.asarray(rotate(q, tuple(row[:3]))) + [0.4, -0.3, 0.0]
        row[3:] = qmul(q, tuple(row[3:]))
    current = target_from_vector(points)
    goal = anchored_goal(template, current)
    np.testing.assert_allclose(vector(goal), vector(current), atol=1e-12)
    other = anchored_goal(control.rig.forward(states[1]), current)
    assert other.pelvis.position[:2] == pytest.approx(current.pelvis.position[:2])
    assert other.pelvis.position[2] == pytest.approx(
        control.rig.forward(states[1]).pelvis.position[2]
    )


def test_independent_decision_runs_learned_body_during_pending_dialogue(monkeypatch, tmp_path):
    control, states, fits = controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "myumiq_vrchat.articulated_tasks.ArticulatedController", lambda *a, **kw: control
    )
    settings = ArticulatedTasks(
        execution_mode="feedback",
        actor=tmp_path / "actor.pt",
        actor_sha256="b" * 64,
        rig_sha256=hashlib.sha256(control.rig.model_dump_json().encode()).hexdigest(),
        reference_floor=0.0,
        goals={"STAND": control.rig.forward(states[0]), "SIT": control.rig.forward(states[1])},
    )
    path = tmp_path / "tasks.json"
    path.write_text(settings.model_dump_json(), "utf-8")
    config = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "legacy.json",
        purpose=PurposeSettings(state=tmp_path / "purpose.json"),
        decision=DecisionSettings(backend=BackendConfig(backend="http")),
        articulated_tasks=path,
    )
    owner = AutonomousBody(config, tmp_path, control.rig.forward(states[0]), None)
    owner.enable(True)
    owner.snapshot = simulated_body(control.rig.forward(states[0]), time.perf_counter())
    jobs = []

    def background(fn):
        future = Future()
        jobs.append((future, fn))
        return future

    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", background)
    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", lambda s, r: result(r, "SIT"))
    monkeypatch.setattr(
        "myumiq_vrchat.autonomous_body.ProceduralMotor",
        lambda *a: pytest.fail("learned task must not start a procedural motor"),
    )
    monkeypatch.setattr(owner.attention_motor, "apply", lambda *a: pytest.fail("no pose overlays"))
    runner = PurposeRunner(owner, Services())
    now = time.perf_counter()
    try:
        runner.shared.hear("座って")
        pending = Future()
        runner.dialogue_pending = (pending, "dialogue", runner.dialogue_epoch)
        runner.tick(now)
        jobs[0][0].set_result(jobs[0][1]())
        runner.tick(now + 0.01)
        assert owner.choice[2].skill == "SIT" and not pending.done()
        assert runner.registry.get("SIT").available and not runner.registry.get("WAVE").available
        current = control.rig.forward(states[0])
        body = simulated_body(current, now + 0.02)
        assert owner.step(body, now + 0.02, 0.02) == current and fits
        bound_goal = owner.learned_motor.pose_goal
        preparing = now + owner.choice[2].duration_s + 0.1
        assert owner.step(simulated_body(current, preparing), preparing, 0.02) == current
        assert owner.action_timing(preparing)["phase"] == "preparing"
        assert not runner._observe_step(preparing)  # accepted time is not execution time
        assert runner.running["observations"] == 0 and not pending.done()
        fits[0][0].set_result((states[0], {}))
        control.actor.action[0] = 0.1
        moved = owner.step(simulated_body(current, preparing + 0.02), preparing + 0.02, 0.02)
        assert moved.pelvis.position[0] > current.pelvis.position[0]
        assert owner.learning_metadata["goal_owner"] == "autonomous"
        assert owner.learning_metadata["intent_generation"] == owner.choice[0]
        assert owner.action_timing(preparing + 0.02)["deadline"] == pytest.approx(
            preparing + 0.02 + owner.choice[2].duration_s
        )
        next_body = simulated_body(moved, preparing + 0.04)
        owner.step(next_body, preparing + 0.04, 0.02)
        assert owner.learned_motor.pose_goal is bound_goal  # no moving goal/anchor each frame
        held = owner.step(next_body, now + 25, 0.02)
        assert held == moved and owner.learning_metadata is None
        np.testing.assert_array_equal(control.previous, np.zeros(66))
        # The outcome belongs to the actual anchored goal, never a legacy posture.
        monkeypatch.setattr(
            "myumiq_vrchat.postures.posture_target",
            lambda *a: pytest.fail("must evaluate the learned goal"),
        )
        owner.snapshot = simulated_body(bound_goal, now + 25)
        runner.running["observations"] = 3
        assert runner._observe_step(now + 25)
        evidence = runner.history[-1]["evidence"]
        assert evidence["success"] is True and evidence["scope"] == "simulated_tracker_goal"
        owner._choose(now + 26, Intent(skill="WAVE", hand="left"), "operator_test")
        assert owner.step(next_body, now + 26.01, 0.02) == moved
        assert owner.learning_metadata is None and owner.learned_motor.error
    finally:
        runner.close()
        control.close()


@pytest.mark.parametrize("case", ["goal", "wrong_hand", "stale_foot", "other_goal", "actor_error"])
def test_learned_outcome_uses_fresh_whole_body_and_matching_goal(monkeypatch, tmp_path, case):
    control, states, _ = controller(monkeypatch, tmp_path)
    motor = object.__new__(ArticulatedIntentMotor)
    motor.controller, motor.policy_id, motor.key = control, "candidate", 7
    motor.pose_goal = control.rig.forward(states[1])
    pose = vector(motor.pose_goal)
    if case == "wrong_hand":
        pose[1, 0] += 0.2
    body = simulated_body(target_from_vector(pose), 10.0)
    if case == "stale_foot":
        body = body.model_copy(
            update={"left_foot": body.left_foot.model_copy(update={"timestamp": 9.0})}
        )
    if case == "actor_error":
        control.error = "floor rejection"
    evidence = motor.goal_evidence(body, 10.0, 8 if case == "other_goal" else 7)
    assert evidence["success"] is (None if case in ("stale_foot", "other_goal") else case == "goal")


def test_task_trial_requires_separate_decision_and_rejects_inconsistent_catalogue(tmp_path):
    with pytest.raises(ValueError, match="independent decision"):
        AutonomousConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            memory=tmp_path / "memory",
            articulated_tasks=tmp_path / "tasks",
        )
    with pytest.raises(ValueError):
        ArticulatedTasks.model_validate_json(
            json.dumps(
                {
                    "actor": "x",
                    "actor_sha256": "b" * 64,
                    "rig_sha256": "a" * 64,
                    "reference_floor": 0.0,
                    "goals": {"LOOK_AT": {}},
                }
            )
        )


@pytest.fixture
def timed_motor(monkeypatch, tmp_path):
    control, states, fits = controller(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "myumiq_vrchat.articulated_tasks.ArticulatedController", lambda *a, **kw: control
    )
    settings = ArticulatedTasks(
        execution_mode="feedback",
        actor=tmp_path / "actor.pt",
        actor_sha256="b" * 64,
        rig_sha256=hashlib.sha256(control.rig.model_dump_json().encode()).hexdigest(),
        reference_floor=0.0,
        goals={"STAND": control.rig.forward(states[0]), "SIT": control.rig.forward(states[1])},
    )
    path = tmp_path / "timed-tasks.json"
    path.write_text(settings.model_dump_json(), "utf-8")
    motor = ArticulatedIntentMotor(path)
    yield motor, states, fits
    motor.close()


def test_preparation_preserves_full_execution_time_and_deadline(timed_motor):
    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    choice = (7, 10.0, Intent(skill="SIT", duration_s=2.0), "test")
    assert motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02) == current
    # The accepted duration has already elapsed, but fitting is still bounded preparation.
    assert motor.step(simulated_body(current, 13.0), choice, 13.0, 0.02) == current
    assert motor.timing(choice, 13.0)["phase"] == "preparing"
    assert motor.learning_metadata is None
    fits[0][0].set_result((states[0], {}))
    motor.step(simulated_body(current, 13.1), choice, 13.1, 0.02)
    timing = motor.timing(choice, 13.1)
    assert timing["started_at"] == 13.1 and timing["deadline"] == 15.1
    assert motor.learning_metadata["integration_dt"] == 0.02  # no preparation time catch-up
    motor.step(simulated_body(current, 15.09), choice, 15.09, 0.02)
    assert motor.learning_metadata["integration_dt"] == pytest.approx(0.01)
    assert motor.step(simulated_body(current, 15.1), choice, 15.1, 0.02) == current
    assert motor.learning_metadata is None and motor.timing(choice, 15.1)["phase"] == "expired"


def test_posture_arrival_finishes_on_distinct_readback_and_holds(timed_motor):
    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    motor.controller.state, motor.controller.expected = states[0], current
    choice = (7, 10.0, Intent(skill="STAND", duration_s=5.0), "test")
    motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02)
    ticks = motor.controller.steps_executed
    for now in (10.05, 10.1, 10.21):
        assert motor.step(simulated_body(current, now), choice, now, 0.02) == current
    assert motor.timing(choice, 10.21)["phase"] == "completed"
    assert motor.goal_evidence(simulated_body(current, 10.21), 10.21, 7)["success"]
    completed = motor.goal_evidence(None, 12.0, 7)
    assert completed["success"] and completed["observed_at"] == 10.21
    assert completed["settled_samples"] >= 3
    assert motor.goal_evidence(None, 12.0, 8)["success"] is None
    assert motor.controller.steps_executed == ticks
    assert motor.step(simulated_body(current, 12.0), choice, 12.0, 0.02) == current
    assert motor.learning_metadata is None


def test_repeated_same_readback_cannot_complete_posture(timed_motor):
    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    motor.controller.state, motor.controller.expected = states[0], current
    choice = (7, 10.0, Intent(skill="STAND", duration_s=5.0), "test")
    motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02)
    for now in (10.1, 10.2, 10.3):
        motor.step(simulated_body(current, 10.1), choice, now, 0.02)
    assert motor.timing(choice, 10.3)["phase"] == "running"


@pytest.mark.parametrize("periodic", [False, True])
def test_complete_finite_sequence_settles_while_periodic_gait_keeps_running(timed_motor, periodic):
    from types import SimpleNamespace

    from myumiq_vrchat.articulated_tasks import ActionExecution

    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    motor.controller.state, motor.controller.expected = states[0], current
    motor.key, motor.pose_goal = 7, current
    motor.active = True
    motor.execution = ActionExecution(7, 10.0, 18.0, started_at=10.0, deadline=28.0)
    motor.motions["MOTION_TEST"] = (None, SimpleNamespace(hand=None))
    motor.playback = SimpleNamespace(
        completed=True,
        target=lambda: current,
        observe=lambda *args, **kwargs: None,
        evidence=lambda: {"endpoint_required": not periodic, "sequence_observed": True},
    )
    choice = (7, 10.0, Intent(skill="MOTION_TEST", duration_s=18.0), "test")
    for now in (10.05, 10.1, 10.21):
        motor.step(simulated_body(current, now), choice, now, 0.02)
    assert motor.timing(choice, 10.21)["phase"] == ("running" if periodic else "completed")
    assert bool(motor.controller.steps_executed) is periodic


def test_facing_holds_without_actor_steps_during_brief_visual_loss(timed_motor):
    from test_body_facing import world

    from myumiq_vrchat.body import WorldState

    motor, states, _ = timed_motor
    motor.settings = motor.settings.model_copy(update={"facing": True})
    control = motor.controller
    current = control.rig.forward(states[0])
    control.state, control.expected = states[0], current
    choice = (7, 10.0, Intent(skill="LOOK_AT", target="person", duration_s=5.0), "test")
    motor.world = world(10.0)
    motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02)
    count = control.steps_executed
    assert count > 0
    motor.world = WorldState(timestamp=10.2)
    assert motor.step(simulated_body(current, 10.2), choice, 10.2, 0.02) == current
    assert control.steps_executed == count and motor.learning_metadata is None
    assert motor.error is None
    np.testing.assert_array_equal(control.previous, np.zeros(66))
    motor.world = world(10.5)
    motor.step(simulated_body(current, 10.5), choice, 10.5, 0.02)
    assert control.steps_executed == count + 1 and motor.error is None
    # Losing the same target for longer terminates; another target is not substituted.
    motor.world = WorldState(timestamp=11.0)
    assert motor.step(simulated_body(current, 11.0), choice, 11.0, 0.02) == current
    assert motor.step(simulated_body(current, 12.6), choice, 12.6, 0.02) == current
    assert motor.timing(choice, 12.6)["phase"] == "failed"


def test_facing_completes_on_fresh_images_and_body_settling_then_holds(timed_motor):
    from test_body_facing import world

    from myumiq_vrchat.body import WorldState

    motor, states, _ = timed_motor
    motor.settings = motor.settings.model_copy(update={"facing": True})
    control = motor.controller
    current = control.rig.forward(states[0])
    control.state, control.expected = states[0], current
    choice = (7, 10.0, Intent(skill="LOOK_AT", target="person", duration_s=5.0), "test")
    motor.world = world(10.0, 0.01)
    motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02)
    for now in (10.1, 10.2, 10.3):
        motor.step(simulated_body(current, now), choice, now, 0.02)
    assert motor.timing(choice, 10.3)["phase"] == "running"  # one image is insufficient
    motor.world = world(10.5, 0.01)
    for now in (10.5, 10.6, 10.7):
        motor.step(simulated_body(current, now), choice, now, 0.02)
    assert motor.timing(choice, 10.7)["phase"] == "completed"
    motor.world = WorldState(timestamp=11.0)
    evidence = motor.goal_evidence(None, 11.0, 7)
    assert evidence["success"] and evidence["scope"] == "fresh_visual_body_heading"
    assert evidence["centred_frames"] == 2 and not evidence["avatar_verified"]
    assert motor.step(simulated_body(current, 11.0), choice, 11.0, 0.02) == current
    assert motor.learning_metadata is None


@pytest.mark.parametrize("cancel", [False, True])
def test_preparation_timeout_or_cancel_cannot_apply_late_result(timed_motor, cancel):
    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    choice = (7, 10.0, Intent(skill="SIT", duration_s=2.0), "test")
    motor.step(simulated_body(current, 10.0), choice, 10.0, 0.02)
    cancellation = motor.controller.pending[3]
    if cancel:
        motor.hold()
    now = 11.0 if cancel else 18.0
    assert motor.step(simulated_body(current, now), choice, now, 0.02) == current
    assert motor.timing(choice, now)["phase"] == ("cancelled" if cancel else "failed")
    assert cancellation.is_set()
    fits[0][0].set_result((states[0], {}))
    motor.step(simulated_body(current, now + 0.1), choice, now + 0.1, 0.02)
    assert motor.learning_metadata is None and motor.execution.started_at is None
    assert len(fits) == 1 and motor.controller.steps_executed == 0


def test_replacement_does_not_reuse_old_fit_or_timing(timed_motor):
    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    old = (7, 10.0, Intent(skill="SIT", duration_s=2.0), "test")
    new = (8, 11.0, Intent(skill="STAND", duration_s=3.0), "test")
    motor.step(simulated_body(current, 10.0), old, 10.0, 0.02)
    cancellation = motor.controller.pending[3]
    motor.step(simulated_body(current, 11.0), new, 11.0, 0.02)
    assert cancellation.is_set() and len(fits) == 1
    fits[0][0].set_result((states[0], {}))
    motor.step(simulated_body(current, 11.1), new, 11.1, 0.02)
    assert len(fits) == 2 and motor.execution.started_at is None
    fits[1][0].set_result((states[0], {}))
    motor.step(simulated_body(current, 11.2), new, 11.2, 0.02)
    assert motor.execution.generation == 8 and motor.execution.deadline == 14.2
    assert motor.learning_metadata["integration_dt"] == 0.02


@pytest.mark.parametrize("cancel", [False, True])
def test_executive_reports_cancel_or_timeout_without_claiming_completion(
    timed_motor, tmp_path, cancel
):
    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    owner = AutonomousBody(
        AutonomousConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            memory=tmp_path / "legacy.json",
            purpose=PurposeSettings(state=tmp_path / "purpose.json"),
        ),
        tmp_path,
        current,
        None,
    )
    owner.learned_motor = motor
    owner.enable(True)
    runner = PurposeRunner(owner, Services())
    try:
        runner.accept(
            Purpose(
                description="stand",
                reason="test",
                success_description="standing",
                steps=(SkillRequest(capability="STAND", duration_s=2.0),),
            ),
            10.0,
            "body_decision",
        )
        intent = Intent(skill="STAND", duration_s=2.0)
        owner._choose(10.0, intent, "body_decision:test")
        runner.running = {
            "intent": intent,
            "capability": "STAND",
            "started": 10.0,
            "intent_generation": owner.choice[0],
            "start": {},
            "movement": 0.0,
            "observations": 3,
            "deferred": False,
        }
        owner.step(simulated_body(current, 10.0), 10.0, 0.02)
        assert not runner._observe_step(13.0)
        if cancel:
            motor.hold()
        owner.snapshot = simulated_body(current, 18.0)
        # Even if motor frames stop, the executive uses the same bounded preparation deadline.
        assert runner._observe_step(18.0)
        result = runner.history[-1]
        assert result["status"] == ("interrupted" if cancel else "execution_failed")
        assert result["evidence"]["success"] is (None if cancel else False)
        assert result["evidence"]["execution"]["error"] == (
            None if cancel else "articulated preparation timed out"
        )
        assert result["action"]["execution"]["started_at"] is None
    finally:
        runner.close()


def test_pulse_test_exploration_waits_for_gait_preparation_and_releases_controller_inputs(
    timed_motor, tmp_path
):
    from myumiq_vrchat.body import Controls
    from myumiq_vrchat.exploration import ExplorationSettings
    from myumiq_vrchat.motion_prior import FiniteImitation, MotionReference
    from myumiq_vrchat.postures import posture_target

    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    motion = FiniteImitation.fit([(float(t), current) for t in np.linspace(0, 1, 32)], 1.0, "gait")
    motor.install_motion(
        "WALK_IN_PLACE", motion, MotionReference(policy=tmp_path / "motion", sha256="a" * 64)
    )
    owner = AutonomousBody(
        AutonomousConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            memory=tmp_path / "memory",
            purpose=PurposeSettings(state=tmp_path / "purpose.json"),
            exploration=ExplorationSettings(enabled=True, mode="pulse_test"),
        ),
        tmp_path,
        current,
        None,
    )
    owner.learned_motor = motor
    owner.enable(True)
    owner._choose(10.0, Intent(skill="EXPLORE_HOME", duration_s=2.0), "test")
    owner.exploration_lease = (owner.generation, 14.0, "forward")
    owner.exploration_status = {"visual_frame_at": 10.0}
    assert owner.step(simulated_body(current, 10.0), 10.0, 0.02).left.controls == Controls()
    fits[0][0].set_result((states[0], {}))
    owner.exploration_status = {"visual_frame_at": 13.0}
    moved = owner.step(simulated_body(current, 13.0), 13.0, 0.02)
    assert (
        owner.action_timing(13.0)["deadline"] == 15.0
    )  # preparation did not consume movement duration
    assert moved.left.controls.sticks[1] == (0.0, 0.45)
    assert moved.head == current.head and moved.pelvis == current.pelvis
    owner.generation += 1
    assert owner.step(simulated_body(moved, 13.02), 13.02, 0.02).left.controls == Controls()
    owner.exploration_status = {"visual_frame_at": 12.0}
    held = owner.step(simulated_body(current, 13.1), 13.1, 0.02)
    assert held == current
    playback = motor.playback
    phase = playback.phase
    steps = motor.controller.steps_executed
    assert owner.action_timing(13.1)["phase"] == "waiting_observation"
    assert owner.learning_metadata is None
    assert owner.intent_metadata["body_state"] == "holding_without_exploration_image"
    # A delayed readback must not replace the latched pose while vision is missing.
    assert owner.step(simulated_body(posture_target("crouching"), 13.2), 13.2, 0.02) == held
    assert motor.controller.steps_executed == steps and playback.phase == phase
    owner.exploration_status = {"visual_frame_at": 13.3}
    owner.step(simulated_body(current, 13.3), 13.3, 0.02)
    assert motor.playback is playback and not motor.execution.cancelled
    assert owner.action_timing(13.3)["phase"] == "running"
    assert owner.action_timing(13.3)["deadline"] == 15.0
    # A prolonged camera outage does not extend the original finite deadline.
    owner.exploration_status = {"visual_frame_at": None}
    owner.step(simulated_body(current, 14.0), 14.0, 0.02)
    assert owner.action_timing(14.0)["phase"] == "waiting_observation"
    assert owner.action_timing(15.1)["phase"] == "expired"


def test_automatic_new_motion_acquisition_does_not_replace_body_choice(
    timed_motor, tmp_path, monkeypatch
):
    from concurrent.futures import Future

    from myumiq_vrchat.capability_learning import LearningSettings, MotionLesson
    from myumiq_vrchat.motion_prior import FiniteImitation

    motor, states, _ = timed_motor
    current = motor.controller.rig.forward(states[0])
    asset, license = tmp_path / "motion.glb", tmp_path / "license.txt"
    asset.touch()
    license.write_text("CC0")
    owner = AutonomousBody(
        AutonomousConfig(
            llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            memory=tmp_path / "memory",
            purpose=PurposeSettings(
                state=tmp_path / "purpose.json",
                learning=LearningSettings(
                    acquire_configured_motions=True,
                    motions={
                        "MOTION_DANCE": MotionLesson(
                            clip="Dance",
                            source=asset,
                            license=license,
                            source_url="https://example.org",
                            description="踊る",
                        )
                    },
                ),
            ),
        ),
        tmp_path,
        current,
        None,
    )
    owner.learned_motor = motor
    owner.enable(True)
    runner = PurposeRunner(owner, Services())
    future = Future()
    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", lambda fn: future)
    choice = owner.choice
    try:
        runner._learning()
        assert runner.tasks[-1]["capability"] == "MOTION_DANCE"
        assert runner.tasks[-1]["status"] == "running"
        assert not runner.registry.is_available("MOTION_DANCE") and owner.choice == choice
        motion = FiniteImitation.fit(
            [(float(t), current) for t in np.linspace(0, 1, 32)], 1.0, "Dance"
        )
        p = tmp_path / "learning" / runner.tasks[-1]["id"] / "policy.json"
        p.parent.mkdir(parents=True)
        motion.save(p)
        future.set_result(
            (
                motion,
                {
                    "policy": str(p),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "description": "踊る",
                    "playback_rate": 0.25,
                },
            )
        )
        runner._learning()
        assert runner.tasks[-1]["status"] == "trained"
        assert runner.registry.is_available("MOTION_DANCE") and owner.choice == choice
        assert "MOTION_DANCE" in motor.motions and len(runner.tasks) == 1
    finally:
        runner.close()


def test_new_motion_intents_share_origin_without_resetting_current_pose(timed_motor, tmp_path):
    from myumiq_vrchat.motion_prior import FiniteImitation, MotionReference
    from myumiq_vrchat.whole_body import target_from_vector

    motor, states, fits = timed_motor
    current = motor.controller.rig.forward(states[0])
    motion = FiniteImitation.fit([(float(t), current) for t in np.linspace(0, 1, 32)], 1.0, "gait")
    motor.install_motion(
        "WALK_IN_PLACE", motion, MotionReference(policy=tmp_path / "motion", sha256="a" * 64)
    )
    motor.step(
        simulated_body(current, 10.0), (7, 10.0, Intent(skill="WALK_IN_PLACE"), "test"), 10.0, 0.02
    )
    origin = motor.playback.target().pelvis.position[:2]
    p = vector(current)
    p[:, 0] += 0.08
    shifted = target_from_vector(p)
    held = motor.step(
        simulated_body(shifted, 11.0), (8, 11.0, Intent(skill="WALK_IN_PLACE"), "test"), 11.0, 0.02
    )
    np.testing.assert_allclose(motor.playback.target().pelvis.position[:2], origin, atol=1e-8)
    assert held == shifted  # preparation holds feedback; only the goal uses the origin
