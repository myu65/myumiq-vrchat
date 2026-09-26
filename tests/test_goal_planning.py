import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from myumiq_vrchat.agent_memory import AgentMemory
from myumiq_vrchat.autonomy import Drives
from myumiq_vrchat.body import WorldState
from myumiq_vrchat.capabilities import CapabilityRegistry
from myumiq_vrchat.generation_config import LLMConfig
from myumiq_vrchat.goal_planning import GoalPlanning
from myumiq_vrchat.purpose_runtime import PurposeSettings
from myumiq_vrchat.purposes import Purpose, SkillRequest


@pytest.fixture
def planning(tmp_path, monkeypatch):
    settings = PurposeSettings(
        state=tmp_path / "state.json",
        planner=LLMConfig(base_url="http://localhost:1234", model="thought"),
    )
    memory = AgentMemory(tmp_path / "memory.sqlite3")
    memory.working["plan"] = {"id": "existing-action", "status": "active"}
    owner = SimpleNamespace(
        generation=1,
        control_generation=0,
        health={},
        world=WorldState(),
        drives=Drives(),
        snapshot=None,
        memory=SimpleNamespace(recent=[]),
    )
    emitted, jobs = [], []
    runner = SimpleNamespace(
        owner=owner,
        settings=settings,
        dialogue_epoch=0,
        goal_context_revision=0,
        shared=memory,
        registry=CapabilityRegistry(walking=False),
        history=[],
        emit=lambda event, **data: emitted.append((event, data)),
    )

    def background(fn):
        future = Future()
        jobs.append((future, fn))
        return future

    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", background)
    yield GoalPlanning(runner), runner, jobs, emitted
    memory.close()


def proposal():
    return Purpose(
        description="周りを観察する",
        reason="変化を確認する",
        success_description="状態を確かめる",
        steps=(SkillRequest(capability="WAIT"),),
    )


def test_planner_gets_state_and_history_but_never_takes_motor_ownership(planning, monkeypatch):
    planner, runner, jobs, emitted = planning
    runner.shared.hear("少し待って")
    seen = []

    def infer(config, world, drives, body, recent, capabilities, history, context):
        seen.append((config.model, context["working"]["turns"][-1]["text"]))
        return proposal()

    monkeypatch.setattr("myumiq_vrchat.goal_planning.request_purpose", infer)
    now = time.perf_counter()
    planner.tick(now)
    planner.tick(now + 1)
    assert len(jobs) == 1 and runner.shared.commitment is None
    jobs[0][0].set_result(jobs[0][1]())
    planner.tick(now + 2)
    assert seen == [("thought", "少し待って")]
    assert runner.shared.commitment["description"] == proposal().description
    assert runner.shared.working["plan"] == {"id": "existing-action", "status": "active"}
    assert runner.owner.generation == 1 and emitted[-1][0] == "planner_proposal"
    planner.tick(now + 3)
    assert len(jobs) == 1


@pytest.mark.parametrize(
    "change", ["conversation", "manual", "commitment", "expired", "failure", "unknown_capability"]
)
def test_stale_or_failed_thought_does_not_replace_current_goal(planning, change):
    planner, runner, jobs, emitted = planning
    runner.shared.propose_goal(proposal())
    original = dict(runner.shared.commitment)
    now = time.perf_counter()
    planner.tick(now)
    if change == "conversation":
        runner.goal_context_revision += 1
    elif change == "manual":
        runner.owner.control_generation += 1
    elif change == "commitment":
        runner.shared.commitment["status"] = "suspended"
        original = dict(runner.shared.commitment)
    if change == "failure":
        jobs[0][0].set_exception(RuntimeError("provider unavailable"))
    else:
        value = proposal()
        if change == "unknown_capability":
            value = value.model_copy(update={"steps": (SkillRequest(capability="UNKNOWN"),)})
        jobs[0][0].set_result(value)
    planner.tick(now + (31 if change == "expired" else 1))
    assert runner.shared.commitment == original
    assert runner.shared.working["plan"]["id"] == "existing-action"

    assert runner.owner.health["planner"]["state"] in ("discarded", "unavailable")


def test_body_progress_and_speech_onset_do_not_starve_planning(planning):
    planner, runner, jobs, emitted = planning
    now = time.perf_counter()
    planner.tick(now)
    runner.owner.generation += 3  # Independent finite motor actions.
    runner.dialogue_epoch += 2  # Voice onset/cancel without a new recognized directive.
    jobs[0][0].set_result(proposal())
    planner.tick(now + 1)
    assert runner.owner.health["planner"]["state"] == "proposed"
    assert runner.shared.commitment["description"] == proposal().description


def test_supported_learning_proposal_keeps_body_and_commitment(planning):
    planner, runner, jobs, emitted = planning
    runner.shared.propose_goal(proposal())
    original = dict(runner.shared.commitment)
    queued = []
    runner.queue_learning = lambda purpose: queued.append(purpose) or ["WALK_IN_PLACE"]
    now = time.perf_counter()
    planner.tick(now)
    value = proposal().model_copy(update={"steps": (SkillRequest(capability="WALK_IN_PLACE"),)})
    jobs[0][0].set_result(value)
    planner.tick(now + 1)
    assert queued == [value]
    assert runner.owner.health["planner"]["state"] == "learning_queued"
    assert runner.shared.commitment == original and runner.owner.generation == 1
    assert runner.shared.working["plan"]["id"] == "existing-action"
