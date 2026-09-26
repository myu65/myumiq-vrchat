from concurrent.futures import Future

import pytest

from myumiq_vrchat.autonomy import AutonomousPlanner, Drives, InteractionMemory
from myumiq_vrchat.body import WorldState, rest_target, simulated_body
from myumiq_vrchat.cognition import Decision, Goal, LLMConfig
from myumiq_vrchat.replay import Observation, device_progress_reward


def test_autonomous_planner_waits_for_llm_then_replans(monkeypatch):
    requests = []
    future = Future()

    def decide(config, instruction, world):
        requests.append(instruction)
        return future

    monkeypatch.setattr("myumiq_vrchat.autonomy.decide_async", decide)
    planner = AutonomousPlanner(LLMConfig(base_url="http://localhost:1/v1", model="fixture"))
    world = WorldState()
    assert planner.tick(0.0, world, dt=0.01).goal.skill == "WAIT"
    assert planner.tick(0.1, world, dt=0.01) is None
    assert planner.drives.boredom > 0
    assert len(requests) == 1
    future.set_result(
        Decision(goal=Goal(skill="WAVE", hand="right", duration_s=0.2), source="local_llm")
    )
    assert planner.tick(0.2, world, dt=0.01).goal.skill == "WAVE"
    planner.observe_reward(0.4)
    assert planner.tick(0.3, world, dt=0.01) is None
    assert planner.tick(0.6, world, dt=0.01).goal.skill == "WAIT"
    assert planner.memory.recent[-1]["reward"] == 0.4
    assert "最近の経験" in requests[-1]


def test_drive_bounds_and_device_reward_provenance():
    drives = Drives()
    for _ in range(1000):
        drives.advance(1, interacting=True)
    assert 0 <= drives.fatigue <= 1
    with pytest.raises(ValueError):
        drives.advance(2, interacting=False)
    target = rest_target()
    old = Observation(timestamp=0, world=WorldState(), body=simulated_body(target, 0))
    new = Observation(timestamp=1, world=WorldState(), body=simulated_body(target, 1))
    assert device_progress_reward(old, target, new) == 0
    missing = new.model_copy(
        update={
            "body": new.body.model_copy(
                update={"head": new.body.head.model_copy(update={"valid": False})}
            )
        }
    )
    assert device_progress_reward(old, target, missing) is None


def test_interaction_memory_round_trip(tmp_path):
    path = tmp_path / "memory.json"
    memory = InteractionMemory()
    memory.record_interaction("person-1", positive=True)
    memory.record(
        Decision(goal=Goal(skill="WAVE", hand="right", duration_s=1), source="fixed"),
        "observed_progress",
        0.2,
    )
    memory.save(path)
    loaded = InteractionMemory.load(path)
    assert loaded.familiarity == {"person-1": 0.05}
    assert loaded.affinity == {"person-1": 0.05}
    assert loaded.recent[-1]["skill"] == "WAVE"


def test_neutral_contact_does_not_invent_affinity():
    memory = InteractionMemory()
    memory.record_interaction("person")
    assert memory.familiarity["person"] == 0.05
    assert memory.affinity == {}
    memory.record_interaction("person", positive=False)
    assert memory.familiarity["person"] == 0.1
    assert memory.affinity["person"] == -0.05
