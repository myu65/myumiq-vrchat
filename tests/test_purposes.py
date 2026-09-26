import json
import time

import pytest

from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig
from myumiq_vrchat.body import WorldObject, WorldState, simulated_body
from myumiq_vrchat.capabilities import CapabilityRegistry
from myumiq_vrchat.capability_learning import LearningSettings, make_task
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.purpose_runtime import PurposeRunner, PurposeSettings
from myumiq_vrchat.purposes import Purpose, SkillRequest


def purpose(*steps):
    return Purpose(
        description="自分の目的を実現する",
        reason="今の状況から必要",
        success_description="結果を観測する",
        steps=tuple(SkillRequest(**s) for s in steps),
    )


def test_configured_motion_is_discovered_by_planner_and_body_after_acquisition(tmp_path):
    from myumiq_vrchat.capability_learning import MotionLesson
    from myumiq_vrchat.embodied_decision import candidates
    from myumiq_vrchat.purposes import purpose_schema

    source, license = tmp_path / "asset.glb", tmp_path / "LICENSE"
    source.touch()
    license.write_text("CC0")
    settings = LearningSettings(
        motions={
            "MOTION_DANCE": MotionLesson(
                clip="Dance",
                description="踊る",
                source=source,
                license=license,
                source_url="https://example.org/source",
            )
        }
    )
    registry = CapabilityRegistry()
    item = registry.get("MOTION_DANCE")
    item.method, item.description = "imitation", "踊る"
    task = make_task(item.name, None, registry, settings)
    assert task["status"] == "queued"
    goal = purpose({"capability": item.name})
    assert registry.resolve(goal, WorldState()).missing == [item.name]
    schema = purpose_schema(WorldState(), registry.summary())
    assert any(
        v["properties"]["capability"].get("const") == item.name
        for v in schema["$defs"]["SkillRequest"]["anyOf"]
    )
    item.available = True
    assert registry.resolve(goal, WorldState()).actions[0][1].skill == item.name
    choice = next(c for c in candidates(registry, WorldState()) if c.id == "body_MOTION_DANCE")
    assert choice.description == "踊る"


def test_open_goal_and_missing_capability_are_not_arbitrary_actuation():
    registry = CapabilityRegistry()
    goal = purpose({"capability": "INVENT_NEW_MOTION"})
    result = registry.resolve(goal, WorldState())
    assert result.route == "learning" and result.actions == []
    assert result.missing == ["INVENT_NEW_MOTION"]
    with pytest.raises(ValueError):
        SkillRequest(capability="WAVE", coordinates=[1, 2, 3])
    with pytest.raises(ValueError):
        SkillRequest(capability="powershell.exe")


def test_handshake_missing_prerequisites_never_executes_reach_prefix():
    registry = CapabilityRegistry()
    world = WorldState(
        objects=(WorldObject(name="hand", position=(0.3, -0.2, 1.3), source="fixture"),)
    )
    goal = purpose(
        {"capability": "REACH", "target": "hand", "hand": "right"}, {"capability": "HANDSHAKE"}
    )
    result = registry.resolve(goal, world)
    assert not result.actions and "HANDSHAKE" in result.missing
    assert "CONTACT_FEEDBACK" in result.blockers
    task = make_task("HANDSHAKE", "g", registry, LearningSettings())
    assert task["status"] == "blocked" and task["method"] == "imitation"


def test_composition_availability_tracks_nested_dependencies_and_adapter_replacement():
    registry = CapabilityRegistry()
    world = WorldState(
        objects=(WorldObject(name="hand", position=(0.3, -0.2, 1.3), source="fixture"),)
    )
    goal = purpose({"capability": "HOLD_HAND", "target": "hand", "hand": "right"})
    reach = registry.get("REACH")
    reach.prerequisites = ("HAND_TRACKER",)
    registry.get("HAND_TRACKER").available = False

    def summary():
        return {r["name"]: r for r in registry.summary()}

    assert not summary()["HOLD_HAND"]["available"]
    assert summary()["HOLD_HAND"]["unavailable_prerequisites"] == ["REACH"]
    assert registry.resolve(goal, world).actions == []
    registry.get("HAND_TRACKER").available = True
    assert summary()["HOLD_HAND"]["available"]
    assert registry.resolve(goal, world).actions[0][1].skill == "REACH"
    reach.available = False  # switching to a body adapter without reach
    assert not summary()["HOLD_HAND"]["available"]
    assert registry.resolve(goal, world).missing == ["HOLD_HAND"]


def test_unknown_or_cyclic_dependencies_are_unavailable_without_mutating_inventory():
    registry = CapabilityRegistry()
    registry.get("REACH").prerequisites = ("HOLD_HAND",)
    registry.get("LOOK_AT").prerequisites = ("UNKNOWN_SENSOR",)
    before = set(registry.items)
    rows = {r["name"]: r for r in registry.summary()}
    assert not rows["REACH"]["available"] and not rows["HOLD_HAND"]["available"]
    assert not rows["EXPLORE"]["available"]
    assert set(registry.items) == before
    result = registry.resolve(purpose({"capability": "EXPLORE"}), WorldState())
    assert result.actions == [] and result.missing == ["EXPLORE"]
    assert result.blockers == ["LOOK_AT"]


def test_exploration_composition_and_context_requirements():
    registry = CapabilityRegistry()
    world = WorldState(
        objects=tuple(
            WorldObject(
                name=n,
                position=(1, 0, 1),
                source="vision",
                image_position=(0.2, 0.1),
                last_seen=1,
                confidence=0.8,
            )
            for n in ("a", "b")
        ),
        timestamp=1,
    )
    registry.seen["a"] = 2
    result = registry.resolve(purpose({"capability": "EXPLORE"}), world)
    assert result.route == "composition"
    assert [intent.target for _, intent in result.actions] == ["b", "a"]
    assert (
        registry.resolve(
            purpose({"capability": "REACH", "target": "a", "hand": "right"}), world
        ).route
        == "blocked"
    )
    assert registry.resolve(purpose({"capability": "APPROACH"}), world).route == "learning"


def test_unknown_evidence_not_success_and_availability_not_restored():
    registry = CapabilityRegistry()
    registry.observe("SIT", None, "unavailable")
    assert registry.get("SIT").summary()["success_rate"] is None
    registry.observe("SIT", True, "simulated")
    assert registry.get("SIT").summary()["outcomes_by_scope"]["simulated"]["success"] == 1
    saved = registry.dump()
    saved["HANDSHAKE"]["available"] = True
    registry.restore_statistics(saved)
    assert not registry.get("HANDSHAKE").available


def test_planner_and_exploration_exclude_unusable_visual_feedback():
    from myumiq_vrchat.purposes import purpose_schema

    good = WorldObject(
        name="good",
        position=(1, 0, 1.6),
        source="vision",
        image_position=(0.2, 0.1),
        confidence=0.8,
        last_seen=10,
    )
    world = WorldState(
        timestamp=10,
        objects=(
            good,
            good.model_copy(update={"name": "weak", "confidence": 0.2}),
            good.model_copy(update={"name": "old", "last_seen": 8}),
        ),
    )
    variants = purpose_schema(world)["$defs"]["SkillRequest"]["anyOf"]
    look = next(v for v in variants if v["properties"]["capability"].get("const") == "LOOK_AT")
    assert look["properties"]["target"]["enum"] == ["good"]
    registry = CapabilityRegistry()
    result = registry.resolve(purpose({"capability": "EXPLORE"}), world)
    assert [i.target for _, i in result.actions] == ["good"]
    assert (
        registry.resolve(purpose({"capability": "LOOK_AT", "target": "weak"}), world).route
        == "blocked"
    )
    assert (
        registry.resolve(purpose({"capability": "LOOK_AT", "target": "old"}), world).route
        == "blocked"
    )


class Services:
    voice = vision = None

    def speak(self, *args):
        pass


def runner(tmp_path):
    cfg = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "memory.json",
        purpose=PurposeSettings(state=tmp_path / "purposes.json"),
    )
    owner = AutonomousBody(cfg, tmp_path, posture_target("standing"), None)
    owner.enable(True)
    return owner, PurposeRunner(owner, Services())


def test_multi_step_motor_observations_update_memory_and_capabilities(tmp_path):
    owner, executive = runner(tmp_path)
    now = time.perf_counter()
    executive.accept(
        purpose({"capability": "SIT", "duration_s": 5}, {"capability": "STAND", "duration_s": 5}),
        now,
    )
    body = simulated_body(owner.rest, now)
    for _ in range(900):
        now += 0.02
        body = simulated_body(owner.step(body, now, 0.02), now)
        executive.tick(now)
        if executive.history:
            break
        time.sleep(0.0005)
    assert executive.history[-1]["status"] == "plan_completed"
    assert executive.registry.get("SIT").successes == 1
    assert executive.registry.get("STAND").successes == 1
    assert owner.memory.recent[-1]["goal_id"]
    saved = json.loads((tmp_path / "purposes.json").read_text("utf-8"))
    assert saved["capabilities"]["SIT"]["successes"] == 1
    assert "purpose" in owner.intent_metadata
    executive.close()


def test_manual_interrupt_invalidates_plan_without_resuming_it(tmp_path):
    owner, executive = runner(tmp_path)
    now = time.perf_counter()
    executive.accept(purpose({"capability": "SIT"}, {"capability": "STAND"}), now)
    owner.enable(False)
    executive.tick(now + 0.01)
    assert executive.goal is None
    assert executive.history[-1]["status"] == "interrupted"
    restored = PurposeRunner(owner, Services())
    assert restored.goal is None
    executive.close()


def test_look_at_outcome_uses_gaze_geometry_without_stopping_executive(tmp_path):
    owner, executive = runner(tmp_path)
    now = time.perf_counter()
    owner.world = WorldState(
        objects=(WorldObject(name="person", position=(1, 0, 1.6), source="fixture"),)
    )
    executive.accept(purpose({"capability": "LOOK_AT", "target": "person", "duration_s": 1}), now)
    for _ in range(100):
        now += 0.02
        body = simulated_body(owner.rest, now)
        owner.snapshot = body
        executive.tick(now)
        if executive.history:
            break
        time.sleep(0.001)
    assert executive.history[-1]["status"] == "plan_completed"
    assert executive.registry.get("LOOK_AT").successes == 1
    executive.close()


def test_learning_routes_and_deduplication(tmp_path):
    owner, executive = runner(tmp_path)
    g = purpose({"capability": "HANDSHAKE"})
    executive.accept(g, 10)
    executive.accept(g, 20)
    assert len(executive.tasks) == 1 and executive.tasks[0]["status"] == "blocked"
    assert (
        make_task("SIT", "g", executive.registry, LearningSettings())["method"]
        == "reference_free_rl"
    )
    assert (
        make_task("REACH", "g", executive.registry, LearningSettings())["method"] == "vrchat_replay"
    )
    executive.close()


def test_late_llm_result_after_manual_preemption_is_discarded(tmp_path):
    from concurrent.futures import Future

    owner, executive = runner(tmp_path)
    future = Future()
    executive.pending = (future, "purpose", owner.generation)
    owner.enable(False)
    executive.tick(time.perf_counter())
    future.set_result(purpose({"capability": "WAVE", "hand": "right"}))
    owner.enable(True)
    executive.tick(time.perf_counter())
    assert executive.goal is None and executive.running is None
    assert owner.choice[2].skill == "WAIT"
    executive.close()


def test_generated_schema_allows_learning_but_never_unknown_actuation():
    from myumiq_vrchat.purposes import purpose_schema

    schema = purpose_schema(WorldState())
    variants = schema["$defs"]["SkillRequest"]["anyOf"]
    known = {v["properties"]["capability"].get("const"): v for v in variants}
    assert "LOOK_AT" not in known and "REACH" not in known
    assert known["SIT"]["properties"]["learn"] == {"type": "boolean"}
    assert known[None]["properties"]["learn"]["const"] is True


def test_planner_focus_cannot_invent_a_name_from_an_image(monkeypatch):
    from myumiq_vrchat.autonomy import Drives
    from myumiq_vrchat.purposes import purpose_schema, request_purpose

    world = WorldState(
        objects=(WorldObject(name="known-target", source="fixture", position=(1.0, 0.0, 1.0)),)
    )
    caps = [{"name": "WAIT"}]
    assert set(purpose_schema(world, caps)["properties"]["focus"]["enum"]) == {
        "known-target",
        "WAIT",
        None,
    }

    def reply(config, messages, schema, *args):
        assert "invented poster name" not in schema["properties"]["focus"]["enum"]
        return purpose({"capability": "WAIT"}).model_copy(
            update={"focus": "invented poster name"}
        ).model_dump(), 0.1

    monkeypatch.setattr("myumiq_vrchat.purposes._request", reply)
    with pytest.raises(ValueError, match="supplied target"):
        request_purpose(
            LLMConfig(base_url="http://localhost:1", model="test"),
            world,
            Drives(),
            {},
            [],
            caps,
            [],
        )
