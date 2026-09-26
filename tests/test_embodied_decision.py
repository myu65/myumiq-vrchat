import time
from concurrent.futures import Future

import pytest

from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig, Intent
from myumiq_vrchat.body import WorldObject, WorldState, simulated_body
from myumiq_vrchat.capabilities import CapabilityRegistry
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.decision import BackendConfig, CandidateScore, ScoreResult
from myumiq_vrchat.decision_runtime import DecisionSettings
from myumiq_vrchat.decision_selection import SelectionResult, SelectionSettings
from myumiq_vrchat.embodied_decision import candidates, score
from myumiq_vrchat.events import RuntimeEvent
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.purpose_runtime import PurposeRunner, PurposeSettings
from myumiq_vrchat.shared_dialogue import DialogueSettings


class Services:
    voice = vision = None

    def reply(self, *args):
        raise AssertionError("speech is deliberately pending")


@pytest.mark.parametrize("text", ["今日は暑いね", "何が見える？", "手を振って"])
def test_default_dialogue_starts_without_body_assessment(runtime, monkeypatch, text):
    from myumiq_vrchat.shared_dialogue import Dialogue

    owner, runner, jobs = runtime
    owner.config = owner.config.model_copy(
        update={
            "decision": DecisionSettings(
                selection=SelectionSettings(
                    llm=LLMConfig(base_url="http://localhost:2", model="thought"),
                    understand_requests=True,
                )
            )
        }
    )
    seen = []

    def respond(*args, **kwargs):
        seen.append(kwargs["action_context"])
        return Dialogue(reply="うん、聞いているよ。", topic="会話")

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue.request_dialogue", respond)
    now = time.perf_counter()
    runner.on_event(RuntimeEvent("utterance", now, text=text), now)
    runner._dialogue_tick(now)
    assert len(jobs) == 1 and runner.body_decision.request_assessment is None
    jobs[0][1]()
    assert seen[0]["status"] == "parallel_pending"


def test_new_reply_starts_before_old_generation_finishes_and_inputs_survive(runtime, monkeypatch):
    from myumiq_vrchat.shared_dialogue import Dialogue

    owner, runner, jobs = runtime
    now = time.perf_counter()
    monkeypatch.setattr(
        "myumiq_vrchat.shared_dialogue.request_dialogue",
        lambda *args, **kwargs: Dialogue(reply="新しい話だね。", topic="新しい話"),
    )
    spoken = []
    monkeypatch.setattr(runner.services, "reply", lambda text, at: spoken.append(text) or True)
    runner.on_event(RuntimeEvent("utterance", now, text="古い話"), now)
    runner._dialogue_tick(now)
    old = runner.dialogue_generation
    runner.on_event(RuntimeEvent("utterance", now + 0.1, text="新しい話"), now + 0.1)
    runner._dialogue_tick(now + 0.1)
    assert old.cancelled.is_set() and len(jobs) == 2 and not jobs[0][0].done()
    jobs[1][0].set_result(jobs[1][1]())
    runner._dialogue_tick(now + 0.2)
    jobs[0][0].set_result(Dialogue(reply="古い返答だよ。", topic="古い話"))
    runner._dialogue_tick(now + 0.3)
    assert spoken == ["新しい話だね。"]
    assert [t["text"] for t in runner.shared.working["turns"] if t["role"] == "user"] == [
        "古い話",
        "新しい話",
    ]


def test_repeated_preemption_bounds_noncooperative_extensions(runtime):
    _, runner, jobs = runtime
    now = time.perf_counter()
    for index in range(6):
        runner.on_event(RuntimeEvent("utterance", now + index, text=f"話{index}"), now + index)
        runner._dialogue_tick(now + index)
    assert len(jobs) == 2 and len(runner.retired_dialogues) == 2
    assert runner.utterances[0]["text"] == "話5"
    jobs[0][0].set_exception(RuntimeError("obsolete"))
    runner._dialogue_tick(now + 7)
    assert len(jobs) == 3
    assert runner.reply_to["text"] == "話5"


@pytest.mark.parametrize("preempt", [False, True])
def test_streamed_sentences_wait_for_audio_and_do_not_replay_full_reply(
    runtime, monkeypatch, preempt
):
    from types import SimpleNamespace

    from myumiq_vrchat.shared_dialogue import Dialogue

    _, runner, jobs = runtime
    busy, spoken = [False], []
    output = SimpleNamespace(stop=lambda: busy.__setitem__(0, False))
    runner.services.voice = SimpleNamespace(pipeline=SimpleNamespace(output=output))

    def reply(text, now, **kwargs):
        assert kwargs["wait_until_idle"]
        if busy[0]:
            return False
        busy[0] = True
        spoken.append(text)
        return True

    monkeypatch.setattr(runner.services, "reply", reply)
    now = time.perf_counter()
    runner.on_event(RuntimeEvent("utterance", now, text="元気？"), now)
    runner._dialogue_tick(now)
    stream = runner.reply_stream
    stream.feed('{"reply":"こんにちは。元気だよ。')
    runner._dialogue_tick(now + 0.1)
    assert spoken == ["こんにちは。"] and not jobs[0][0].done()
    runner._dialogue_tick(now + 0.2)
    assert spoken == ["こんにちは。"]
    if preempt:
        runner.on_event(RuntimeEvent("speech_started", now + 0.3), now + 0.3)
    else:
        busy[0] = False
    jobs[0][0].set_result(Dialogue(reply="こんにちは。元気だよ。", topic="挨拶"))
    runner._dialogue_tick(now + 0.4)
    runner._dialogue_tick(now + 0.5)
    expected = ["こんにちは。"] if preempt else ["こんにちは。", "元気だよ。"]
    assert spoken == expected
    replies = [t["text"] for t in runner.shared.working["turns"] if t["role"] == "assistant"]
    assert replies == ["".join(expected)]


def test_voice_attention_preserves_navigation_but_explicit_stop_is_immediate(runtime):
    owner, runner, _ = runtime
    now = time.perf_counter()
    owner._choose(now, Intent(skill="MOVE_FORWARD", duration_s=8), "operator_goal_plan")
    runner.epoch = owner.generation
    lease = owner.exploration_lease = (owner.generation, now + 0.15, "forward")
    choice = owner.choice
    runner.body_decision.pending = (Future(), owner.generation, 0, 0, None)
    for kind in ("speech_started", "speech_active", "partial_transcript", "utterance"):
        runner.on_event(RuntimeEvent(kind, now, text="こんにちは"), now)
        assert owner.choice == choice and owner.exploration_lease == lease
    runner.on_event(RuntimeEvent("utterance", now + 0.1, text="止まって"), now + 0.1)
    assert owner.choice[2].skill == "WAIT" and owner.exploration_lease is None
    assert runner.body_decision.stop_latched
    assert not runner.body_decision.pending[0].done()  # Stop did not await the model.


def test_explicit_stop_survives_chat_and_requires_new_executable_request(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    mode = ["stop", "WAIT"]

    def choose(settings, request):
        selected, report = result(request, mode[1])
        return selected, report | {
            "request_status": mode[0],
            "request_reason": "停止する",
            "basis": "latest_utterance" if mode[0] in ("stop", "action") else "observation",
        }

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.evaluate", choose)
    runner.shared.hear("じゃあテスト終わろっか")
    runner.body_decision.tick(now)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner.body_decision.tick(now + 0.1)
    assert runner.body_decision.stop_latched and owner.choice[2].skill == "WAIT"
    mode[:] = ["action", "STAND"]
    runner.body_decision.next_request = 0.0
    runner.body_decision.tick(now + 0.12)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner.body_decision.tick(now + 0.14)
    assert owner.choice[2].skill == "WAIT"  # Reinterpreting the stop is not a new request.
    runner.shared.hear("ありがとう")
    runner.body_decision.conversation_changed()
    mode[:] = ["none", "STAND"]
    runner.body_decision.tick(now + 0.2)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner.body_decision.tick(now + 0.3)
    assert owner.choice[2].skill == "WAIT" and runner.body_decision.stop_latched
    runner.shared.hear("今度は立って")
    runner.body_decision.conversation_changed()
    mode[:] = ["action", "STAND"]
    runner.body_decision.tick(now + 0.4)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner.body_decision.tick(now + 0.5)
    assert owner.choice[2].skill == "STAND" and not runner.body_decision.stop_latched
    assert runner.body_decision.request_assessment["execution"] == "accepted"


def test_unsupported_request_does_not_keep_exploring_as_if_following(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    owner._choose(now, Intent(skill="EXPLORE_HOME"), "fixture")
    heard = runner.shared.hear("ついてこれる？")
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.evaluate",
        lambda settings, request: (
            None,
            {
                "backend": "fixture",
                "captured_at": request.captured_at,
                "request_status": "unsupported",
                "request_reason": "追従はまだできない",
            },
        ),
    )
    runner.body_decision.tick(now)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner.body_decision.tick(now + 0.1)
    assert owner.choice[2].skill == "WAIT"
    assert runner.shared.working["body_request"]["utterance_id"] == heard["episode_id"]
    assert runner.shared.working["body_request"]["status"] == "unsupported"


def test_observation_churn_does_not_bypass_inference_failure_backoff(runtime):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    runner.body_decision.tick(now)
    jobs[-1][0].set_exception(RuntimeError("temporary failure"))
    runner.body_decision.tick(now + 0.1)
    for step in range(1, 5):
        owner.world = WorldState(
            objects=(WorldObject(name=str(step), source="fixture", position=(1.0, 0.0, 1.0)),)
        )
        runner.body_decision.tick(now + step)
    assert len(jobs) == 1


@pytest.mark.parametrize("assessment_arrives", [True, False])
def test_dialogue_wait_is_bounded_and_motor_does_not_wait(runtime, monkeypatch, assessment_arrives):
    owner, runner, jobs = runtime
    owner.config = owner.config.model_copy(
        update={
            "decision": DecisionSettings(
                selection=SelectionSettings(
                    llm=LLMConfig(base_url="http://localhost:2", model="thought"),
                    understand_requests=True,
                )
            ),
            "dialogue": DialogueSettings(await_body_assessment=True),
        }
    )
    now = time.perf_counter()
    heard = runner.shared.hear("ついてきて")
    runner.utterances = [heard]
    runner._dialogue_tick(now)
    assert owner.health["dialogue"]["state"] == "waiting_body_assessment" and not jobs
    owner.step(simulated_body(posture_target("standing"), now), now, 0.02)
    seen = []
    from myumiq_vrchat.shared_dialogue import Dialogue

    def respond(*args, **kwargs):
        seen.append(kwargs["action_context"])
        return Dialogue(reply="今はついていけないよ。", topic="移動")

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue.request_dialogue", respond)
    if assessment_arrives:
        runner.body_decision.assess_request(
            {"request_status": "unsupported", "request_reason": "追従未対応"},
            heard["episode_id"],
            now + 0.1,
        )
    runner._dialogue_tick(now + (0.2 if assessment_arrives else 3.1))
    assert len(jobs) == 1
    jobs[0][1]()
    assert seen[0]["status"] == ("unsupported" if assessment_arrives else "pending")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "legacy.json",
        purpose=PurposeSettings(state=tmp_path / "purpose.json"),
        decision=DecisionSettings(backend=BackendConfig(backend="http"), refresh_s=2),
    )
    owner = AutonomousBody(config, tmp_path, posture_target("standing"), None)
    owner.enable(True)
    runner = PurposeRunner(owner, Services())
    jobs = []

    def background(fn):
        f = Future()
        jobs.append((f, fn))
        return f

    monkeypatch.setattr("myumiq_vrchat.purpose_runtime.background", background)
    monkeypatch.setattr(
        "myumiq_vrchat.purpose_runtime.request_purpose",
        lambda *args: pytest.fail("chat must not plan the body"),
    )
    yield owner, runner, jobs
    runner.close()


def result(request, skill, target=None):
    chosen = next(
        c
        for c in request.candidates
        if c.intent.get("skill") == skill and c.intent.get("target") == target
    )
    report = ScoreResult(
        backend="test_separate_model",
        model="independent_weights",
        captured_at=request.captured_at,
        image_used=False,
        semantics="relevance_logit",
        scores=tuple(CandidateScore(id=c.id, score=float(c == chosen)) for c in request.candidates),
    )
    return chosen, report.model_dump(mode="json") | {"selected_id": chosen.id}


def test_idle_refresh_is_configurable_but_new_speech_wakes_immediately(runtime, monkeypatch):
    owner, runner, jobs = runtime
    owner.config = owner.config.model_copy(
        update={"decision": owner.config.decision.model_copy(update={"refresh_s": 10.0})}
    )
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.evaluate",
        lambda settings, request: (
            None,
            {"backend": "fixture", "captured_at": request.captured_at},
        ),
    )
    now = time.perf_counter()
    runner.body_decision.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.body_decision.tick(now + 0.1)
    runner.body_decision.tick(now + 9.0)
    assert len(jobs) == 1
    runner.body_decision.conversation_changed()
    runner.body_decision.tick(now + 9.1)
    assert len(jobs) == 2


def test_world_and_history_reach_separate_model_while_speech_and_motion_continue(
    runtime, monkeypatch
):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    runner.shared.hear("手を振って")
    runner.shared.reply("うん、聞いたよ", "挨拶", (), runner.shared.working["turns"][-1], True)
    runner.dialogue_pending = (Future(), "dialogue", runner.dialogue_epoch)
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(
            request,
            "WAVE" if len(requests) == 1 else "LOOK_AT",
            None if len(requests) == 1 else "new_person",
        )

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    assert owner.choice[2].skill == "WAVE" and runner.running
    assert runner.shared.commitment is None
    assert owner.purpose_metadata["goal_id"] == runner.goal_id
    assert requests[0].state["conversation"][-2]["text"] == "手を振って"
    assert requests[0].state["conversation"][-1]["role"] == "assistant"
    first_decision = owner._decision_summary(owner.choice[3])
    owner.world = WorldState(
        timestamp=now + 2.1,
        objects=(
            WorldObject(
                name="new_person", source="fixture", kind="player", position=(1.0, 0.5, 1.6)
            ),
        ),
    )
    runner.tick(now + 2.1)
    assert runner.running and len(jobs) == 2 and not runner.dialogue_pending[0].done()
    # New scoring must not replace metadata of the action already running.
    assert owner._decision_summary(owner.choice[3]) == first_decision
    jobs[1][0].set_result(jobs[1][1]())
    runner.tick(now + 2.2)
    assert requests[1].world.objects[0].name == "new_person"
    assert owner.choice[2].skill == "LOOK_AT"
    assert not runner.dialogue_pending[0].done()


def test_new_utterance_discards_older_decision_and_passes_latest_context(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    captured = []

    def decide(settings, request):
        captured.append(request)
        return result(request, "SIT")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.dialogue_pending = (Future(), "dialogue", runner.dialogue_epoch)
    runner.on_event(RuntimeEvent("utterance", now + 0.1, text="今は立ったままで話そう"), now + 0.1)
    choice = owner.choice
    runner.tick(now + 0.2)
    assert owner.choice == choice and len(jobs) == 3  # New dialogue and body work both start.
    jobs[-1][1]()
    assert captured[-1].state["conversation"][-1]["text"] == "今は立ったままで話そう"
    assert owner.attention is None and runner.pending is None


def test_decision_failure_keeps_current_pose_and_never_calls_chat(runtime):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    owner._choose(now, Intent(skill="SIT", duration_s=1), "test")
    runner.epoch = owner.generation
    body = simulated_body(posture_target("sitting_floor"), now)
    runner.tick(now)
    jobs[0][0].set_exception(TimeoutError("decision model unavailable"))
    runner.tick(now + 0.1)
    assert owner.choice[2].skill == "SIT" and runner.pending is None
    assert owner.health["decision"]["state"] == "unavailable"
    target = owner.step(body, now + 2, 0.02)
    assert target.pelvis == body.pelvis.pose and target.head == body.head.pose


@pytest.mark.parametrize("change", ["target_gone", "stale_vision", "manual", "expired"])
def test_result_revalidated_before_body_application(runtime, monkeypatch, change):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    owner.world = WorldState(
        objects=(WorldObject(name="p", source="fixture", position=(1.0, 0.0, 1.6)),)
    )
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.score",
        lambda settings, request: result(request, "LOOK_AT", "p"),
    )
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    if change == "target_gone":
        owner.world = WorldState()
    elif change == "stale_vision":
        owner.world = WorldState(
            timestamp=now - 2,
            objects=(
                WorldObject(
                    name="p",
                    source="vision",
                    position=(1.0, 0.0, 1.6),
                    image_position=(0.2, 0.0),
                    last_seen=now - 2,
                ),
            ),
        )
    elif change == "manual":
        owner.enable(False)
        runner.tick(now + 0.1)
        owner.enable(True)
    choice = owner.choice
    runner.tick(now + 6 if change == "expired" else now + 0.2)
    assert owner.choice == choice and runner.running is None


def test_body_candidates_filter_unavailable_skills_and_unobserved_geometry():
    world = WorldState(
        timestamp=10.0,
        objects=(
            WorldObject(
                name="fresh",
                source="vision",
                position=(1.0, 0.0, 1.0),
                image_position=(0.0, 0.0),
                last_seen=10.0,
            ),
            WorldObject(
                name="stale",
                source="vision",
                position=(1.0, 0.0, 1.0),
                image_position=(0.0, 0.0),
                last_seen=1.0,
            ),
            WorldObject(name="calibrated", source="fixture", position=(0.2, 0.2, 1.0)),
        ),
    )
    choices = candidates(CapabilityRegistry(), world)
    assert any(c.id == "continue" and not c.intent for c in choices)
    assert not any(c.intent.get("skill") in ("WALK_IN_PLACE", "EXPLORE_HOME") for c in choices)
    assert not any(c.intent.get("target") == "stale" for c in choices)
    assert all(
        c.intent["target"] == "calibrated" for c in choices if c.intent.get("skill") == "REACH"
    )


def test_expired_snapshot_not_sent_to_model(runtime, monkeypatch):
    owner, runner, jobs = runtime
    from myumiq_vrchat.decision import DecisionInput

    request = DecisionInput(candidates=candidates(runner.registry, owner.world), captured_at=0.0)
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.make_scorer",
        lambda *args: pytest.fail("stale input must not trigger inference"),
    )
    with pytest.raises(ValueError, match="before inference"):
        score(owner.config.decision, request)


def test_bad_snapshot_does_not_kill_executive_or_start_chat(runtime, monkeypatch):
    owner, runner, jobs = runtime

    def fail(*args):
        raise ValueError("invalid state snapshot")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.candidates", fail)
    choice = owner.choice
    runner.tick(time.perf_counter())
    assert owner.enabled and owner.choice == choice and not jobs and runner.pending is None
    assert owner.health["decision"]["state"] == "unavailable"


def test_candidate_identity_survives_catalogue_and_target_reordering():
    registry = CapabilityRegistry()
    first = WorldObject(name="人 一", source="fixture", position=(1.0, 0.0, 1.6))
    second = WorldObject(name="人 二", source="fixture", position=(0.0, 1.0, 1.6))
    before = candidates(registry, WorldState(objects=(first, second)))
    registry.get("CROUCH").available = False
    after = candidates(registry, WorldState(objects=(second, first)))
    old = {
        (c.intent.get("skill"), c.intent.get("hand"), c.intent.get("target")): c.id for c in before
    }
    assert len({c.id for c in after}) == len(after)
    for choice in after:
        identity = (
            choice.intent.get("skill"),
            choice.intent.get("hand"),
            choice.intent.get("target"),
        )
        assert choice.id == old[identity]


def test_world_changes_coalesce_while_speech_and_decision_are_pending(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(request, "SIT")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.dialogue_pending = (Future(), "dialogue", runner.dialogue_epoch)
    runner.tick(now)
    initial = owner.choice
    for offset, name in ((0.1, "first"), (0.2, "second")):
        owner.world = WorldState(
            timestamp=now + offset,
            objects=(WorldObject(name=name, source="fixture", position=(1.0, 0.0, 1.6)),),
        )
        runner.tick(now + offset)
        assert len(jobs) == 1 and owner.choice == initial
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.3)
    assert owner.choice[2].skill == "SIT" and len(jobs) == 1
    runner.tick(now + 2.4)
    assert len(jobs) == 2
    jobs[1][1]()
    assert requests[-1].world.objects[0].name == "second"
    assert requests[-1].state["conversation"] == []
    assert not runner.dialogue_pending[0].done()


def test_action_completion_during_inference_keeps_the_next_valid_proposal(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    selected = ["STAND"]
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.score", lambda s, r: result(r, selected[0])
    )
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    end = runner.running["started"] + runner.running["intent"].duration_s + 0.1
    # Start the next decision just before the existing action completes.
    selected[0] = "SIT"
    runner.tick(end - 0.2)
    pending = jobs[-1][0]
    pending.set_result(jobs[-1][1]())
    owner.snapshot = simulated_body(posture_target("standing"), end)
    runner.running["observations"] = 3
    runner.tick(end)
    assert owner.choice[2].skill == "SIT"
    assert runner.running and runner.history[-1]["status"] == "plan_completed"


def test_same_targets_with_new_frames_do_not_force_new_inference(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    a = WorldObject(name="a", source="fixture", position=(1.0, 0.0, 1.6))
    b = WorldObject(name="b", source="fixture", position=(0.0, 1.0, 1.6))
    owner.world = WorldState(timestamp=now, objects=(a, b))

    def keep(settings, request):
        return next(c for c in request.candidates if c.id == "continue"), {
            "captured_at": request.captured_at,
            "backend": "fixture",
        }

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", keep)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    owner.world = WorldState(timestamp=now + 0.2, objects=(b, a))
    runner.tick(now + 0.2)
    assert len(jobs) == 1
    runner.tick(now + 2.1)
    assert len(jobs) == 2


def test_completed_action_identity_reaches_fresh_decision_without_new_speech(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    heard = runner.shared.hear("しゃがんで", source="operator_text")
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(request, "CROUCH")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    action_goal = runner.goal_id
    completed_at = runner.running["started"] + runner.running["intent"].duration_s + 0.1
    # Neither an assistant reply nor elapsed duration proves successful execution.
    runner.shared.reply("しゃがんだよ", "", (), heard, True)
    runner.body_decision.next_request = completed_at + 10.0
    runner.tick(completed_at)
    assert len(jobs) == 2 and runner.running is None
    jobs[1][1]()
    state = requests[-1].state
    assert state["current_action"] is None
    last = state["last_outcome"]
    assert last["status"] == "plan_completed_unverified"
    assert last["evidence"]["success"] is None
    assert last["action"]["action_id"] == action_goal + ":0"
    assert last["action"]["context_utterance_id"] == heard["episode_id"]
    assert last["action"]["candidate_id"] == "body_CROUCH"
    assert state["conversation_epoch"] == 0
    assert state["commitment"] is None


def test_body_decision_preserves_planner_commitment_status_and_speaker(runtime, monkeypatch):
    owner, runner, jobs = runtime
    from myumiq_vrchat.purposes import Purpose, SkillRequest

    now = time.perf_counter()
    goal = Purpose(
        description="この人と一緒に過ごす",
        reason="継続した交流",
        success_description="観測できた交流",
        criterion="interaction",
        focus="visitor",
        steps=(SkillRequest(capability="WAIT"),),
    )
    runner.shared.start_plan(goal, "planner-goal")
    runner.shared.commitment["status"] = "needs_revalidation"
    commitment = dict(runner.shared.commitment)
    runner.shared.hear("少し待って", partner="confirmed-person")
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(request, "WAIT")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    assert runner.shared.commitment == commitment
    assert requests[0].state["commitment"]["status"] == "needs_revalidation"
    assert requests[0].state["commitment"]["id"] == commitment["id"]
    assert requests[0].state["conversation"][-1]["partner"] == "confirmed-person"
    runner.finish("interrupted", "test stop")
    assert runner.shared.commitment == commitment


def test_new_utterance_is_not_fulfilled_by_an_old_result_or_speech(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    runner.history.append(
        {
            "status": "plan_completed",
            "action": {"context_utterance_id": "previous-request", "intent": {"skill": "WAIT"}},
        }
    )
    runner.shared.hear("足踏みをして")
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(request, "WAIT")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    state = requests[0].state
    assert state["latest_utterance_id"] not in (None, "previous-request")
    assert state["actions_for_latest_utterance"] == []
    assert state["last_outcome_matches_latest_utterance"] is False


def test_fulfilled_request_survives_other_actions_and_detector_churn(runtime):
    owner, runner, _ = runtime
    heard = runner.shared.hear("手を振って")
    identity = heard["episode_id"]
    runner.history.append(
        {
            "status": "plan_completed",
            "evidence": {"success": True},
            "action": {
                "candidate_id": "body_WAVE_right",
                "context_utterance_id": identity,
                "intent": {"skill": "WAVE", "hand": "right"},
            },
        }
    )
    decision = runner.body_decision
    assert "body_WAVE_right" in decision._fulfilled_for_utterance(identity)
    # Even after the bounded outcome journal rolls over, unrelated perception or
    # autonomous exploration cannot resurrect this completed speech request.
    runner.history[:] = [
        {
            "status": "plan_completed",
            "evidence": {"success": True},
            "action": {
                "candidate_id": "body_EXPLORE_HOME",
                "context_utterance_id": identity,
                "intent": {"skill": "EXPLORE_HOME"},
            },
        }
    ]
    owner.world = WorldState(
        objects=(WorldObject(name="new-detection", source="vision", position=(2.0, 0.0, 1.6)),)
    )
    decision._observe_changes()
    assert set(decision._fulfilled_for_utterance(identity)) == {"body_WAVE_right"}
    repeated = runner.shared.hear("手を振って")
    assert repeated["episode_id"] != identity
    assert not decision._fulfilled_for_utterance(repeated["episode_id"])


def test_failed_candidates_are_removed_before_selection(runtime, monkeypatch):
    owner, runner, jobs = runtime
    condition = runner.condition(Intent(skill="STAND"))
    for success in (True, False, False):
        runner.shared.skill_outcome("STAND", condition, success, "motor_execution", {})
    requests = []

    def decide(settings, request):
        requests.append(request)
        return result(request, "WAIT")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    runner.tick(time.perf_counter())
    jobs[0][1]()
    assert "body_STAND" not in [c.id for c in requests[0].candidates]
    assert "body_WAIT" in [c.id for c in requests[0].candidates]


def test_new_request_gets_one_attempt_despite_old_failures(runtime, monkeypatch):
    owner, runner, jobs = runtime
    intent = Intent(skill="STAND")
    for _ in range(2):
        runner.shared.skill_outcome("STAND", runner.condition(intent), False, "motor_execution", {})
    heard = runner.shared.hear("今度は立って")
    captured = []

    def decide(settings, request):
        captured.append(request)
        return result(request, "STAND")

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)
    now = time.perf_counter()
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    assert captured[0].state["failed_candidates_requiring_new_request"] == ["body_STAND"]
    assert owner.choice[2].skill == "STAND" and runner.running
    assert not runner.body_decision.retry_available(intent, heard["episode_id"])
    runner.finish("execution_failed", {"success": False})
    runner.history.clear()  # bounded history rotation cannot renew an attempt
    runner.body_decision.conversation_changed()
    assert not runner.body_decision.retry_available(intent, heard["episode_id"])
    next_heard = runner.shared.hear("もう一度立って")
    assert runner.body_decision.retry_available(intent, next_heard["episode_id"])
    assert not runner.body_decision.retry_available(intent, heard["episode_id"])


def test_requested_posture_survives_outcome_and_chat_until_new_body_request(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    runner.shared.hear("しゃがんで")
    selected = ["CROUCH", "latest_utterance"]

    def decide(settings, request):
        chosen, report = result(request, selected[0])
        return chosen, report | {"basis": selected[1]}

    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", decide)

    def advance(at):
        runner.body_decision.next_request = 0.0
        runner.tick(at)
        jobs[-1][0].set_result(jobs[-1][1]())
        runner.tick(at + 0.01)

    advance(now)
    assert owner.choice[2].skill == "CROUCH"
    runner.finish("execution_failed", {"success": False})
    selected[:] = ["STAND", "latest_utterance"]
    advance(now + 0.2)  # reinterpreting the same request cannot restore standing
    assert owner.choice[2].skill == "CROUCH"
    runner.shared.hear("今日は楽しいね")
    runner.body_decision.conversation_changed()
    selected[1] = "observation"
    advance(now + 0.4)
    assert owner.choice[2].skill == "CROUCH"
    assert runner.shared.working["maintained_posture"]["skill"] == "CROUCH"
    runner.shared.hear("今度は立って")
    runner.body_decision.conversation_changed()
    selected[1] = "latest_utterance"
    advance(now + 0.6)
    assert owner.choice[2].skill == "STAND"
    runner.interrupt("manual")
    assert runner.body_decision.maintained_posture is None


@pytest.mark.parametrize("choice", ["body_CROUCH", None])
def test_selection_owns_body_without_scores_or_chat_fallback(runtime, monkeypatch, choice):
    owner, runner, jobs = runtime
    profile = SelectionSettings(
        llm=LLMConfig(base_url="http://127.0.0.1:1234/v1", model="body-weights")
    )
    owner.config = owner.config.model_copy(update={"decision": DecisionSettings(selection=profile)})

    def choose(self, request):
        return SelectionResult(
            backend="local_selection",
            model="body-weights",
            captured_at=request.captured_at,
            available_ids=tuple(c.id for c in request.candidates),
            candidate_id=choice,
            image_used=False,
        )

    monkeypatch.setattr("myumiq_vrchat.decision_selection.LocalSelection.choose", choose)
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.make_scorer",
        lambda *a: pytest.fail("Selection cannot become independent scores"),
    )
    now = time.perf_counter()
    runner.dialogue_pending = (Future(), "dialogue", runner.dialogue_epoch)
    before = owner.choice
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    if choice:
        assert owner.choice[2].skill == "CROUCH"
        assert owner._decision_summary(owner.choice[3])["selected_id"] == choice
    else:
        assert owner.choice == before and runner.running is None
    assert not runner.dialogue_pending[0].done()
    assert "scores" not in owner.health["decision"]["result"]


@pytest.mark.parametrize(
    "invalid", ["both", "neither", "same_model", "same_endpoint", "no_purpose"]
)
def test_selection_configuration_keeps_model_roles_separate(runtime, invalid):
    owner, runner, jobs = runtime
    profile = SelectionSettings(
        llm=LLMConfig(base_url="http://127.0.0.1:1234/v1", model="body-weights")
    )
    with pytest.raises(ValueError):
        if invalid == "both":
            DecisionSettings(backend=owner.config.decision.backend, selection=profile)
        elif invalid == "neither":
            DecisionSettings()
        else:
            llm = profile.llm.model_copy(
                update={"model": owner.config.llm.model}
                if invalid == "same_model"
                else {"base_url": owner.config.llm.base_url}
                if invalid == "same_endpoint"
                else {}
            )
            config = owner.config.model_copy(
                update={
                    "decision": DecisionSettings(selection=profile.model_copy(update={"llm": llm})),
                    "purpose": None if invalid == "no_purpose" else owner.config.purpose,
                }
            )
            AutonomousConfig.model_validate_json(config.model_dump_json())


@pytest.mark.parametrize("change", ["none", "utterance", "world", "unknown"])
def test_successful_action_is_not_reexecuted_without_a_new_stimulus(runtime, monkeypatch, change):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", lambda s, r: result(r, "STAND"))
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    original = owner.choice
    end = runner.running["started"] + runner.running["intent"].duration_s + 0.1
    if change != "unknown":
        owner.snapshot = simulated_body(posture_target("standing"), end)
        runner.running["observations"] = 3
    runner.tick(end)
    jobs[-1][0].set_result(jobs[-1][1]())
    if change == "utterance":
        runner.body_decision.conversation_changed()
    elif change == "world":
        owner.world = WorldState(
            objects=(WorldObject(name="new", source="fixture", position=(1.0, 0.0, 1.6)),)
        )
    runner.tick(end + 0.1)
    if change in ("utterance", "world"):
        jobs[-1][0].set_result(jobs[-1][1]())
        runner.tick(end + 0.2)
    assert (owner.choice == original) is (change in ("none", "world"))
    assert (runner.running is None) is (change in ("none", "world"))


def test_unrelated_visual_identity_change_does_not_discard_posture_decision(runtime, monkeypatch):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    monkeypatch.setattr("myumiq_vrchat.embodied_decision.score", lambda s, r: result(r, "CROUCH"))
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    owner.world = WorldState(
        objects=(WorldObject(name="new", source="fixture", position=(1.0, 0.0, 1.6)),)
    )
    runner.tick(now + 0.1)
    assert owner.choice[2].skill == "CROUCH"


@pytest.mark.parametrize("moved", [False, True])
def test_completed_target_action_can_resume_when_the_same_target_moves(runtime, monkeypatch, moved):
    owner, runner, jobs = runtime
    now = time.perf_counter()
    target = WorldObject(
        name="p", source="fixture", position=(1.0, 0.0, owner.rest.head.position[2])
    )
    owner.world = WorldState(objects=(target,))
    monkeypatch.setattr(
        "myumiq_vrchat.embodied_decision.score", lambda s, r: result(r, "LOOK_AT", "p")
    )
    runner.tick(now)
    jobs[0][0].set_result(jobs[0][1]())
    runner.tick(now + 0.1)
    original = owner.choice
    end = runner.running["started"] + runner.running["intent"].duration_s + 0.1
    owner.snapshot = simulated_body(owner.rest, end)
    runner.running["observations"] = 3
    runner.tick(end)
    assert runner.history[-1]["evidence"]["success"] is True
    jobs[-1][0].set_result(jobs[-1][1]())
    if moved:
        owner.world = WorldState(objects=(target.model_copy(update={"position": (1.0, 0.6, 1.6)}),))
    runner.tick(end + 0.1)
    assert (owner.choice != original) is moved


@pytest.mark.parametrize("execution", ["accepted", "running", "completed", "failed", "unverified"])
def test_reassessment_does_not_rewrite_execution_evidence(runtime, execution):
    owner, runner, jobs = runtime
    decision = runner.body_decision
    decision.assess_request(
        {"request_status": "action", "requested_capabilities": ["MOVE_FORWARD"]}, "u1", 1.0
    )
    decision.mark_request(
        "u1",
        execution=execution,
        skill="MOVE_FORWARD",
        evidence_scope="image_change_not_metric_displacement",
    )
    decision.assess_request(
        {"request_status": "unsupported", "request_reason": "camera missing"}, "u1", 2.0
    )
    decision.mark_request("u1", execution="unavailable", reason="later inference failed")
    assert decision.request_assessment["execution"] == execution
    assert decision.request_assessment["status"] == "action"
    assert decision.request_assessment["evidence_scope"] == "image_change_not_metric_displacement"
    decision.assess_request({"request_status": "none"}, "u2", 3.0)
    assert decision.request_assessment["utterance_id"] == "u2"
    assert decision.request_assessment["execution"] == "not_accepted"


@pytest.mark.parametrize("cancel", [None, "new_input", "deadline"])
def test_pending_ack_gets_one_grounded_followup_only_for_current_input(
    runtime, monkeypatch, cancel
):
    from myumiq_vrchat.shared_dialogue import Dialogue, request_dialogue

    owner, runner, jobs = runtime
    now = time.perf_counter()
    heard = runner.shared.hear("ついてきて")
    runner.utterances = [heard]
    owner.config = owner.config.model_copy(
        update={
            "decision": DecisionSettings(
                selection=SelectionSettings(
                    llm=LLMConfig(base_url="http://localhost:2", model="thought"),
                    understand_requests=True,
                )
            ),
            "dialogue": DialogueSettings(await_body_assessment=True),
        }
    )
    spoken = []
    monkeypatch.setattr(runner.services, "reply", lambda text, *args: spoken.append(text) or True)

    def reply(*args, **kwargs):
        if kwargs["action_context"]["status"] == "pending":
            return request_dialogue(*args, **kwargs)
        assert kwargs["action_context"]["status"] == "unsupported"
        return Dialogue(reply="今はついていけないよ。", topic="移動")

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue.request_dialogue", reply)
    runner._dialogue_tick(now)
    runner._dialogue_tick(now + 3.1)
    jobs[-1][0].set_result(jobs[-1][1]())
    runner._dialogue_tick(now + 3.2)
    assert len(spoken) == 1 and runner.assessment_followup is not None
    runner.body_decision.assess_request(
        {"request_status": "unsupported"}, heard["episode_id"], now + 3.3
    )
    if cancel == "new_input":
        runner.dialogue_epoch += 1
    runner._dialogue_tick(now + (20 if cancel == "deadline" else 3.4))
    if cancel:
        assert len(jobs) == 1 and runner.assessment_followup is None
    else:
        assert len(jobs) == 2
        jobs[-1][0].set_result(jobs[-1][1]())
        runner._dialogue_tick(now + 3.5)
        runner._dialogue_tick(now + 4.0)
        assert spoken[-1] == "今はついていけないよ。"
        assert len(spoken) == 2 and runner.assessment_followup is None
        assert sum(t["role"] == "user" for t in runner.shared.working["turns"]) == 1
