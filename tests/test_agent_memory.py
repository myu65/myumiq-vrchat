import time
from concurrent.futures import Future

import pytest

from myumiq_vrchat.agent_memory import AgentMemory
from myumiq_vrchat.autonomous_body import AutonomousBody, AutonomousConfig, Intent
from myumiq_vrchat.body import WorldObject, WorldState, simulated_body
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.events import RuntimeEvent
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.purpose_runtime import PurposeRunner, PurposeSettings
from myumiq_vrchat.purposes import Purpose, SkillRequest
from myumiq_vrchat.shared_dialogue import Dialogue


def goal(**kwargs):
    return Purpose(
        description="この人をもっと知りたい",
        reason="交流への関心",
        success_description="会話を続ける",
        steps=(SkillRequest(capability="WAIT"),),
        **kwargs,
    )


def test_social_retrieval_reencounter_and_anonymous_separation(tmp_path):
    path = tmp_path / "memory.db"
    memory = AgentMemory(path)
    heard = memory.hear("私は猫と暮らしています", partner="confirmed-aki", source="operator_text")
    memory.reply("猫のお名前は何ですか", "猫の話", ["猫と暮らしています", "犬が好き"], heard, True)
    assert "猫と暮らしています" in memory.context()["semantic"][0]["text"]
    assert len(memory.context()["semantic"]) == 1
    memory.close()
    memory = AgentMemory(path)
    memory.hear("また会いましたね", partner="confirmed-aki")
    context = memory.context()
    assert context["social"]["known"] and context["social"]["prior_sessions"] == 1
    assert "猫と暮らしています" in context["semantic"][0]["text"]
    memory.hear("こんにちは", partner=None)
    assert memory.context()["semantic"] == []
    assert not memory.context()["social"]["known"]
    memory.close()


def test_commitment_outlives_steps_and_restart_never_restores_actuation(tmp_path):
    memory = AgentMemory(tmp_path / "memory.db")
    memory.start_plan(goal(criterion="interaction"), "plan1")
    identity = memory.commitment["id"]
    memory.plan_result("plan_completed", {"scope": "device_execution_only"})
    assert memory.commitment["status"] == "active"
    memory.start_plan(goal(), "plan2")
    assert memory.commitment["id"] == identity
    snapshot = memory.context()
    memory.working["topic"] = "別の話題"
    assert snapshot["working"]["topic"] != "別の話題"
    memory.close()
    memory = AgentMemory(tmp_path / "memory.db")
    assert (
        memory.commitment["id"] == identity and memory.commitment["status"] == "needs_revalidation"
    )
    assert memory.working["plan"] is None and memory.working["turns"] == []
    memory.close()


@pytest.mark.parametrize("visual_scope", ["fresh_visual_alignment", "fresh_visual_body_heading"])
def test_only_observed_visual_success_completes_observation_goal(tmp_path, visual_scope):
    memory = AgentMemory(tmp_path / "memory.db")
    memory.start_plan(goal(criterion="observe_target", focus="target"), "plan")
    memory.skill_outcome("LOOK_AT", "vision", None, "unavailable", {}, "target")
    assert memory.commitment["progress"] == 0
    memory.skill_outcome("LOOK_AT", "vision", True, "device_execution_only", {}, "target")
    assert memory.commitment["progress"] == 0
    memory.skill_outcome("LOOK_AT", "vision", True, visual_scope, {}, "other-target")
    assert memory.commitment["status"] == "active"
    memory.skill_outcome("LOOK_AT", "vision", True, visual_scope, {}, "target")
    assert memory.commitment["status"] == "completed"
    memory.close()


def test_finite_action_does_not_create_or_reactivate_a_commitment(tmp_path):
    memory = AgentMemory(tmp_path / "memory.db")
    memory.start_plan(goal(), "action1", manage_commitment=False)
    memory.plan_result("plan_completed", {"success": True, "scope": "device_execution_only"})
    assert memory.commitment is None
    assert memory.working["plan"]["status"] == "plan_completed"
    memory.start_plan(goal(criterion="observe_target", focus="target"), "planner")
    memory.commitment["status"] = "suspended"
    before = dict(memory.commitment)
    memory.start_plan(goal(continuity="replace"), "action2", manage_commitment=False)
    memory.plan_result("blocked_context", "missing target")
    memory.plan_result("interrupted", "manual action cancellation")
    assert memory.commitment == before
    # Only the matching observed evidence may advance the broader goal.
    memory.skill_outcome("LOOK_AT", "vision", True, "fresh_visual_alignment", {}, "target")
    assert memory.commitment["status"] == "completed"
    memory.close()


class Services:
    voice = vision = None

    def __init__(self):
        self.spoken = []

    def reply(self, text, now):
        self.spoken.append(text)
        return True


def executive(tmp_path):
    config = AutonomousConfig(
        llm=LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        memory=tmp_path / "legacy.json",
        purpose=PurposeSettings(state=tmp_path / "purpose.json"),
    )
    owner = AutonomousBody(config, tmp_path, posture_target("standing"), None)
    owner.enable(True)
    services = Services()
    return owner, PurposeRunner(owner, services), services


@pytest.mark.parametrize("kind", ["asr_no_speech", "asr_error", "audio_input_overflow"])
@pytest.mark.parametrize("current", [False, True])
def test_terminal_asr_releases_only_its_own_listening_attention(tmp_path, kind, current):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    runner.accept(goal(), now)
    existing_goal = runner.goal_id
    runner.on_event(
        RuntimeEvent("speech_started", now, input_session="capture", input_sequence=2), now
    )
    until = runner.listening_until
    revision = runner.goal_context_revision
    runner.on_event(
        RuntimeEvent(kind, now + 1, input_session="capture", input_sequence=2 if current else 1),
        now + 1,
    )
    assert runner.listening_until == (now + 1 if current else until)
    assert runner.shared.working["attention"] == ("environment" if current else "conversation")
    assert runner.goal_id == existing_goal and runner.goal_context_revision == revision
    runner.close()


def test_visual_failure_conditions_follow_control_not_detector_id(tmp_path):
    owner, runner, _ = executive(tmp_path)
    obj = WorldObject(
        name="p1",
        source="vision",
        position=(2, 0, 1.6),
        image_position=(0.6, 0.1),
        confidence=0.8,
        last_seen=1,
    )
    owner.world = WorldState(objects=(obj,))
    intent = Intent(skill="LOOK_AT", target="p1", duration_s=3)
    initial = runner.condition(intent)
    for _ in range(2):
        runner.shared.skill_outcome("LOOK_AT", initial, False, "fresh_visual_alignment", {})
    obj = obj.model_copy(update={"name": "p2"})
    owner.world = WorldState(objects=(obj,))
    intent = intent.model_copy(update={"target": "p2"})
    assert runner.shared.repeated_failure("LOOK_AT", runner.condition(intent))
    assert not runner.shared.repeated_failure(
        "LOOK_AT", runner.condition(intent.model_copy(update={"duration_s": 8}))
    )
    owner.world = WorldState(objects=(obj.model_copy(update={"image_position": (0.1, 0.1)}),))
    assert not runner.shared.repeated_failure("LOOK_AT", runner.condition(intent))
    runner.close()


def test_dialogue_preserves_goal_and_topic_without_body_control(tmp_path, monkeypatch):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    runner.accept(goal(), now)
    identity = runner.shared.commitment["id"]
    seen = []

    def respond(config, world, drives, body, context, text, **kwargs):
        seen.append(context)
        return Dialogue(
            reply="猫のお話を続けましょう", topic="猫の話", remember_quotes=("猫が好き",)
        )

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue.request_dialogue", respond)
    runner.on_event(
        RuntimeEvent("utterance", now, text="猫が好き", speaker_id="aki", source="operator_text"),
        now,
    )
    for _ in range(100):
        runner.tick(time.perf_counter())
        if services.spoken:
            break
        time.sleep(0.002)
    assert services.spoken and owner.gesture is None
    assert runner.goal is not None and runner.shared.commitment["id"] == identity
    assert runner.shared.working["topic"] == "猫の話"
    assert seen[0]["commitment"]["id"] == identity
    assert seen[0]["working"]["turns"][-1]["text"] == "猫が好き"
    body = simulated_body(owner.rest, now)
    owner.step(body, now + 0.1, 0.02)
    assert all(t.id != "conversation_gesture" for t in owner.goal.tasks)
    assert owner.intent_metadata["shared_memory"]["commitment_id"] == identity
    runner.close()


def test_new_speech_discards_old_reply_without_forgetting_interest(tmp_path):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    runner.accept(goal(), now)
    identity = runner.shared.commitment["id"]
    future = Future()
    runner.dialogue_pending = (future, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("古い発言")
    runner.on_event(RuntimeEvent("speech_started", now), now)
    future.set_result(Dialogue(reply="古い返答です", topic="古い話"))
    runner.tick(now + 0.1)
    assert not services.spoken and runner.shared.commitment["id"] == identity
    owner.enable(False)
    runner.tick(now + 0.2)
    assert runner.goal is None
    runner.close()


def test_speech_does_not_cancel_autonomous_plan_and_dialogue_starts_while_it_waits(
    tmp_path, monkeypatch
):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    planning = Future()
    runner.pending = (planning, "purpose", owner.generation)
    generation = owner.generation
    monkeypatch.setattr(
        "myumiq_vrchat.shared_dialogue.request_dialogue",
        lambda *args, **kwargs: Dialogue(reply="話しながら見ているよ", topic="周り"),
    )
    runner.on_event(RuntimeEvent("speech_started", now), now)
    runner.on_event(RuntimeEvent("utterance", now, text="こんにちは"), now)
    for _ in range(100):
        runner.tick(time.perf_counter())
        if services.spoken:
            break
        time.sleep(0.002)
    assert services.spoken
    assert runner.pending[0] is planning and not planning.done()
    assert owner.generation == generation
    runner.close()


def test_autonomous_step_starts_while_dialogue_generation_is_unfinished(tmp_path):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    owner.snapshot = simulated_body(owner.rest, now)
    runner.accept(goal(), now)
    dialogue = Future()
    runner.dialogue_pending = (dialogue, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("ゆっくり返事して")
    step = Future()
    step.set_result((Intent(skill="WAIT", duration_s=1), None))
    runner.pending = (step, "step", owner.generation)
    runner.tick(now + 0.1)
    assert runner.running is not None
    assert owner.choice[3] == "local_llm_plan"
    assert not dialogue.done() and not services.spoken
    # Starting a new autonomous action must not discard the independent reply.
    dialogue.set_result(Dialogue(reply="話しながら続けるね", topic="会話"))
    runner.tick(now + 0.2)
    assert services.spoken == ["話しながら続けるね"]
    runner.tick(now + 1.2)
    assert runner.running is None
    runner.close()


def test_manual_preemption_discards_pending_dialogue(tmp_path):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    future = Future()
    runner.dialogue_pending = (future, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("前の依頼")
    owner.enable(False)
    runner.tick(now)
    future.set_result(Dialogue(reply="古い返答だよ", topic="前"))
    owner.enable(True)
    runner.tick(now + 0.1)
    assert not services.spoken and owner.gesture is None
    runner.close()


@pytest.mark.parametrize(
    "action",
    [
        None,
        SkillRequest(capability="LOOK_AT", target="gone"),
        SkillRequest(capability="SIT"),
    ],
)
def test_dialogue_worker_cannot_preempt_body_even_with_legacy_control_result(tmp_path, action):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    runner.accept(goal(), now)
    commitment = runner.shared.commitment["id"]
    runner.pending = (Future(), "step", owner.generation)
    owner.choice = (owner.generation, now, Intent(skill="SIT", duration_s=10), "local_llm_plan")
    choice = owner.choice
    future = Future()
    from types import SimpleNamespace

    future.set_result(
        SimpleNamespace(reply="こんにちは、話そう", topic="挨拶", action=action, remember_quotes=())
    )
    runner.dialogue_pending = (future, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("こんにちは")
    runner.tick(now + 0.1)
    assert services.spoken == ["こんにちは、話そう"]
    assert owner.choice == choice and runner.goal is not None
    assert runner.shared.commitment["id"] == commitment
    assert owner.attention is None and owner.gesture is None
    assert "dialogue_action" not in owner.health
    runner.close()


def test_reply_retries_temporary_output_unavailability_once_without_repeating_action(tmp_path):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    attempts = []

    def reply(text, at):
        attempts.append(text)
        if len(attempts) < 3:
            return False
        services.spoken.append(text)
        return True

    services.reply = reply
    future = Future()
    future.set_result(Dialogue(reply="うん、聞こえるよ", topic="確認"))
    runner.dialogue_pending = (future, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("聞こえる？")
    runner._dialogue_tick(now)
    gesture = owner.gesture
    assert runner.reply_pending and not services.spoken
    assert (
        runner.shared.db.execute("SELECT COUNT(*) FROM episode WHERE kind='reply'").fetchone()[0]
        == 0
    )
    for offset in (0.1, 0.2, 0.3):
        runner._dialogue_tick(now + offset)
    assert owner.gesture == gesture
    assert services.spoken == ["うん、聞こえるよ"] and len(attempts) == 3
    assert runner.reply_pending is None
    assert (
        runner.shared.db.execute("SELECT COUNT(*) FROM episode WHERE kind='reply'").fetchone()[0]
        == 1
    )
    runner.close()


@pytest.mark.parametrize("preemption", ["speech", "manual", "timeout"])
def test_pending_speech_does_not_play_after_preemption_or_deadline(tmp_path, preemption):
    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    services.reply = lambda *args: False
    future = Future()
    future.set_result(Dialogue(reply="前の返答だよ", topic="前"))
    runner.dialogue_pending = (future, "dialogue", runner.dialogue_epoch)
    runner.reply_to = runner.shared.hear("前の質問")
    runner._dialogue_tick(now)
    assert runner.reply_pending
    if preemption == "speech":
        runner.on_event(RuntimeEvent("speech_started", now + 0.1), now + 0.1)
    elif preemption == "manual":
        owner.enable(False)
        runner.tick(now + 0.1)
        owner.enable(True)
    services.reply = lambda *args: pytest.fail("expired/preempted speech must not be submitted")
    runner._dialogue_tick(now + 6 if preemption == "timeout" else now + 0.2)
    assert runner.reply_pending is None
    if preemption == "timeout":
        import json

        data = json.loads(
            runner.shared.db.execute("SELECT data FROM episode WHERE kind='reply'").fetchone()[0]
        )
        assert data["submitted"] is False
        assert owner.health["dialogue"]["state"] == "output_expired"
    else:
        assert (
            runner.shared.db.execute("SELECT COUNT(*) FROM episode WHERE kind='reply'").fetchone()[
                0
            ]
            == 0
        )
    runner.close()


def test_visual_feedback_sample_hold_and_fresh_alignment(tmp_path):
    owner, runner, _ = executive(tmp_path)
    now = time.perf_counter()
    owner.world = WorldState(
        objects=(
            WorldObject(
                name="object",
                position=(1.0, 0.0, 1.6),
                source="vision",
                last_seen=now,
                image_position=(0.5, 0.0),
            ),
        ),
        timestamp=now,
    )
    owner._choose(now, Intent(skill="LOOK_AT", target="object", duration_s=5), "test")
    body = simulated_body(owner.rest, now)
    first = owner.step(body, now, 0.02)
    rotation = owner.attention_motor.orientation
    body = simulated_body(first, now + 0.02)
    owner.step(body, now + 0.02, 0.02)
    assert owner.attention_motor.orientation == rotation
    assert first.head.orientation != owner.rest.head.orientation
    runner.close()


def test_repeated_failures_change_next_plan_and_learning_changes_retrieval(tmp_path):
    owner, runner, _ = executive(tmp_path)
    intent = Intent(skill="WAVE", hand="right")
    condition = runner.condition(intent)
    for _ in range(2):
        runner.shared.skill_outcome("WAVE", condition, False, "device_execution_only", {})
    proposed = Purpose(
        description="手で挨拶したい",
        reason="交流",
        success_description="手を振る",
        steps=(SkillRequest(capability="WAVE", hand="right"),),
    )
    runner.accept(proposed, time.perf_counter())
    assert runner.goal is None and runner.history[-1]["status"] == "blocked_context"
    runner.shared.learned(
        {"capability": "WALK_IN_PLACE", "result": {"evaluation_scope": "same_clip"}}
    )
    context = runner.shared.context()
    assert any("WALK_IN_PLACE" in k["text"] for k in context["semantic"])
    assert context["failure_conditions"][0]["failure"] == 2
    runner.close()


def test_recent_failure_streak_is_not_hidden_by_an_old_success(tmp_path):
    owner, runner, _ = executive(tmp_path)
    condition = runner.condition(Intent(skill="STAND"))
    for success in (True, False, False):
        runner.shared.skill_outcome("STAND", condition, success, "motor_execution", {})
    assert runner.shared.repeated_failure("STAND", condition)
    assert not runner.shared.repeated_failure("STAND", condition + "changed")
    runner.shared.skill_outcome("STAND", condition, True, "device_tracker_goal", {})
    assert not runner.shared.repeated_failure("STAND", condition)
    runner.close()


def test_audio_in_shared_mode_only_publishes_transcript():
    from myumiq_vrchat.audio import SpeechEvent
    from myumiq_vrchat.conversation import ConversationPipeline
    from myumiq_vrchat.events import EventInbox

    class ASR:
        def transcribe(self, *args):
            return "前の話を続けたい"

    class Output:
        speaking = False

        def stop(self):
            pass

        def speak(self, text):
            raise AssertionError("audio worker must not decide or speak")

    events = EventInbox()
    pipeline = ConversationPipeline(ASR(), Output(), events, None)
    pipeline.accept(SpeechEvent("speech_started", 1.0), WorldState())
    pipeline.accept(SpeechEvent("speech_ended", 2.0, audio=(0.1,) * 512), WorldState())
    for _ in range(100):
        pipeline.poll()
        result = events.drain()
        if any(e.kind == "utterance" for e in result):
            assert next(e for e in result if e.kind == "utterance").text == "前の話を続けたい"
            return
        time.sleep(0.002)
    raise AssertionError("transcript event missing")


def test_late_transcripts_are_remembered_without_restarting_obsolete_replies(tmp_path, monkeypatch):
    import json

    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    runner.accept(goal(), now)
    commitment = runner.shared.commitment["id"]
    running_goal = runner.goal_id
    runner.on_event(
        RuntimeEvent("speech_started", now, input_session="capture", input_sequence=3), now
    )
    epoch = runner.dialogue_epoch
    seen = []

    def respond(config, world, drives, body, context, text, **kwargs):
        seen.append((text, context))
        return Dialogue(reply="うん、聞いているよ", topic="会話")

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue.request_dialogue", respond)
    for sequence, text in enumerate(("しゃがんで", "今度は立って", "手も振って"), 1):
        runner.on_event(
            RuntimeEvent(
                "utterance",
                now - 3 + sequence,
                text=text,
                input_session="capture",
                input_sequence=sequence,
                utterance_id=f"capture:{sequence}",
                audio_start_at=now - 4 + sequence,
                audio_end_at=now - 3 + sequence,
            ),
            now,
        )
        if sequence < 3:
            assert runner.dialogue_epoch == epoch
            assert not runner.utterances and not runner.dialogue_pending
    assert [turn["text"] for turn in runner.shared.working["turns"]] == [
        "しゃがんで",
        "今度は立って",
        "手も振って",
    ]
    assert [turn["text"] for turn in runner.utterances] == ["手も振って"]
    assert runner.goal_id == running_goal and runner.shared.commitment["id"] == commitment
    runner._dialogue_tick(now)
    runner.dialogue_pending[0].result(timeout=2)
    runner._dialogue_tick(now + 0.1)
    assert services.spoken == ["うん、聞いているよ"]
    assert [text for text, _ in seen] == ["手も振って"]
    assert [turn["text"] for turn in seen[0][1]["working"]["turns"]] == [
        "しゃがんで",
        "今度は立って",
        "手も振って",
    ]
    rows = runner.shared.db.execute("SELECT data FROM episode WHERE kind='heard'").fetchall()
    assert {json.loads(row[0])["input"]["utterance_id"] for row in rows} == {
        "capture:1",
        "capture:2",
        "capture:3",
    }
    runner.close()


def test_new_capture_blocks_old_reply_even_before_onset_event_is_delivered(tmp_path):
    from types import SimpleNamespace

    from myumiq_vrchat.audio import SpeechEvent
    from myumiq_vrchat.conversation import ConversationPipeline
    from myumiq_vrchat.events import EventInbox

    owner, runner, services = executive(tmp_path)
    now = time.perf_counter()
    events = EventInbox()
    pipeline = ConversationPipeline(
        SimpleNamespace(transcribe=lambda *args: "前の質問"),
        SimpleNamespace(speaking=False, stop=lambda: None),
        events,
        None,
    )
    services.voice = SimpleNamespace(pipeline=pipeline)
    pipeline.accept(SpeechEvent("speech_started", now), WorldState())
    pipeline.accept(SpeechEvent("speech_ended", now + 1, (0.1,) * 512), WorldState())
    pipeline.pending.result(timeout=2)
    pipeline.poll()
    for event in events.drain():
        runner.on_event(event, now + 1)
    heard = runner.utterances.pop()
    runner.reply_pending = (
        Dialogue(reply="前の返答だよ", topic="前"),
        heard,
        runner.dialogue_epoch,
        now + 5,
    )
    # Capture changes synchronously; deliberately do not drain the onset event.
    pipeline.accept(SpeechEvent("speech_started", now + 2), WorldState())
    runner._submit_reply(now + 2)
    assert runner.reply_pending is None and not services.spoken
    pipeline.close()
    services.voice = None
    runner.close()
