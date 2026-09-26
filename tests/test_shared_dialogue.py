import json

import pytest

from myumiq_vrchat.body import WorldState
from myumiq_vrchat.cognition import LLMConfig
from myumiq_vrchat.shared_dialogue import Dialogue, dialogue_memory, request_dialogue


def test_recent_fact_survives_intervening_turns_and_reply_cancellation(tmp_path):
    from myumiq_vrchat.agent_memory import AgentMemory

    memory = AgentMemory(tmp_path / "memory.sqlite3")
    try:
        memory.hear("今日の合言葉は青い傘だよ。")  # Its reply was cancelled.
        heard = memory.hear("覚えておいてね。")
        memory.reply("うん、覚えておくね。", "合言葉", [], heard, True)
        heard = memory.hear("赤と青ならどっちが好き？")
        memory.reply("青が好きかな。", "色", [], heard, True)
        memory.hear("さっきの合言葉は？")
        context = memory.context()
        projected = dialogue_memory(context)
        assert projected["working"]["turns"][0]["quoted_text"] == "今日の合言葉は青い傘だよ。"
        assert projected["semantic"] == []  # Anonymous input is not a personal fact.
        for index in range(12):
            memory.hear(str(index) + "あ" * 2000)
        projected = dialogue_memory(memory.context())
        assert len(projected["working"]["turns"]) == 10
        assert all(len(t["quoted_text"]) <= 240 for t in projected["working"]["turns"])
        assert "青い傘" not in str(projected["working"]["turns"])
    finally:
        memory.close()


def test_speech_projection_keeps_ownership_and_outcome_without_motor_payload():
    context = {
        "working": {
            "turns": [{"role": "user", "text": "私は赤が好き"}],
            "plan": {"description": "座る", "status": "execution_failed", "raw": "x" * 50000},
        },
        "commitment": {"description": "遊ぶ", "status": "active", "evidence": ["x" * 50000]},
        "semantic": [{"text": "相手は赤が好き", "scope": "reported", "evidence": "quote-id"}],
        "episodic": [
            {
                "kind": "skill_outcome",
                "data": {
                    "skill": "SIT",
                    "success": False,
                    "scope": "device_execution",
                    "condition": "x" * 50000,
                    "evidence": {"raw": "x" * 50000},
                },
            }
        ],
    }
    projected = dialogue_memory(context)
    assert len(json.dumps(projected)) < 2000
    assert projected["working"]["turns"][0]["speaker"] == "相手"
    assert projected["semantic"][0]["evidence"] == "quote-id"
    assert projected["episodic"][0]["data"]["success"] is False
    assert projected["commitment"]["description"] == "遊ぶ"


def test_speech_only_uses_one_request_with_history_and_no_control_schema(monkeypatch):
    calls = []

    def reply(config, messages, schema, name, tokens):
        calls.append(name)
        assert set(schema["properties"]) == {"reply", "topic", "remember_quotes"}
        assert messages[-1]["content"] == "こんにちは"
        assert "昔の返答" in json.dumps(messages, ensure_ascii=False)
        assert "requested_body_action" not in json.dumps(messages)
        return {"reply": "こんにちは、元気？", "topic": "挨拶", "remember_quotes": []}, 0.1

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue._request", reply)
    value = request_dialogue(
        LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
        WorldState(),
        {},
        {},
        {"working": {"turns": [{"role": "assistant", "text": "昔の返答"}]}},
        "こんにちは",
    )
    assert value.reply == "こんにちは、元気？"
    assert not hasattr(value, "action") and calls == ["shared_dialogue"]


@pytest.mark.parametrize("field", ["action", "action_error"])
def test_speech_generation_cannot_supply_body_control(monkeypatch, field):
    monkeypatch.setattr(
        "myumiq_vrchat.shared_dialogue._request",
        lambda *args: ({"reply": "こんにちは", "topic": "会話", field: None}, 0.1),
    )
    with pytest.raises(ValueError, match="speech generator"):
        request_dialogue(
            LLMConfig(base_url="http://127.0.0.1:1/v1", model="test"),
            WorldState(),
            {},
            {},
            {"working": {"turns": []}},
            "こんにちは",
        )
    with pytest.raises(ValueError):
        Dialogue(reply="こんにちは", topic="会話", **{field: None})


def test_dialogue_receives_actual_image_and_capability_evidence(monkeypatch):
    import time

    from myumiq_vrchat.visual_context import VisualContext

    visual = VisualContext(WorldState(), "encoded-jpeg", time.perf_counter())

    def reply(config, messages, schema, name, tokens):
        parts = messages[-1]["content"]
        assert parts[0] == {"type": "text", "text": "何が見える？"}
        assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,encoded-jpeg"
        prompt = messages[0]["content"]
        assert '"available": false' in prompt and '"status": "unsupported"' in prompt
        assert '"image_used": true' in prompt
        return {"reply": "赤い壁が見えるよ。", "topic": "景色"}, 0.1

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue._request", reply)
    value = request_dialogue(
        LLMConfig(base_url="http://localhost:1", model="vision"),
        visual.world,
        {},
        {},
        {"working": {}},
        "何が見える？",
        visual=visual,
        capabilities=[{"name": "FOLLOW", "available": False}],
        action_context={"status": "unsupported"},
    )
    assert value._inference["visual"]["image_used"]
    assert "encoded-jpeg" not in str(value._inference)


def test_old_image_is_not_sent_after_a_worker_delay(monkeypatch):
    import time

    from myumiq_vrchat.visual_context import VisualContext

    visual = VisualContext(WorldState(), "old-image", time.perf_counter() - 10)

    def reply(config, messages, *args):
        assert messages[-1]["content"] == "何が見える？"
        assert "old-image" not in str(messages)
        assert '"image_used": false' in messages[0]["content"]
        return {"reply": "今の視界はまだ取得できていないよ。", "topic": "視界"}, 0.1

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue._request", reply)
    request_dialogue(
        LLMConfig(base_url="http://localhost:1", model="vision"),
        visual.world,
        {},
        {},
        {"working": {}},
        "何が見える？",
        visual=visual,
    )


def test_pending_assessment_cannot_generate_an_action_promise(monkeypatch):
    def forbidden(*args):
        raise AssertionError("pending action must not be promised by a model")

    monkeypatch.setattr("myumiq_vrchat.shared_dialogue._request", forbidden)
    result = request_dialogue(
        LLMConfig(base_url="http://localhost:1", model="test"),
        WorldState(),
        {},
        {},
        {"working": {}},
        "右に向いて",
        action_context={"status": "pending"},
    )
    assert result.topic == "確認"
    assert result._inference["adapter"] == "pending_assessment_acknowledgement"
    assert not result._inference["visual"]["image_used"]
