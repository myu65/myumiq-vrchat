import json

import pytest

from myumiq_vrchat.dialogue_stream import SentenceReply
from myumiq_vrchat.shared_dialogue import is_immediate_stop


@pytest.mark.parametrize("ascii_escapes", [True, False])
def test_sentences_arrive_before_json_completion_without_overlap(ascii_escapes):
    stream = SentenceReply()
    reply = "こんにちは！今日は「いい日」だね。そう思うよ"
    raw = json.dumps({"reply": reply, "topic": "話題の文字は読まない"}, ensure_ascii=ascii_escapes)
    spoken = []
    for index, char in enumerate(raw):
        stream.feed(char)
        if stream.peek():
            assert index < len(raw) - 1
            spoken.append(stream.peek())
            stream.acknowledge()
    stream.finish(reply)
    spoken.append(stream.peek())
    stream.acknowledge()
    assert spoken == ["こんにちは！", "今日は「いい日」だね。", "そう思うよ"]
    assert stream.peek() is None and stream.submitted == reply


def test_property_order_and_escaped_quotes_do_not_leak_other_json_fields():
    stream = SentenceReply()
    stream.feed('{"topic":"こんにちは。", "reply":"うん。"}')
    assert stream.peek() is None
    stream.finish("うん。")
    assert stream.peek() == "うん。"
    stream = SentenceReply()
    stream.feed('{"reply":"こんにちは\\"うん。", "topic":"だめ。"}')
    assert stream.peek() == 'こんにちは"うん。'


def test_stream_bounds_and_final_prefix_validation():
    stream = SentenceReply()
    stream.feed('{"reply":"こんにちは。')
    with pytest.raises(ValueError, match="differs"):
        stream.finish("違う文章。")
    with pytest.raises(ValueError, match="too long"):
        SentenceReply().feed('{"reply":"' + "あ" * 181)


@pytest.mark.parametrize(
    "text", ["止まって", "ちょっと止まって！", "停止してください", "動かないで"]
)
def test_unambiguous_stop(text):
    assert is_immediate_stop(text)


@pytest.mark.parametrize(
    "text",
    [
        "止まってほしくない",
        "「止まって」と言ったらどうなる？",
        "止まっている人は誰？",
        "ストップウォッチ",
        "こんにちは",
    ],
)
def test_description_or_negation_is_not_a_stop_command(text):
    assert not is_immediate_stop(text)
