import json
import math
import weakref
from concurrent.futures import Future

import pytest

from myumiq_vrchat.body import WorldState, rest_target, simulated_body
from myumiq_vrchat.cognition import Decision, Goal, LLMConfig
from myumiq_vrchat.events import EventInbox, RuntimeEvent
from myumiq_vrchat.replay import Observation, ReplayBuffer, Transition, summarize
from myumiq_vrchat.runtime import BodyAgent, run_session


def test_agent_keeps_stepping_while_llm_pending_then_records_goal():
    future = Future()
    agent = BodyAgent(future, 60.0)
    records = []

    class Collector:
        def collect(self, record):
            assert isinstance(record, str)
            records.append(Transition.model_validate_json(record))

    agent.collector = Collector()
    for n in range(10):
        observation = Observation(
            timestamp=n / 60, world=WorldState(), body=simulated_body(rest_target(), n / 60)
        )
        if n == 5:
            future.set_result(
                Decision(
                    goal=Goal(skill="WAVE", duration_s=2.0, hand="left"),
                    source="local_llm",
                    model="fixture",
                )
            )
        agent.step(observation)
    assert len(records) == 9
    assert records[0].command.goal.skill == "WAIT"
    assert records[-1].command.goal.skill == "WAVE"
    assert records[-1].next_observation.body.head.source == "simulated"
    assert records[-1].decision.source == "local_llm"


def test_invalid_llm_response_stops_agent_instead_of_reusing_motion():
    future = Future()
    future.set_exception(ValueError("invalid intention"))
    agent = BodyAgent(future, 60.0)
    observation = Observation(
        timestamp=1.0, world=WorldState(), body=simulated_body(rest_target(), 1.0)
    )
    with pytest.raises(ValueError, match="invalid intention"):
        agent.step(observation)
    assert agent.error == "invalid intention"


def test_late_frame_bounds_motion_without_extending_the_goal():
    agent = BodyAgent(
        Decision(goal=Goal(skill="WAVE", hand="right", duration_s=1), source="fixed"), 60.0
    )
    agent.collector = type("Collector", (), {"collect": lambda self, item: None})()
    rest = rest_target()

    def observation(t, target):
        return Observation(timestamp=t, world=WorldState(), body=simulated_body(target, t))

    before = agent.step(observation(1.0, rest))
    after = agent.step(observation(1.2, before))
    assert math.dist(before.right.pose.position, after.right.pose.position) <= 0.060001
    expired = agent.step(observation(2.1, after))
    assert expired.right.controls == rest.right.controls
    assert expired.right.pose == after.right.pose


def test_pending_replay_does_not_retain_old_body_graphs(tmp_path):
    queued = []
    agent = BodyAgent(Decision(goal=Goal(skill="WAIT", duration_s=1), source="fixed"), 60)
    agent.collector = type(
        "Collector", (), {"collect": lambda self, record: queued.append(record)}
    )()
    old_body = None
    for n in range(4):
        obs = Observation(
            timestamp=1 + n / 60, world=WorldState(), body=simulated_body(rest_target(), 1 + n / 60)
        )
        if n == 0:
            old_body = weakref.ref(obs.body)
        agent.step(obs)
    assert len(queued) == 3 and old_body() is None
    buffer = ReplayBuffer(2)
    for record in queued:
        buffer.add(record)
    assert [x.observation.timestamp for x in buffer.get_data()] == pytest.approx(
        [1 + 1 / 60, 1 + 2 / 60]
    )
    buffer.save_state(tmp_path / "encoded")
    restored = ReplayBuffer(2)
    restored.load_state(tmp_path / "encoded")
    assert restored.get_data() == buffer.get_data()


def test_pamiq_mock_session_and_replay_roundtrip(tmp_path):
    output = tmp_path / "mock-run"
    result = run_session(
        output,
        WorldState(),
        Decision(goal=Goal(skill="WAVE", hand="right", duration_s=0.5), source="fixed"),
        duration=1.0,
        hz=30.0,
    )
    assert result["frames_attempted"] > 5
    assert result["error"] is None
    files = list(output.glob("states/*/data/experience/buffer.jsonl"))
    assert len(files) == 1
    first = ReplayBuffer(100)
    first.load_state(files[0])
    assert len(first) == result["transitions"]
    first.save_state(tmp_path / "roundtrip")
    second = ReplayBuffer(100)
    second.load_state(tmp_path / "roundtrip")
    assert first.get_data() == second.get_data()
    assert summarize(files[0])["head_sources"] == ["simulated"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema_version":999}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        second.load_state(bad)
    assert first.get_data() == second.get_data()
    events = [
        json.loads(line) for line in (output / "output-events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["state"] == "closed"
    assert not events[-1]["errors"]


def test_speech_onset_changes_body_intent_before_transcription():
    events = EventInbox()
    agent = BodyAgent(
        Decision(goal=Goal(skill="WAVE", hand="right", duration_s=4), source="fixed"),
        60,
        events=events,
    )
    agent.collector = type("Collector", (), {"collect": lambda self, item: None})()
    events.publish(RuntimeEvent("speech_started", 1.0))
    observation = Observation(
        timestamp=1.0, world=WorldState(), body=simulated_body(rest_target(), 1.0)
    )
    action = agent.step(observation)
    assert agent.decision.goal.skill == "WAIT"
    assert action.right.controls == rest_target().right.controls


@pytest.mark.parametrize("speaker", [None, "person", "unknown"])
def test_conversation_keeps_body_until_completion_then_autonomy_resumes(monkeypatch, speaker):
    from myumiq_vrchat.autonomy import AutonomousPlanner
    from myumiq_vrchat.body import WorldObject

    requests = []

    def request(*args):
        future = Future()
        requests.append(future)
        return future

    monkeypatch.setattr("myumiq_vrchat.autonomy.decide_async", request)
    planner = AutonomousPlanner(LLMConfig(base_url="http://localhost:1/v1", model="test"))
    events = EventInbox()
    agent = BodyAgent(
        Decision(goal=Goal(skill="WAIT", duration_s=1), source="fixed"),
        60,
        planner=planner,
        events=events,
    )
    agent.collector = type("Collector", (), {"collect": lambda self, item: None})()

    def step(now):
        world = WorldState(
            objects=(
                WorldObject(
                    name="person", kind="player", position=(0.5, 0.0, 1.5), source="fixture"
                ),
            )
        )
        agent.step(Observation(timestamp=now, world=world, body=simulated_body(rest_target(), now)))

    step(0)
    assert len(requests) == 1
    events.publish(RuntimeEvent("speech_started", 0.1))
    step(0.1)
    requests[0].set_result(
        Decision(goal=Goal(skill="WAVE", hand="left", duration_s=1), source="local_llm")
    )
    step(0.2)
    step(0.8)
    assert len(requests) == 1
    assert agent.decision.goal.skill == "WAIT"
    reply = Decision(goal=Goal(skill="WAVE", hand="right", duration_s=1), source="local_llm")
    events.publish(RuntimeEvent("conversation_decision", 1, decision=reply, speaker_id=speaker))
    step(1)
    assert planner.memory.familiarity == ({"person": 0.05} if speaker == "person" else {})
    assert planner.memory.affinity == {}
    assert planner.memory.recent[-1]["outcome"] == "conversation_reply_planned"
    step(1.1)
    step(1.9)
    assert agent.decision == reply
    assert len(requests) == 1
    step(2.1)
    assert len(requests) == 2
    assert agent.decision.goal.skill == "WAIT"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://127.0.0.1.evil/v1",
        "http://user:password@localhost/v1",
        "file:///a",
    ],
)
def test_llm_endpoint_is_local_and_not_credential_bearing(url):
    with pytest.raises(ValueError):
        LLMConfig(base_url=url, model="test")


def test_machine_files_stay_outside_git_checkouts(tmp_path):
    from myumiq_vrchat.cli import outside_repo

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../actual-git")
    with pytest.raises(ValueError, match="outside"):
        outside_repo(checkout / "config.json")
    assert outside_repo(tmp_path / "local" / "config.json") == tmp_path / "local" / "config.json"
