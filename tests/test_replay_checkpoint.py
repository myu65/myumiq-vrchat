import threading

import pytest

from myumiq_vrchat.body import BodyGoal, BodyTask, WorldState, simulated_body
from myumiq_vrchat.postures import posture_target
from myumiq_vrchat.replay import Observation, ReplayBuffer, WholeBodyTransition
from myumiq_vrchat.replay_checkpoint import ReplayCheckpoint
from myumiq_vrchat.whole_body import PARTS


def record(timestamp):
    pose = posture_target("standing")
    observation = Observation(
        timestamp=timestamp, world=WorldState(), body=simulated_body(pose, timestamp)
    )
    return WholeBodyTransition(
        observation=observation,
        next_observation=observation,
        body_goal=BodyGoal(
            tasks=(BodyTask(id="stand", kind="posture", effectors=PARTS, posture="standing"),),
            duration_s=1.0,
        ),
        action=pose,
        policy_id="fixture",
        outcome="simulated",
        environment="mock",
    ).model_dump_json()


def test_checkpoint_survives_before_shutdown_and_freezes_collector_state(tmp_path, monkeypatch):
    buffer = ReplayBuffer(4)
    buffer.add(record(1))
    saver = ReplayCheckpoint(tmp_path / "experience.jsonl")
    entered, release = threading.Event(), threading.Event()
    write = saver._write

    def delayed(records):
        entered.set()
        assert release.wait(2)
        write(records)

    monkeypatch.setattr(saver, "_write", delayed)
    try:
        assert saver.poll(buffer, 1)
        assert entered.wait(1)
        buffer.add(record(2))
        assert not saver.poll(buffer, 40)  # one disk operation, no accumulating queue
        release.set()
        saver.pending.result(timeout=2)
        restored = ReplayBuffer(4)
        restored.load_state(saver.path)
        assert [r.observation.timestamp for r in restored.get_data()] == [1]
        assert saver.poll(buffer, 40)
        saver.pending.result(timeout=2)
        restored.load_state(saver.path)
        assert [r.observation.timestamp for r in restored.get_data()] == [1, 2]
    finally:
        release.set()
        saver.close()


def test_closing_prevents_delayed_snapshot_overwriting_final_save(tmp_path, monkeypatch):
    buffer = ReplayBuffer(4)
    buffer.add(record(1))
    saver = ReplayCheckpoint(tmp_path / "experience.jsonl")
    entered, release = threading.Event(), threading.Event()
    write = saver._write

    def delayed(records):
        entered.set()
        assert release.wait(4)
        write(records)

    monkeypatch.setattr(saver, "_write", delayed)
    try:
        saver.poll(buffer, 1)
        assert entered.wait(1)
        with pytest.raises(TimeoutError):
            saver.close()
        buffer.add(record(2))
        buffer.save_state(saver.path)
        release.set()
        saver.pending.result(timeout=2)
        restored = ReplayBuffer(4)
        restored.load_state(saver.path)
        assert len(restored) == 2
    finally:
        release.set()
        saver.close()


def test_failed_write_preserves_previous_checkpoint_and_is_reported(tmp_path, monkeypatch):
    import myumiq_vrchat.replay_checkpoint as module

    buffer = ReplayBuffer(4)
    buffer.add(record(1))
    saver = ReplayCheckpoint(tmp_path / "experience.jsonl")
    buffer.save_state(saver.path)
    previous = saver.path.read_bytes()
    buffer.add(record(2))

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(module.os, "replace", fail)
    try:
        saver.poll(buffer, 1)
        with pytest.raises(OSError, match="disk unavailable"):
            saver.pending.result(timeout=2)
        with pytest.raises(OSError, match="disk unavailable"):
            saver.poll(buffer, 40)
        assert saver.path.read_bytes() == previous
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        with pytest.raises(OSError, match="disk unavailable"):
            saver.close()
