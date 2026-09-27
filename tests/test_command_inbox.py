import json
import time
from pathlib import Path
from threading import Event, current_thread, main_thread
from types import SimpleNamespace

import pytest

from myumiq_vrchat.body_console import Command, atomic_json
from myumiq_vrchat.command_inbox import CommandInbox


@pytest.mark.parametrize("operation", ["scan", "stop_stat", "receipt"])
def test_blocked_command_disk_io_does_not_stop_body_updates(tmp_path, monkeypatch, operation):
    import myumiq_vrchat.body_console as module

    blocked, release = Event(), Event()
    original_glob, original_exists = Path.glob, Path.exists
    session = tmp_path / operation

    def pause():
        assert current_thread() is not main_thread()
        blocked.set()
        assert release.wait(3.0)

    def glob(path, *args, **kwargs):
        if operation == "scan" and path == session / "commands" and not release.is_set():
            pause()
        return original_glob(path, *args, **kwargs)

    def exists(path, *args, **kwargs):
        if operation == "stop_stat" and path == session / "stop.txt" and not release.is_set():
            pause()
        return original_exists(path, *args, **kwargs)

    def write(path, value):
        if operation == "receipt" and path.parent == session / "receipts" and not release.is_set():
            pause()
        atomic_json(path, value)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(module, "atomic_json", write)
    updates = []

    class Owner:
        alive, failed = True, False

        def __init__(self, *args):
            pass

        def start(self):
            token = json.loads((session / "session.json").read_text())["session"]
            atomic_json(
                session / "commands" / "posture.json",
                Command(
                    session=token, issued=time.perf_counter(), kind="posture", name="crouching"
                ).model_dump(mode="json"),
            )

        def publish(self, target):
            if blocked.is_set() and not release.is_set():
                updates.append(target)
                if len(updates) >= 20:
                    release.set()

        def close(self):
            release.set()

    monkeypatch.setattr(module, "OutputSupervisor", Owner)
    module.run(
        SimpleNamespace(
            duration=1,
            live_config=None,
            hmd_serial=None,
            policy=None,
            reference_pose=None,
            session=session,
        )
    )
    assert blocked.is_set() and len(updates) >= 20
    assert json.loads((session / "receipts" / "posture.json").read_text())["accepted"]


def test_inbox_preserves_expiry_and_never_rereads_consumed_commands(tmp_path):
    (tmp_path / "commands").mkdir()
    received = Command(session="one", issued=time.perf_counter() - 3, kind="stop")
    atomic_json(tmp_path / "commands" / "a.json", received.model_dump(mode="json"))
    inbox = CommandInbox(tmp_path, atomic_json)
    inbox.start()
    try:
        until = time.monotonic() + 2
        rows = []
        while not rows and time.monotonic() < until:
            rows = inbox.poll()
            time.sleep(0.005)
        assert len(rows) == 1
        decoded = Command.model_validate_json(rows[0][1])
        assert decoded == received and not decoded.fresh("one", time.perf_counter())
        assert not inbox.stop_requested.is_set()  # Unvalidated/expired JSON cannot stop output.
        time.sleep(0.05)
        assert inbox.poll() == []
        (tmp_path / "stop.txt").touch()
        assert inbox.stop_requested.wait(1.0)
    finally:
        inbox.close()


def test_disk_pause_longer_than_output_lease_keeps_real_supervisor_active(tmp_path, monkeypatch):
    import myumiq_vrchat.body_console as module

    session = tmp_path / "supervised-stall"
    original = Path.glob
    stalled = Event()

    def glob(path, *args, **kwargs):
        if path == session / "commands" and not stalled.is_set():
            assert current_thread() is not main_thread()
            stalled.set()
            time.sleep(0.9)  # The real output supervisor's lease remains 500 ms.
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    module.run(
        SimpleNamespace(
            duration=2,
            live_config=None,
            hmd_serial=None,
            policy=None,
            reference_pose=None,
            session=session,
        )
    )
    events = [
        json.loads(line) for line in (session / "output-events.jsonl").read_text().splitlines()
    ]
    states = [event["state"] for event in events if "state" in event]
    assert stalled.is_set() and states == ["waiting", "active", "closed"]
    result = json.loads((session / "result.json").read_text())
    assert result["error"] is None and result["cleanup_errors"] == []


def test_receipt_backpressure_defers_commands_without_blocking_or_loss(tmp_path):
    (tmp_path / "commands").mkdir()
    for index in range(40):
        atomic_json(tmp_path / "commands" / f"{index:03}.json", {"index": index})
    writes = []
    inbox = CommandInbox(tmp_path, lambda path, value: writes.append(value))
    inbox._scan()
    for index in range(49):
        inbox.write(tmp_path / "receipt", index)
    assert inbox.poll() == []
    inbox._flush()
    delivered = []
    while len(delivered) < 40:
        delivered.extend(inbox.poll())
        inbox._scan()
    assert len({path.name for path, _ in delivered}) == 40
    assert writes == list(range(49))
    inbox.close()  # Also safe when startup failed before start().
