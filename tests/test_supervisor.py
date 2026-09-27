import json
import socket
import struct
import subprocess
import sys
import time

import pytest
from pythonosc.osc_packet import OscPacket
from test_osc import configuration

from myumiq_vrchat.backends.supervisor import OutputSupervisor
from myumiq_vrchat.backends.vmt import SafetyConfig
from myumiq_vrchat.body import ActuationTarget, Controls, HandTarget, rest_target


def test_compositor_repeats_pose_and_expires_inputs_without_stopping_pose_owner(
    tmp_path, endpoints
):
    from myumiq_vrchat.actuation import HandInputCommand, LocomotionCommand, PoseTarget

    vmt, hmd = endpoints
    supervisor = OutputSupervisor(
        tmp_path / "compositor.jsonl", configuration(vmt.getsockname()[1], hmd.getsockname()[1])
    ).start()
    pose = PoseTarget.from_target(rest_target())
    try:
        supervisor.publish_pose(pose)
        supervisor.publish_locomotion(LocomotionCommand(forward=0.4))
        supervisor.publish_hands(HandInputCommand(left=Controls(buttons=(True,) + (False,) * 17)))
        early = receive_until(vmt, 0.1)
        # A single producer pose is emitted on multiple servo ticks.
        assert sum(address == "/VMT/Raw/Driver" for address, _ in early) >= 4
        assert any(address == "/VMT/Input/Button" and params[-1] == 1 for address, params in early)
        for _ in range(3):
            supervisor.publish_pose(pose)
            later = receive_until(vmt, 0.08)
        latest = {}
        for address, params in later:
            if address.startswith("/VMT/Input/"):
                latest[(address, params[0], params[1])] = params[3:]
        assert latest and all(all(value == 0 for value in values) for values in latest.values())
        assert supervisor.alive  # Fresh poses outlive the released input commands.
    finally:
        supervisor.close()


def receive_until(sock, seconds):
    messages = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(65536)
            messages.extend((m.message.address, m.message.params) for m in OscPacket(data).messages)
        except TimeoutError:
            pass
    return messages


def test_joint_horizon_is_played_by_worker_during_producer_stall(tmp_path, endpoints):
    from test_joint_trajectory import trajectory

    from myumiq_vrchat.backends.osc import TrackerBinding
    from myumiq_vrchat.body import EXTRA_PARTS

    vmt, hmd = endpoints
    pose, command = trajectory()
    config = configuration(vmt.getsockname()[1], hmd.getsockname()[1]).model_copy(
        update={
            "trackers": tuple(
                TrackerBinding(part=p, index=i + 3) for i, p in enumerate(EXTRA_PARTS)
            ),
            "safe_target": pose,
            "full_body_envelope": 2.5,
        }
    )
    supervisor = OutputSupervisor(tmp_path / "joint-servo.jsonl", config).start()
    try:
        now = time.perf_counter()
        command = command.model_copy(
            update={
                "origin": command.origin.model_copy(update={"time": now}),
                "knots": tuple(
                    k.model_copy(update={"time": now + k.time - 1}) for k in command.knots
                ),
            }
        )
        supervisor.publish_trajectory(pose, command)
        messages = receive_until(vmt, 0.38)
        positions = [
            tuple(params[3:6])
            for address, params in messages
            if address == "/VMT/Raw/Driver" and params[0] == 1 and params[1] != 0
        ]
        assert len(set(positions)) >= 5
        assert supervisor.alive
        receive_until(vmt, 0.3)
        assert supervisor.failed  # The trajectory cannot renew the producer's lease.
    finally:
        supervisor.close()


@pytest.fixture
def endpoints():
    vmt, hmd = (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
    )
    for sock in (vmt, hmd):
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(0.02)
    yield vmt, hmd
    vmt.close()
    hmd.close()


def pressed_target():
    rest = rest_target()
    return ActuationTarget(
        head=rest.head,
        left=rest.left,
        right=HandTarget(
            pose=rest.right.pose,
            controls=Controls(
                buttons=(True,) * 18,
                triggers=(0.8,) * 9,
                sticks=((0.5, -0.5),) * 4,
                curls=(1.0,) * 5,
            ),
        ),
    )


def assert_released(messages):
    latest = {}
    for address, params in messages:
        if address.startswith("/VMT/Input/"):
            latest[(address, params[0], params[1])] = params[3:]
    assert len(latest) == 2 * (18 * 2 + 9 * 3 + 4 * 3)
    assert all(all(x == 0 for x in values) for values in latest.values())
    last_poses = {}
    for address, params in messages:
        if address == "/VMT/Raw/Driver":
            last_poses[params[0]] = params[1]
    assert last_poses == {1: 0, 2: 0}


def test_stalled_producer_releases_controls_and_disables_owned_devices(tmp_path, endpoints):
    vmt, hmd = endpoints
    config = configuration(vmt.getsockname()[1], hmd.getsockname()[1])
    supervisor = OutputSupervisor(
        tmp_path / "watchdog.jsonl", config, SafetyConfig(heartbeat_timeout=0.3, target_timeout=0.3)
    ).start()
    try:
        supervisor.publish(pressed_target())
        messages = receive_until(vmt, 0.75)
        assert any(a == "/VMT/Input/Button" and p[0] == 2 and p[3] == 1 for a, p in messages)
        assert_released(messages)
        received_head = []
        while True:
            try:
                received_head.append(struct.unpack("<6d", hmd.recvfrom(100)[0]))
            except TimeoutError:
                break
        assert received_head[-1][:3] == pytest.approx((0.0, 0.0, 160.0))
        events = [json.loads(x) for x in (tmp_path / "watchdog.jsonl").read_text().splitlines()]
        stopped = next(x for x in events if x.get("state") == "timed_out")
        active = next(x for x in events if x.get("state") == "active")
        assert 0.25 <= stopped["timestamp"] - active["timestamp"] <= 0.6
    finally:
        supervisor.close()


def test_heartbeat_only_does_not_refresh_the_target(tmp_path):
    path = tmp_path / "heartbeat.jsonl"
    supervisor = OutputSupervisor(
        path, safety=SafetyConfig(heartbeat_timeout=0.6, target_timeout=0.2)
    ).start()
    try:
        supervisor.publish(rest_target())
        for _ in range(3):
            time.sleep(0.05)
            supervisor.heartbeat()
        time.sleep(0.1)
    finally:
        supervisor.close()
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert any(x.get("reason") == "target_timeout" for x in events)


def test_expiry_inside_freshness_validation_preserves_timeout_cleanup(tmp_path, monkeypatch):
    from queue import Queue
    from threading import Event, Thread
    from types import SimpleNamespace

    from myumiq_vrchat.backends import supervisor as module

    class BoundaryBackend(module.VMTBackend):
        def submit(self, lease, sequence, stamp, target):
            if sequence == 1:
                self._clock = lambda: time.perf_counter() + 1
                raise ValueError("message must be fresh in the shared monotonic clock domain")
            return super().submit(lease, sequence, stamp, target)

    monkeypatch.setattr(module, "VMTBackend", BoundaryBackend)
    ready, stop, failed = Queue(), Event(), Event()
    log = tmp_path / "boundary.jsonl"
    worker = Thread(
        target=module._worker,
        args=(
            None,
            SafetyConfig(),
            "test-token",
            SimpleNamespace(send=ready.put, close=lambda: None),
            stop,
            failed,
            log,
        ),
    )
    worker.start()
    try:
        port = ready.get(timeout=5)["port"]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(
                json.dumps(
                    dict(
                        kind="frame",
                        token="test-token",
                        sequence=0,
                        timestamp=time.perf_counter(),
                        target=rest_target().model_dump(mode="json"),
                    )
                ).encode(),
                ("127.0.0.1", port),
            )
        worker.join(4)
        assert not worker.is_alive() and failed.is_set()
        events = [json.loads(row) for row in log.read_text().splitlines()]
        assert [row["state"] for row in events if "state" in row] == [
            "waiting",
            "active",
            "timed_out",
            "closed",
        ]
        assert not any("error" in row or "rejected" in row for row in events)
        assert events[-1]["errors"] == []
    finally:
        stop.set()
        worker.join(5)


def test_safe_pose_timeout_releases_inputs_without_disconnecting(tmp_path, endpoints):
    from myumiq_vrchat.backends.osc import LiveConfig

    vmt, hmd = endpoints
    data = configuration(vmt.getsockname()[1], hmd.getsockname()[1]).model_dump()
    data.update(stop_policy="safe_pose", heartbeat_timeout=0.3, target_timeout=0.3)
    config = LiveConfig.model_validate_json(json.dumps(data))
    supervisor = OutputSupervisor(tmp_path / "safe.jsonl", config).start()
    try:
        supervisor.publish(pressed_target())
        messages = receive_until(vmt, 0.75)
        latest_inputs, latest_poses = {}, {}
        for address, params in messages:
            if address.startswith("/VMT/Input/"):
                latest_inputs[(address, params[0], params[1])] = params[3:]
            if address == "/VMT/Raw/Driver":
                latest_poses[params[0]] = params
        assert len(latest_inputs) == 2 * (18 * 2 + 9 * 3 + 4 * 3)
        assert all(all(x == 0 for x in values) for values in latest_inputs.values())
        assert {i: p[1] for i, p in latest_poses.items()} == {1: 5, 2: 6}
        assert latest_poses[2][3:6] == pytest.approx([0.25, 1.15, -0.2])
        assert supervisor.failed
    finally:
        supervisor.close()


def test_producer_process_abrupt_exit_still_runs_neutralization(tmp_path, endpoints):
    vmt, hmd = endpoints
    config_path = tmp_path / "config.json"
    config_path.write_text(
        configuration(vmt.getsockname()[1], hmd.getsockname()[1]).model_dump_json()
    )
    # Only this test producer exits abruptly. The independent output process survives it.
    script = tmp_path / "crash_producer.py"
    script.write_text(
        """import os, sys, time
from pathlib import Path
from myumiq_vrchat.backends.osc import LiveConfig
from myumiq_vrchat.backends.supervisor import OutputSupervisor
from myumiq_vrchat.body import rest_target
if __name__ == "__main__":
    config = LiveConfig.model_validate_json(Path(sys.argv[1]).read_text())
    worker = OutputSupervisor(Path(sys.argv[2]), config).start()
    worker.publish(rest_target())
    time.sleep(.08)
    os._exit(17)
""",
        encoding="utf-8",
    )
    log_path = tmp_path / "crash.jsonl"
    process = subprocess.Popen(
        [sys.executable, str(script), str(config_path), str(log_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    messages = receive_until(vmt, 3.5)
    assert process.wait(timeout=3) == 17
    # Spawn time varies under concurrent rendering/training. The orphan's
    # bounded cleanup starts at its watchdog expiry, not at Popen above.
    deadline = time.perf_counter() + 3
    while True:
        events = [json.loads(x) for x in log_path.read_text().splitlines()]
        if events[-1].get("state") == "closed" or time.perf_counter() >= deadline:
            break
        time.sleep(0.02)
    assert any(x.get("state") == "active" for x in events)
    assert any(x.get("state") == "timed_out" for x in events), events
    assert events[-1]["state"] == "closed"
    assert not events[-1]["errors"]
    assert_released(messages)
