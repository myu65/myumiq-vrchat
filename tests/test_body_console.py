import json

import pytest

from myumiq_vrchat.body_console import Command, atomic_json, pulse_controls


@pytest.fixture
def synchronous_commands(monkeypatch):
    """Control-command delivery is deterministic in virtual-clock motor tests."""
    import myumiq_vrchat.body_console as module
    from myumiq_vrchat.command_inbox import CommandInbox

    class ImmediateInbox(CommandInbox):
        def start(self):
            pass

        def poll(self):
            self._scan()
            return super().poll()

        def write(self, path, value):
            self.write_json(path, value)

        def close(self):
            pass

    monkeypatch.setattr(module, "CommandInbox", ImmediateInbox)


def test_commands_are_bound_to_session_and_short_delivery_window():
    command = Command(session="one", issued=10.0, kind="menu", hand="left")
    assert command.fresh("one", 11.0)
    assert not command.fresh("two", 11.0)
    assert not command.fresh("one", 9.0)
    assert not command.fresh("one", 12.01)


@pytest.mark.parametrize(
    "payload",
    [
        dict(kind="aim", hand="both", pose=dict(position=[0, 0, 1])),
        dict(kind="posture"),
        dict(kind="menu", hand="both"),
        dict(kind="play", name="lying"),
        dict(kind="trigger", hand="left", duration_s=2),
        dict(kind="fist", hand="both", duration_s=2),
        dict(kind="fist"),
        dict(kind="open_hand", hand="other"),
        dict(kind="tracker_rates", rates=[0.0] * 65),
        dict(kind="tracker_rates", rates=[2.0] * 66),
        dict(kind="posture", name="standing", rates=[0.0] * 66),
        dict(kind="learned_pose"),
        dict(kind="learned_pose", goal_duration_s=21),
    ],
)
def test_malformed_or_unbounded_commands_are_rejected(payload):
    with pytest.raises(ValueError):
        Command.model_validate_json(json.dumps(dict(session="one", issued=1.0, **payload)))


def test_trigger_snapshot_has_consistent_index_and_released_other_inputs():
    controls = pulse_controls("trigger")
    assert controls.trigger_clicks[0] and controls.trigger_touches[0]
    assert controls.triggers[0] == controls.curls[1] == 1
    assert not any(controls.buttons)
    assert not any(controls.trigger_clicks[1:])


@pytest.mark.parametrize("kind,curl", [("fist", 1.0), ("open_hand", 0.0)])
@pytest.mark.parametrize("hand", ["left", "right", "both"])
def test_finger_diagnostic_uses_bounded_skeleton_only_input(kind, curl, hand):
    command = Command(session="one", issued=10.0, kind=kind, hand=hand, duration_s=0.5)
    controls = pulse_controls(command.kind)
    assert controls.curls == (curl,) * 5
    assert controls.model_copy(update={"curls": (0.0,) * 5}) == pulse_controls("open_hand")
    assert command.fresh("one", 11.0)
    assert not command.fresh("other", 11.0)


def test_atomic_command_commit_retries_transient_windows_sharing_error(tmp_path, monkeypatch):
    import myumiq_vrchat.body_console as module

    original = module.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("reader owns destination")
        original(source, destination)

    monkeypatch.setattr(module.os, "replace", replace)
    path = tmp_path / "command.json"
    atomic_json(path, {"kind": "stop"})
    assert json.loads(path.read_text()) == {"kind": "stop"}
    assert len(attempts) == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_slow_status_disk_write_cannot_stall_motor_or_overwrite_stopped_state(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from threading import Event
    from types import SimpleNamespace

    import myumiq_vrchat.body_console as module

    writing, release = Event(), Event()
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name.startswith("status.json.") and not writing.is_set():
            writing.set()
            assert release.wait(2.0)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    observations = []

    class Owner:
        alive, failed = True, False

        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            release.set()

        def publish(self, target):
            if writing.is_set() and not release.is_set():
                observations.append(target)
                if len(observations) == 10:
                    release.set()

    monkeypatch.setattr(module, "OutputSupervisor", Owner)
    session = tmp_path / "slow-status"
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
    assert len(observations) == 10
    assert json.loads((session / "status.json").read_text())["running"] is False


def test_output_failure_is_captured_before_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import myumiq_vrchat.body_console as module

    class FailedOwner:
        alive = False
        failed = True

        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            progress = json.loads((session / "cleanup-progress.json").read_text())
            assert progress["phase"] == "output" and not progress["finished"]
            self.failed = False  # prove result captured the pre-cleanup state

        def publish(self, target):
            pytest.fail("must not publish after output failure")

    monkeypatch.setattr(module, "OutputSupervisor", FailedOwner)
    session = tmp_path / "failed-output"
    args = SimpleNamespace(
        duration=1,
        live_config=None,
        hmd_serial=None,
        policy=None,
        reference_pose=None,
        session=session,
    )
    with pytest.raises(RuntimeError, match="output supervisor stopped"):
        module.run(args)
    result = json.loads((session / "result.json").read_text())
    assert result["failure"]["output_failed"] is True
    assert result["failure"]["output_alive"] is False
    assert len(result["failure"]["device_feedback"]) == 11
    assert all(x["valid"] for x in result["failure"]["device_feedback"].values())
    assert result["failure"]["loop_diagnostics"]["phase"] == "feedback_validation"
    assert result["cleanup_errors"] == []
    assert json.loads((session / "cleanup-progress.json").read_text())["finished"] is True
    assert json.loads((session / "status.json").read_text())["running"] is False


def test_tracker_lease_records_next_feedback_and_holds_after_expiry(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    import myumiq_vrchat.body_console as module

    session = tmp_path / "rates"
    published = []
    original_observer = module.simulated_body

    def capture(target, now):
        # Match a real readback adapter: stamp the observation during capture.
        time.sleep(0.001)
        return original_observer(target, time.perf_counter())

    monkeypatch.setattr(module, "simulated_body", capture)

    class Owner:
        alive = True
        failed = False

        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def publish(self, target):
            published.append(target)
            if len(published) == 1:
                token = json.loads((session / "session.json").read_text())["session"]
                command = Command(
                    session=token,
                    issued=time.perf_counter(),
                    kind="tracker_rates",
                    rates=tuple([0.05, 0.0, 0.0, 0.0, 0.0, 0.0] * 11),
                    duration_s=0.15,
                )
                atomic_json(session / "commands" / "rates.json", command.model_dump(mode="json"))

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
    rows = [json.loads(line) for line in (session / "experience.jsonl").read_text().splitlines()]
    rates = [r for r in rows if r["policy_id"] == "supervised-tracker-rates"]
    assert rates
    for row in rates:
        assert row["next_observation"]["timestamp"] > row["observation"]["timestamp"]
        assert row["observation"]["timestamp"] >= row["observation"]["body"]["head"]["timestamp"]
        assert len(row["intent_metadata"]["rates"]) == 66
        assert row["reward"] is None and row["learning"] is None
    assert published[-1].head.position[0] > published[0].head.position[0]
    assert published[-1].head.position == published[-10].head.position


@pytest.mark.parametrize("cancel", ["manual", "expiry"])
@pytest.mark.usefixtures("synchronous_commands")
def test_learned_whole_body_goal_switch_and_manual_override_record_next_feedback(
    tmp_path, monkeypatch, cancel
):
    from types import SimpleNamespace

    import numpy as np

    import myumiq_vrchat.body_console as module
    import myumiq_vrchat.tracker_policy as policies
    from myumiq_vrchat.tracker_action import integrate_tracker_action
    from myumiq_vrchat.whole_body import state_target, vector

    session = tmp_path / "learned-pose"
    published = []
    calls = []

    class Clock:
        now = 1.0

        def perf_counter(self):
            return self.now

        def time(self):
            return self.now

        def sleep(self, dt):
            self.now += 0.02

    clock = Clock()

    class Actor:
        manifest = {"sha256": "a" * 64}

        def __init__(self, path):
            self.previous = np.zeros(66)

        def step(self, body, goal, dt):
            current = state_target(body)
            calls.append((current, goal, self.previous.copy()))
            obs = policies.observation(current, goal, self.previous, dt)
            rates = np.tile([0.0, 0.0, -0.05, 0.0, 0.0, 0.0], 11)
            target = integrate_tracker_action(current, rates.reshape(11, 6), dt)
            self.previous = rates.copy()
            return target, obs, rates

    class Owner:
        alive = True
        failed = False

        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def publish(self, target):
            published.append(target)
            n = len(published)
            if n in (1, 5) or (n == 9 and cancel == "manual"):
                token = json.loads((session / "session.json").read_text())["session"]
                payload = (
                    {"kind": "manual"}
                    if n == 9
                    else dict(
                        kind="learned_pose",
                        pose_goal=module.posture_target("crouching" if n == 1 else "sitting_floor"),
                        goal_duration_s=1.0,
                    )
                )
                command = Command(session=token, issued=clock.now, **payload)
                atomic_json(session / "commands" / f"{n:03}.json", command.model_dump(mode="json"))

    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(module, "OutputSupervisor", Owner)
    monkeypatch.setattr(policies, "TrackerActor", Actor)
    module.run(
        SimpleNamespace(
            duration=2,
            live_config=None,
            hmd_serial=None,
            policy=None,
            reference_pose=None,
            session=session,
            tracker_policy=tmp_path / "actor.pt",
        )
    )
    assert len(calls) == 8 if cancel == "manual" else len(calls) > 45
    # The fifth learned step sees a new goal but the pose and prior action continue.
    np.testing.assert_allclose(vector(calls[4][0]), vector(published[4]))
    np.testing.assert_allclose(calls[4][2], calls[3][2])
    assert calls[4][1] != calls[3][1]
    np.testing.assert_allclose(vector(published[-1]), vector(published[len(calls)]))
    rows = [json.loads(line) for line in (session / "experience.jsonl").read_text().splitlines()]
    learned = [row for row in rows if row["learning"] is not None]
    assert len(learned) == len(calls)
    assert learned[-1]["learning"]["truncated"] is True
    assert any(row["learning"]["truncated"] for row in learned[:-1])  # goal switch
    for row in learned:
        assert row["next_observation"]["timestamp"] > row["observation"]["timestamp"]
        assert row["learning"]["reward_scope"] == "tracker_geometry"
        assert row["reward"] == pytest.approx(sum(row["learning"]["reward_components"].values()))
        assert row["policy_id"].startswith("tracker-sac:")


@pytest.mark.usefixtures("synchronous_commands")
@pytest.mark.parametrize("input_kind", ["drive", "fist"])
@pytest.mark.parametrize("release", ["manual", "expiry"])
def test_manual_override_or_expiry_releases_active_input(
    tmp_path, monkeypatch, input_kind, release
):
    from types import SimpleNamespace

    import myumiq_vrchat.body_console as module
    from myumiq_vrchat.body import Controls
    from myumiq_vrchat.whole_body import PARTS

    session = tmp_path / "manual-hold"
    published = []

    class Clock:
        now = 1.0

        def perf_counter(self):
            return self.now

        def sleep(self, dt):
            self.now += 0.02

    clock = Clock()

    class Owner:
        alive = True
        failed = False

        def __init__(self, *args):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def publish(self, target):
            published.append(target)
            payload = {
                1: dict(kind="posture", name="lying"),
                20: (
                    dict(kind="drive", direction="forward", duration_s=0.5)
                    if input_kind == "drive"
                    else dict(kind="fist", hand="both", duration_s=0.5)
                ),
                21: dict(kind="manual") if release == "manual" else None,
            }.get(len(published))
            if payload:
                token = json.loads((session / "session.json").read_text())["session"]
                command = Command(session=token, issued=clock.now, **payload)
                atomic_json(
                    session / "commands" / f"{len(published):03}.json",
                    command.model_dump(mode="json"),
                )

    monkeypatch.setattr(module, "time", clock)
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
    moving, held = published[20], published[21]
    assert moving.head != published[0].head
    if input_kind == "drive":
        assert moving.left.controls.sticks[1][1] > 0
    else:
        assert moving.left.controls.curls == moving.right.controls.curls == (1.0,) * 5
    assert published[-1].left.controls == published[-1].right.controls == Controls()
    if release == "expiry":
        return
    for target in (held, published[-1]):
        assert target.left.controls == target.right.controls == Controls()
        for part in PARTS:
            assert target.pose_for(part).position == pytest.approx(moving.pose_for(part).position)
            assert target.pose_for(part).orientation == pytest.approx(
                moving.pose_for(part).orientation
            )
