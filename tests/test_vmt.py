"""Lifecycle tests with retained output state; no live VMT/VRChat dependency."""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

# Run the source-only offline slice without installing a runtime package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from myumiq_vrchat.backends.vmt import (  # noqa: E402
    LifecycleError,
    OutputError,
    SafetyConfig,
    State,
    StopPolicy,
    VMTBackend,
)


@dataclass(frozen=True)
class Target:
    pose: str = "reaching"
    button: bool = True
    trigger: float = 0.75
    stick: tuple[float, float] = (0.4, -0.6)
    fingers: tuple[float, ...] = (1.0,) * 5


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class RetainingOutput:
    """Simulates values held indefinitely, including a previous producer's input."""

    def __init__(self):
        self.events = []
        self.fail = set()
        self.target = Target()
        self.pose = self.target.pose
        self.unrelated_tracker = "connected"
        self.calibrated = True
        self.closed = False

    def validate_stop(self, policy):
        if policy == StopPolicy.SAFE_POSE and not self.calibrated:
            raise ValueError("safe pose has no calibration")

    def validate_target(self, target):
        if not isinstance(target, Target) or not 0 <= target.trigger <= 1:
            raise ValueError("invalid complete target")

    def _record(self, name):
        self.events.append(name)
        if name in self.fail:
            raise OSError(f"injected {name} failure")

    def send_target(self, target):
        # Model a failure after some channels have already been sent.
        self.target, self.pose = target, target.pose
        self._record("target")

    def neutralize(self):
        self._record("neutral")
        self.target = Target(self.pose, False, 0.0, (0.0, 0.0), (0.0,) * 5)

    def stop_pose(self, policy):
        self._record(policy.value)
        self.pose = "calibrated_safe" if policy == StopPolicy.SAFE_POSE else "disabled"
        if policy == StopPolicy.RESET_ALL:
            self.unrelated_tracker = "disabled"

    def close(self):
        self.closed = True
        self._record("close")


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.output = RetainingOutput()
        self.backend = VMTBackend(
            self.output,
            SafetyConfig(heartbeat_timeout=2, target_timeout=1, retry_interval=0.25),
            clock=self.clock,
        )

    def activate(self):
        lease = self.backend.arm()
        self.backend.heartbeat(lease, 0, self.clock())
        self.backend.submit(lease, 0, self.clock(), Target())
        return lease

    def assert_neutral(self):
        self.assertFalse(self.output.target.button)
        self.assertEqual(self.output.target.trigger, 0)
        self.assertEqual(self.output.target.stick, (0, 0))
        self.assertEqual(self.output.target.fingers, (0,) * 5)

    def test_start_clears_retained_state_before_accepting_target(self):
        self.assertEqual(self.output.events, [])
        lease = self.backend.arm()
        self.assertEqual(self.output.events, ["neutral", "disable_owned"])
        self.assert_neutral()
        self.assertEqual(self.output.pose, "disabled")
        with self.assertRaises(LifecycleError):
            self.backend.submit(lease, 0, 0, Target())
        self.backend.heartbeat(lease, 0, 0)
        self.assertEqual(self.backend.status.state, State.WAITING)
        self.backend.submit(lease, 0, 0, Target())
        self.assertEqual(self.backend.status.state, State.ACTIVE)

    def test_heartbeat_cannot_keep_a_stale_target_alive(self):
        lease = self.activate()
        self.clock.now = 0.75
        self.backend.heartbeat(lease, 1, 0.75)
        self.clock.now = 0.999
        self.assertEqual(self.backend.poll().state, State.ACTIVE)
        self.clock.now = 1.0
        self.assertEqual(self.backend.poll().reason, "target_timeout")
        self.assert_neutral()
        self.assertEqual(self.output.pose, "disabled")
        self.assertEqual(self.output.unrelated_tracker, "connected")

    def test_fresh_targets_cannot_replace_heartbeat(self):
        lease = self.activate()
        for seq, now in enumerate((0.75, 1.5), 1):
            self.clock.now = now
            self.backend.submit(lease, seq, now, Target())
        self.clock.now = 2.0
        self.assertEqual(self.backend.poll().reason, "heartbeat_timeout")
        self.assert_neutral()

    def test_waiting_for_first_target_also_times_out(self):
        lease = self.backend.arm()
        self.clock.now = 0.75
        self.backend.heartbeat(lease, 0, 0.75)
        self.clock.now = 1
        self.assertEqual(self.backend.poll().state, State.TIMED_OUT)
        self.assertEqual(self.output.events.count("target"), 0)

    def test_late_packet_checks_deadline_without_prior_poll(self):
        lease = self.activate()
        self.clock.now = 1
        with self.assertRaises(LifecycleError):
            self.backend.submit(lease, 1, 1, Target())
        self.assertEqual(self.output.events.count("target"), 1)
        self.assertEqual(self.backend.status.state, State.TIMED_OUT)

    def test_restart_needs_new_lease_heartbeat_and_target(self):
        old = self.activate()
        self.clock.now = 1
        self.backend.poll()
        with self.assertRaises(LifecycleError):
            self.backend.heartbeat(old, 1, 1)
        new = self.backend.arm()
        self.assertNotEqual(old, new)
        with self.assertRaises(LifecycleError):
            self.backend.submit(old, 100, 1, Target())
        with self.assertRaises(LifecycleError):
            self.backend.submit(new, 0, 1, Target())
        self.assert_neutral()
        self.backend.heartbeat(new, 0, 1)
        self.backend.submit(new, 0, 1, Target("new_pose"))
        self.assertEqual(self.output.pose, "new_pose")

    def test_arm_cannot_extend_an_active_lease(self):
        self.activate()
        with self.assertRaises(LifecycleError):
            self.backend.arm()

    def test_replay_invalid_and_future_targets_do_not_refresh_deadline(self):
        lease = self.activate()
        self.clock.now = 0.5
        for seq, stamp, target in (
            (0, 0.5, Target()),
            (-1, 0.5, Target()),
            (1, 0.6, Target()),
            (1, float("nan"), Target()),
            (1, float("inf"), Target()),
            (1, -0.1, Target()),
            (True, 0.5, Target()),
            (1, 0.5, Target(trigger=2)),
        ):
            with self.subTest(seq=seq, stamp=stamp, target=target):
                with self.assertRaises(ValueError):
                    self.backend.submit(lease, seq, stamp, target)
        self.clock.now = 1
        self.assertEqual(self.backend.poll().reason, "target_timeout")
        self.assertEqual(self.output.events.count("target"), 1)

    def test_delayed_message_uses_generation_time_not_receive_time(self):
        lease = self.activate()
        self.clock.now = 0.75
        self.backend.submit(lease, 1, 0.25, Target())
        self.clock.now = 1.25
        self.assertEqual(self.backend.poll().reason, "target_timeout")

    def test_expired_queued_target_rejected_under_new_lease(self):
        self.clock.now = 10
        lease = self.backend.arm()
        self.backend.heartbeat(lease, 0, 10)
        with self.assertRaises(ValueError):
            self.backend.submit(lease, 0, 9, Target())
        self.assertEqual(self.output.events, ["neutral", "disable_owned"])

    def test_messages_must_be_generated_after_current_arm(self):
        self.clock.now = 10
        lease = self.backend.arm()
        # Within the timeout, but belongs to work prepared before this arm.
        with self.assertRaises(ValueError):
            self.backend.heartbeat(lease, 0, 9.75)
        self.backend.heartbeat(lease, 0, 10)
        with self.assertRaises(ValueError):
            self.backend.submit(lease, 0, 9.75, Target())
        self.assertEqual(self.output.events, ["neutral", "disable_owned"])

    def test_higher_sequence_cannot_move_generation_time_backwards(self):
        lease = self.activate()
        self.clock.now = 0.5
        self.backend.heartbeat(lease, 1, 0.5)
        self.backend.submit(lease, 1, 0.5, Target())
        self.clock.now = 0.75
        with self.assertRaises(ValueError):
            self.backend.heartbeat(lease, 2, 0.25)
        with self.assertRaises(ValueError):
            self.backend.submit(lease, 2, 0.25, Target())
        self.assertEqual(self.output.events.count("target"), 2)

    def test_replayed_heartbeat_does_not_extend_liveness(self):
        lease = self.activate()
        self.clock.now = 0.75
        with self.assertRaises(ValueError):
            self.backend.heartbeat(lease, 0, 0.75)
        self.backend.submit(lease, 1, 0.75, Target())
        self.clock.now = 1.5
        self.backend.submit(lease, 2, 1.5, Target())
        self.clock.now = 2
        self.assertEqual(self.backend.poll().reason, "heartbeat_timeout")

    def test_validation_time_cannot_push_target_past_deadline(self):
        lease = self.activate()
        self.output.validate_target = lambda target: setattr(self.clock, "now", 1)
        with self.assertRaises(LifecycleError):
            self.backend.submit(lease, 1, 0, Target())
        self.assertEqual(self.output.events.count("target"), 1)

    def test_timeout_retries_without_resending_old_targets(self):
        self.activate()
        self.clock.now = 1
        self.backend.poll()
        count = len(self.output.events)
        self.clock.now = 1.249
        self.backend.poll()
        self.assertEqual(len(self.output.events), count)
        self.clock.now = 1.25
        self.backend.poll()
        self.assertEqual(self.output.events[-2:], ["neutral", "disable_owned"])
        self.assertEqual(len(self.output.events), count + 2)
        self.assertEqual(self.output.events.count("target"), 1)

    def test_partial_send_failure_neutralizes_and_latches_fault(self):
        lease = self.activate()
        self.output.fail.add("target")
        with self.assertRaises(OutputError):
            self.backend.submit(lease, 1, 0, Target("partial"))
        self.assertEqual(self.backend.status.state, State.FAULTED)
        self.assert_neutral()
        self.assertEqual(self.output.pose, "disabled")
        self.assertIn("send_target", self.backend.status.errors[0])
        with self.assertRaises(LifecycleError):
            self.backend.heartbeat(lease, 1, 0)

    def test_deadline_crossed_during_send_is_stopped_immediately_after_return(self):
        lease = self.activate()
        original_send = self.output.send_target

        def slow_send(target):
            original_send(target)
            self.clock.now = 1.5

        self.output.send_target = slow_send
        self.clock.now = 0.5
        self.backend.submit(lease, 1, 0.5, Target())
        self.assertEqual(self.backend.status.state, State.TIMED_OUT)
        self.assert_neutral()
        self.assertEqual(self.output.events[-2:], ["neutral", "disable_owned"])

    def test_neutral_failure_still_stops_pose_and_poll_survives(self):
        self.activate()
        self.output.fail.add("neutral")
        self.clock.now = 1
        status = self.backend.poll()
        self.assertEqual(status.state, State.FAULTED)
        self.assertEqual(self.output.pose, "disabled")
        self.assertTrue(status.errors)
        self.output.fail.clear()
        self.clock.now = 1.25
        self.backend.poll()
        self.assert_neutral()
        self.assertEqual(self.backend.status.state, State.FAULTED)

    def test_failed_start_does_not_issue_a_working_lease(self):
        self.output.fail.add("disable_owned")
        with self.assertRaises(OutputError):
            self.backend.arm()
        self.assertEqual(self.backend.status.state, State.FAULTED)
        self.output.fail.clear()
        lease = self.backend.arm()
        self.backend.heartbeat(lease, 0, 0)
        self.backend.submit(lease, 0, 0, Target())
        self.assertEqual(self.backend.status.state, State.ACTIVE)

    def test_clean_shutdown_is_terminal_and_idempotent(self):
        lease = self.activate()
        self.backend.close()
        self.assertEqual(self.output.events[-3:], ["neutral", "disable_owned", "close"])
        self.assert_neutral()
        self.assertTrue(self.output.closed)
        count = len(self.output.events)
        self.backend.close()
        self.clock.now = 100
        self.assertEqual(self.backend.poll().state, State.CLOSED)
        self.assertEqual(len(self.output.events), count)
        with self.assertRaises(LifecycleError):
            self.backend.heartbeat(lease, 1, 100)
        with self.assertRaises(LifecycleError):
            self.backend.arm()

    def test_shutdown_before_arm_clears_previous_producer(self):
        self.backend.close()
        self.assert_neutral()
        self.assertTrue(self.output.closed)

    def test_shutdown_attempts_every_step_even_on_multiple_failures(self):
        self.activate()
        self.output.fail = {"neutral", "disable_owned", "close"}
        with self.assertRaises(OutputError):
            self.backend.close()
        self.assertEqual(self.output.events[-3:], ["neutral", "disable_owned", "close"])
        self.assertEqual(len(self.backend.status.errors), 3)
        self.assertEqual(self.backend.status.state, State.CLOSED)

    def test_interruption_during_shutdown_still_closes_output(self):
        self.activate()

        def interrupt():
            raise KeyboardInterrupt()

        self.output.neutralize = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.backend.close()
        self.assertTrue(self.output.closed)
        self.assertEqual(self.backend.status.state, State.CLOSED)

    def test_safe_pose_needs_calibration_and_never_replays_active_pose(self):
        backend = VMTBackend(
            self.output, SafetyConfig(stop_policy=StopPolicy.SAFE_POSE), clock=self.clock
        )
        self.output.calibrated = False
        with self.assertRaises(ValueError):
            backend.arm()
        self.assertEqual(self.output.events, [])
        self.output.calibrated = True
        lease = backend.arm()
        backend.heartbeat(lease, 0, 0)
        backend.submit(lease, 0, 0, Target())
        self.clock.now = 0.5
        backend.poll()
        self.assert_neutral()
        self.assertEqual(self.output.pose, "calibrated_safe")
        self.assertEqual(self.output.unrelated_tracker, "connected")

    def test_reset_is_only_available_with_exclusive_ownership(self):
        with self.assertRaises(ValueError):
            SafetyConfig(stop_policy=StopPolicy.RESET_ALL)
        backend = VMTBackend(
            self.output,
            SafetyConfig(stop_policy=StopPolicy.RESET_ALL, exclusive_vmt=True),
            clock=self.clock,
        )
        backend.arm()
        self.assertEqual(self.output.events, ["neutral", "reset_all"])
        self.assert_neutral()

    def test_invalid_config_is_rejected_without_output(self):
        for field in ("heartbeat_timeout", "target_timeout", "retry_interval"):
            for value in (0, -1, float("inf"), float("nan"), True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        SafetyConfig(**{field: value})
        with self.assertRaises(ValueError):
            SafetyConfig(stop_policy="reset_all", exclusive_vmt=True)
        with self.assertRaises(ValueError):
            SafetyConfig(exclusive_vmt="false")
        self.assertEqual(self.output.events, [])


if __name__ == "__main__":
    unittest.main()
