"""Bounded motor phase timing; no I/O or watchdog renewal on phase changes."""

import time
from collections import deque


class LoopStallTrace:
    """Opt-in native watchdog: trace a GIL stall without weakening output leases."""

    def __init__(self, path, *, timeout_s=0.35):
        import faulthandler

        self.handler = faulthandler
        self.timeout_s = timeout_s
        self.stream = path.open("w", encoding="utf-8")

    def tick(self):
        self.handler.dump_traceback_later(self.timeout_s, file=self.stream)

    def close(self):
        self.handler.cancel_dump_traceback_later()
        self.stream.close()


class LoopDiagnostics:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.phase = "startup"
        self.since = clock()
        self.maximum = {}
        self.slow = deque(maxlen=32)

    def enter(self, phase):
        now = self.clock()
        duration = now - self.since
        self.maximum[self.phase] = max(self.maximum.get(self.phase, 0.0), duration)
        if duration >= 0.1:
            self.slow.append(
                {"phase": self.phase, "started": self.since, "ended": now, "duration_s": duration}
            )
        self.phase, self.since = phase, now

    def snapshot(self):
        return {
            "phase": self.phase,
            "phase_started": self.since,
            "phase_elapsed_s": self.clock() - self.since,
            "max_duration_s": dict(self.maximum),
            "slow_phases": list(self.slow),
        }
