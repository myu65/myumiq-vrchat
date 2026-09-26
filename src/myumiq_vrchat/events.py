"""Bounded cross-loop events delivered to the motor thread at frame boundaries."""

from collections import deque
from dataclasses import dataclass
from threading import Lock

from .cognition import Decision


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    timestamp: float
    target: str | None = None
    decision: Decision | None = None
    # Set only by a source that identified the speaker; proximity is insufficient.
    speaker_id: str | None = None
    text: str | None = None
    source: str = "runtime"
    # Capture identity survives ASR latency and reply cancellation. Times use
    # the session monotonic clock, not wall time or recognition completion.
    input_session: str | None = None
    input_sequence: int | None = None
    utterance_id: str | None = None
    audio_start_at: float | None = None
    audio_end_at: float | None = None


class EventInbox:
    def __init__(self, max_size: int = 32):
        if max_size < 1:
            raise ValueError("event inbox must have positive capacity")
        self._max_size = max_size
        self._queue: deque[tuple[RuntimeEvent, bool]] = deque()
        self._lock = Lock()

    def publish(self, event: RuntimeEvent, *, reliable: bool = False) -> bool:
        """Never evict accepted reliable input; its producer retries on False.

        High-rate provisional events may replace other provisional events.
        This is bounded in-memory delivery, not a persistent journal.
        """
        with self._lock:
            if len(self._queue) >= self._max_size:
                index = next(
                    (i for i, (_, retained) in enumerate(self._queue) if not retained), None
                )
                if index is None:
                    return False
                del self._queue[index]
            self._queue.append((event, reliable))
            return True

    def drain(self) -> list[RuntimeEvent]:
        with self._lock:
            events = [event for event, _ in self._queue]
            self._queue.clear()
            return events
