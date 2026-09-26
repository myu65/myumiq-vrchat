"""Bounded disk checkpoints of the existing PAMIQ buffer, never an action queue."""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class ReplayCheckpoint:
    def __init__(self, path, *, interval_s=30.0):
        if interval_s <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.path, self.interval_s = path, interval_s
        self.next_at = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="myumiq-replay-save")
        self.pending = None
        self.guard = Lock()
        self.closed = False

    def poll(self, buffer, now):
        if self.closed:
            raise RuntimeError("replay checkpoint is closed")
        if self.pending:
            if not self.pending.done():
                return False
            self.pending.result()
        if now < self.next_at:
            return False
        records = buffer.encoded_snapshot()
        self.pending = self.executor.submit(self._write, records)
        self.next_at = now + self.interval_s
        return True

    def _write(self, records):
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                for line in records:
                    stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(10):
                try:
                    with self.guard:
                        if not self.closed:
                            os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.002)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        with self.guard:
            self.closed = True
        try:
            if self.pending:
                self.pending.result(timeout=2.0)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
