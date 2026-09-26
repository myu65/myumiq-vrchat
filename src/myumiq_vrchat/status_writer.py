"""Best-effort status snapshots outside the motor thread; not a replay store."""

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class StatusWriter:
    def __init__(self, path):
        self.path = path
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="myumiq-status")
        self.pending = None
        self.lock = Lock()
        self.closed = False

    def submit(self, value):
        if self.closed:
            raise RuntimeError("status writer is closed")
        if self.pending:
            if not self.pending.done():
                return False
            self.pending.result()
        # Freeze mutable health dictionaries before crossing thread ownership.
        payload = json.dumps(value)
        self.pending = self.executor.submit(self._write, payload)
        return True

    def _write(self, payload):
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            for attempt in range(10):
                try:
                    with self.lock:
                        if not self.closed:
                            os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.002)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        with self.lock:
            self.closed = True
        try:
            if self.pending:
                self.pending.result(timeout=2.0)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
