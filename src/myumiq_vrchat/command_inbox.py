"""File IPC off the motor thread; original issue times survive delivery delays."""

from queue import Empty, Full, Queue
from threading import Event, Thread


class CommandInbox:
    def __init__(self, root, write_json):
        self.root, self.write_json = root, write_json
        self.incoming = Queue(maxsize=32)
        self.outgoing = Queue(maxsize=64)
        self.stop_requested = Event()
        self.closing = Event()
        self.error = None
        self.seen = set()
        self.thread = Thread(target=self._work, name="myumiq-command-io", daemon=True)

    def start(self):
        self.thread.start()

    def _scan(self):
        if (self.root / "stop.txt").exists():
            self.stop_requested.set()
        for path in sorted((self.root / "commands").glob("*.json")):
            if path.name in self.seen:
                continue
            if self.incoming.full() or self.closing.is_set():
                break
            try:
                with path.open("r", encoding="utf-8") as stream:
                    encoded = stream.read(65537)
            except PermissionError:
                continue
            if len(encoded) > 65536:
                raise ValueError("command file exceeds size limit")
            self.incoming.put_nowait((path, encoded))
            self.seen.add(path.name)

    def _flush(self):
        while True:
            try:
                path, value = self.outgoing.get_nowait()
            except Empty:
                return
            self.write_json(path, value)

    def _work(self):
        try:
            while not self.closing.is_set():
                self._scan()
                self._flush()
                self.closing.wait(0.02)
            self._flush()
        except Exception as exc:
            self.error = exc

    def poll(self):
        if self.error is not None:
            raise RuntimeError(f"command file adapter failed: {self.error}")
        # At most two writes per accepted command, plus bounded work per frame.
        if self.outgoing.qsize() > 48:
            return []
        result = []
        for _ in range(8):
            try:
                result.append(self.incoming.get_nowait())
            except Empty:
                break
        return result

    def write(self, path, value):
        try:
            self.outgoing.put_nowait((path, value))
        except Full:
            raise RuntimeError("command receipt queue is full") from None

    def close(self):
        self.closing.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("command file adapter cleanup timed out")
        if self.error is not None:
            raise RuntimeError(f"command file adapter failed: {self.error}")
