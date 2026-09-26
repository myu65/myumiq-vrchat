"""Owned SAPI process: native audio cancellation cannot hold the motor's GIL."""

import multiprocessing
import threading
import time


def _sapi_worker(connection, settings):
    from .tts import SapiSpeechOutput

    output = None
    try:
        import sounddevice as sd

        output = SapiSpeechOutput(**settings)
        device = sd.query_devices(settings["output_device"])
        if settings["expected_device_name"].casefold() not in device["name"].casefold():
            raise RuntimeError("SAPI output device identity changed")
        connection.send(("ready",))
        generation = 0
        while True:
            if connection.poll(0.05):
                message = connection.recv()
                if message[0] == "close":
                    break
                _, generation, text = message
                if text is None:
                    output.stop()
                else:
                    output.speak(text)
            connection.send(
                (
                    "status",
                    generation,
                    output.speaking,
                    str(output.error)[:200] if output.error else None,
                )
            )
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("failed", str(exc)[:200]))
        except (OSError, EOFError):
            pass
    finally:
        if output:
            output.stop()
            if output._thread:
                output._thread.join(timeout=1.0)
        connection.close()


class ProcessSapiSpeechOutput:
    def __init__(self, output_device, *, expected_device_name, voice="Microsoft Haruka Desktop"):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._guard = threading.Lock()
        self._closing = threading.Event()
        self._generation = 0
        self._pending = None
        self._speaking = False
        self._error = None
        self.diagnostics = {"backend": "sapi_process", "isolated": True}
        self._process = context.Process(
            target=_sapi_worker,
            args=(
                child,
                dict(
                    output_device=output_device,
                    expected_device_name=expected_device_name,
                    voice=voice,
                ),
            ),
            name="myumiq-sapi",
            daemon=True,
        )
        try:
            self._process.start()
            child.close()
            if not parent.poll(8.0):
                raise TimeoutError("SAPI process startup deadline")
            message = parent.recv()
            if message != ("ready",):
                raise RuntimeError(f"SAPI process failed to start: {message}")
        except Exception:
            child.close()
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            parent.close()
            raise
        self.diagnostics["pid"] = self._process.pid
        self._thread = threading.Thread(target=self._pump, name="myumiq-sapi-ipc", daemon=True)
        self._thread.start()

    @property
    def speaking(self):
        with self._guard:
            return self._speaking and not self._closing.is_set()

    @property
    def error(self):
        return self._error

    def speak(self, text):
        if not text or len(text) > 300:
            raise ValueError("TTS text must contain 1..300 characters")
        if self._closing.is_set() or not self._thread.is_alive():
            raise RuntimeError("SAPI process is unavailable")
        with self._guard:
            self._error = None
            self._submit(text)

    def _submit(self, text):
        self._generation += 1
        self._pending = ("command", self._generation, text)
        self._speaking = text is not None

    def stop(self):
        with self._guard:
            self._submit(None)

    def _pump(self):
        outstanding = None
        last_seen = time.monotonic()
        try:
            while not self._closing.is_set():
                if outstanding is None:
                    with self._guard:
                        command, self._pending = self._pending, None
                    if command:
                        self._connection.send(command)
                        outstanding = command[1]
                if self._connection.poll(0.01):
                    message = self._connection.recv()
                    last_seen = time.monotonic()
                    if message[0] == "failed":
                        raise RuntimeError(message[1])
                    _, generation, speaking, error = message
                    if generation == outstanding:
                        outstanding = None
                    with self._guard:
                        if generation == self._generation:
                            self._speaking = speaking
                            self._error = RuntimeError(error) if error else None
                    self.diagnostics = {
                        "backend": "sapi_process",
                        "isolated": True,
                        "pid": self._process.pid,
                        "last_health_at": time.perf_counter(),
                    }
                if not self._process.is_alive():
                    raise RuntimeError(f"SAPI process exited: {self._process.exitcode}")
                if time.monotonic() - last_seen > 2.0:
                    raise TimeoutError("SAPI process heartbeat expired")
        except (OSError, EOFError, RuntimeError, TimeoutError) as exc:
            if not self._closing.is_set():
                self._error = exc
            self._speaking = False
        finally:
            try:
                if self._process.is_alive():
                    try:
                        self._connection.send(("close",))
                    except (OSError, EOFError):
                        pass
                    self._process.join(timeout=1.5)
                    if self._process.is_alive():
                        self._process.terminate()
                        self._process.join(timeout=1.0)
            finally:
                self._connection.close()

    def close(self):
        self.stop()
        self._closing.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError("SAPI process cleanup has not finished")
