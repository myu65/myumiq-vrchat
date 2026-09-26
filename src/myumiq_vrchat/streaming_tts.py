"""Cancellable PCM streaming from a machine-local model server."""

import base64
import json
import threading
import time
from collections import deque
from urllib.parse import urlsplit


class _LeadingSilence:
    """Optional bounded startup trim; never alters samples after speech has begun."""

    def __init__(self, rate, maximum_s):
        import numpy as np

        if not 0 <= maximum_s <= 0.75:
            raise ValueError("leading silence trim must be 0..0.75 seconds")
        self.rate, self.limit = rate, int(rate * maximum_s)
        self.frame, self.preroll = max(1, round(rate * 0.005)), round(rate * 0.06)
        self.pending = np.empty(0, dtype=np.float32)
        self.started = not self.limit
        self.trimmed = 0

    def accept(self, audio, *, finished=False):
        import numpy as np

        if self.started:
            return audio
        self.pending = np.concatenate((self.pending, audio))
        count = len(self.pending) // self.frame
        energy = np.sqrt(
            np.mean(self.pending[: count * self.frame].reshape(count, self.frame) ** 2, axis=1)
        )
        audible = np.flatnonzero(energy > 0.001)
        if len(audible):
            cut = max(0, int(audible[0]) * self.frame - self.preroll)
        elif len(self.pending) >= self.limit + self.preroll:
            cut = self.limit
        elif finished:
            cut = max(0, len(self.pending) - self.preroll)
        else:
            return np.empty(0, dtype=np.float32)
        self.trimmed = min(self.limit, cut)
        result = self.pending[self.trimmed :]
        self.pending = np.empty(0, dtype=np.float32)
        self.started = True
        return result


class _PlaybackBuffer:
    """PCM order and rebuffering; callers serialize producer/callback access."""

    def __init__(self, rate, started):
        self.buffers = deque()
        self.queued = 0
        self.playing = False
        self.prebuffer = rate // 2
        self.started = started
        self.diagnostics = {"first_audio_s": None, "underflows": 0, "device_underflows": 0}

    def append(self, audio):
        self.buffers.append(audio)
        self.queued += len(audio)

    def read(self, outdata, finished):
        outdata.fill(0)
        if not self.playing and self.queued and (self.queued >= self.prebuffer or finished):
            self.playing = True
            if self.diagnostics["first_audio_s"] is None:
                self.diagnostics["first_audio_s"] = time.perf_counter() - self.started
        if not self.playing:
            return finished and not self.buffers
        frames, offset = len(outdata), 0
        while self.buffers and offset < frames:
            data = self.buffers[0]
            n = min(len(data), frames - offset)
            outdata[offset : offset + n, 0] = data[:n]
            self.queued -= n
            offset += n
            if n == len(data):
                self.buffers.popleft()
            else:
                self.buffers[0] = data[n:]
        if offset < frames and not finished:
            self.diagnostics["underflows"] += 1
            self.playing = False
        return finished and not self.buffers


class StreamingSpeechOutput:
    def __init__(self, endpoint, device, expected_device_name, *, leading_silence_max_s=0.0):
        url = urlsplit(endpoint)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("TTS endpoint must be local HTTP")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("invalid TTS endpoint")
        self.endpoint, self.device, self.name = endpoint.rstrip("/"), device, expected_device_name
        if not 0 <= leading_silence_max_s <= 0.75:
            raise ValueError("invalid leading silence trim")
        self.leading_silence_max_s = leading_silence_max_s
        self._cancel = threading.Event()
        self._thread = None
        self._error = None
        self.diagnostics = {}

    @property
    def speaking(self):
        return self._thread is not None and self._thread.is_alive() and not self._cancel.is_set()

    @property
    def error(self):
        return self._error

    def stop(self):
        # Playback checks the flag at every 10ms block; never join inference here.
        self._cancel.set()

    def speak(self, text):
        if not text or len(text) > 300:
            raise ValueError("invalid TTS text")
        self.stop()
        cancel = self._cancel = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._work, args=(text, cancel), daemon=True)
        self._thread.start()

    def _work(self, text, cancel):
        import ctypes

        import httpx
        import numpy as np
        import sounddevice as sd

        com = ctypes.WinDLL("ole32")
        result = com.CoInitializeEx(None, 0)
        initialized = result in (0, 1)
        started = time.perf_counter()
        try:
            if result not in (0, 1, -2147417850):
                raise OSError(f"COM initialization failed: {result}")
            device = sd.query_devices(self.device)
            if self.name.casefold() not in device["name"].casefold():
                raise ValueError("TTS output device changed")
            rate = int(device["default_samplerate"])
            buffer = _PlaybackBuffer(rate, started)
            onset = _LeadingSilence(rate, self.leading_silence_max_s)
            self.diagnostics = buffer.diagnostics
            guard = threading.Lock()
            finished = threading.Event()
            drained = threading.Event()

            def callback(outdata, frames, timing, status):
                outdata.fill(0)
                if cancel.is_set():
                    drained.set()
                    raise sd.CallbackAbort
                with guard:
                    if status.output_underflow:
                        self.diagnostics["device_underflows"] += 1
                    if buffer.read(outdata, finished.is_set()):
                        drained.set()
                        raise sd.CallbackStop

            with httpx.stream(
                "POST", self.endpoint + "/speech", json={"text": text}, timeout=60, trust_env=False
            ) as response:
                response.raise_for_status()
                with sd.OutputStream(
                    device=self.device,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=0,
                    latency=0.05,
                    callback=callback,
                ) as output:
                    for line in response.iter_lines():
                        if cancel.is_set():
                            output.abort()
                            return
                        packet = json.loads(line)
                        if "error" in packet:
                            raise RuntimeError(packet["error"])
                        if "pcm" not in packet:
                            continue
                        audio = np.frombuffer(base64.b64decode(packet["pcm"]), dtype="<f4")
                        if not audio.size:
                            continue
                        if rate != 24000:
                            audio = np.interp(
                                np.arange(round(len(audio) * rate / 24000)) * 24000 / rate,
                                np.arange(len(audio)),
                                audio,
                            ).astype("float32")
                        audio = onset.accept(audio)
                        self.diagnostics["leading_silence_trimmed_s"] = onset.trimmed / rate
                        if not audio.size:
                            continue
                        while buffer.queued > rate * 10 and not cancel.wait(0.01):
                            pass
                        if cancel.is_set():
                            output.abort()
                            return
                        with guard:
                            buffer.append(audio)
                    tail = onset.accept(np.empty(0, dtype=np.float32), finished=True)
                    if tail.size:
                        with guard:
                            buffer.append(tail)
                    self.diagnostics["leading_silence_trimmed_s"] = onset.trimmed / rate
                    finished.set()
                    while not drained.wait(0.01):
                        if cancel.is_set():
                            output.abort()
                            return
        except Exception as exc:
            if not cancel.is_set():
                self._error = exc
        finally:
            if initialized:
                com.CoUninitialize()
