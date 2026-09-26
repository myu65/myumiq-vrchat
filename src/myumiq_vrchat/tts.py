"""Cancellable local Windows TTS output to an explicitly selected audio device."""

import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path


class SapiSpeechOutput:
    """Fallback TTS for the vertical slice; output device must be a dedicated route."""

    def __init__(
        self,
        output_device: int,
        *,
        expected_device_name: str,
        voice: str = "Microsoft Haruka Desktop",
    ):
        if output_device < 0 or not expected_device_name.strip() or not voice or len(voice) > 100:
            raise ValueError("invalid TTS output configuration")
        self.output_device = output_device
        self.expected_device_name = expected_device_name.casefold()
        self.voice = voice
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    @property
    def speaking(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._cancel.is_set()

    @property
    def error(self) -> Exception | None:
        return self._error

    def speak(self, text: str) -> None:
        if not text or len(text) > 300:
            raise ValueError("TTS text must contain 1..300 characters")
        self.stop()
        # Cancellation belongs to this utterance, never a reusable shared flag.
        self._cancel = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._work, args=(text, self._cancel), name="myumiq-tts", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._cancel.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        # Only the playback worker touches its stream. A global sd.stop() races
        # sd.play(blocking=True)'s own close, and a join stalls new audio input.

    def _work(self, text: str, cancel: threading.Event) -> None:
        com = None
        try:
            # WASAPI needs COM on the playback worker, including when another
            # thread already owns a capture stream.
            if sys.platform == "win32":
                import ctypes

                ole32 = ctypes.WinDLL("ole32")
                result = ole32.CoInitializeEx(None, 0)
                if result in (0, 1):
                    com = ole32
                elif result != -2147417850:  # RPC_E_CHANGED_MODE: already initialized
                    raise OSError(f"audio COM initialization failed: {result}")
            self._synthesize_and_play(text, cancel)
        except Exception as exc:  # surfaced through error; daemon threads cannot raise to caller
            if not cancel.is_set() and self._cancel is cancel:
                self._error = exc
        finally:
            if com is not None:
                com.CoUninitialize()
            with self._lock:
                if self._cancel is cancel:
                    self._process = None

    def _synthesize_and_play(self, text: str, cancel: threading.Event) -> None:
        if cancel.is_set():
            return
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("install numpy and sounddevice for local TTS output") from exc
        device = sd.query_devices(self.output_device)
        device_name = str(device["name"])
        if self.expected_device_name not in device_name.casefold():
            raise RuntimeError(
                f"refusing TTS route: device {self.output_device} is {device_name!r}, "
                f"expected {self.expected_device_name!r}"
            )
        target_rate = int(device["default_samplerate"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path, wav_path, script_path = (
                root / "text.txt",
                root / "speech.wav",
                root / "speak.ps1",
            )
            text_path.write_text(text, encoding="utf-8")
            script_path.write_text(
                "param($TextPath,$WavePath,$Voice)\n"
                "Add-Type -AssemblyName System.Speech\n"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
                "$s.SelectVoice($Voice)\n"
                "$s.SetOutputToWaveFile($WavePath)\n"
                "$s.Speak([IO.File]::ReadAllText($TextPath,[Text.Encoding]::UTF8))\n"
                "$s.Dispose()\n",
                encoding="utf-8",
            )
            creationflags = (
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            )
            with self._lock:
                if cancel.is_set():
                    return
                process = subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-File",
                        str(script_path),
                        str(text_path),
                        str(wav_path),
                        self.voice,
                    ],
                    creationflags=creationflags,
                )
                self._process = process
            returncode = process.wait()
            if cancel.is_set():
                return
            if returncode:
                raise RuntimeError(f"SAPI synthesis failed with exit code {returncode}")
            with wave.open(str(wav_path), "rb") as wav:
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                samples /= 32768.0
                source_rate = wav.getframerate()
            if source_rate != target_rate:
                count = int(len(samples) * target_rate / source_rate)
                samples = np.interp(
                    np.arange(count) * source_rate / target_rate,
                    np.arange(len(samples)),
                    samples,
                ).astype(np.float32)
            self._play_pcm(samples, target_rate, cancel)

    def _play_pcm(self, samples, rate: int, cancel: threading.Event) -> None:
        import sounddevice as sd

        if cancel.is_set():
            return
        finished = threading.Event()
        position = 0

        def callback(outdata, frames, timing, status):
            nonlocal position
            outdata.fill(0)
            if cancel.is_set():
                raise sd.CallbackAbort
            count = min(frames, len(samples) - position)
            outdata[:count, 0] = samples[position : position + count]
            position += count
            if position >= len(samples):
                raise sd.CallbackStop

        with sd.OutputStream(
            device=self.output_device,
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=0,
            latency=0.05,
            callback=callback,
            finished_callback=finished.set,
        ) as stream:
            while not finished.wait(0.01):
                if cancel.is_set():
                    stream.abort()
                    break
