"""Optional perception/voice adapters with explicit degraded health and retry."""

import time
from threading import Event, Lock, Thread

from .body import WorldState
from .events import EventInbox


class _Startup:
    """Own an initializing component until poll adopts it, or close discards it."""

    def __init__(self, factory):
        self.done, self.cancelled = Event(), Event()
        self.lock = Lock()
        self.component = self.error = None
        self.started = time.perf_counter()
        self.elapsed = None

        def work():
            component = None
            try:
                component = factory()
                if not self.cancelled.is_set():
                    component.start()
                with self.lock:
                    if not self.cancelled.is_set():
                        self.component, component = component, None
            except Exception as exc:
                self.error = exc
            finally:
                if component is not None:
                    try:
                        component.stop()
                    except Exception as exc:
                        self.error = self.error or exc
                self.elapsed = time.perf_counter() - self.started
                self.done.set()

        Thread(target=work, name="myumiq-service-startup", daemon=True).start()

    def take(self):
        with self.lock:
            if self.error:
                raise self.error
            component, self.component = self.component, None
            return component

    def cancel(self):
        with self.lock:
            self.cancelled.set()
            component, self.component = self.component, None
        if component is not None:
            component.stop()


class Services:
    def __init__(self, config, health, world):
        self.config, self.health, self.world = config, health, world
        self.vision = self.voice = None
        self.events = EventInbox()
        self.retry = {"vision": 0.0, "voice": 0.0}
        self.last_speech = -1000.0
        self._starting = {}
        self._startup_s = {}
        self._closed = False

    def _vision(self):
        from .service_adapters import make_detector
        from .vision import LiveVisionLoop, VisionConfig

        cfg = VisionConfig.model_validate_json(self.config.vision.read_text("utf-8-sig"))
        detector = make_detector(cfg)
        return LiveVisionLoop(
            detector,
            title=cfg.window_title,
            hz=cfg.hz,
            capture_backend=cfg.capture_backend,
            fast_hz=cfg.fast_hz,
        )

    def _voice(self):
        from .cognition import decide_conversation
        from .conversation import ConversationPipeline, LiveVoiceLoop, VoiceConfig
        from .service_adapters import make_asr, make_output, make_vad

        cfg = VoiceConfig.model_validate_json(self.config.voice.read_text("utf-8-sig"))
        asr = make_asr(cfg)
        warmup = getattr(asr, "warmup", None)
        if callable(warmup):
            warmup()
        vad = make_vad(cfg)
        output = make_output(cfg)
        # An abandoned startup must not publish into the next voice session.
        pipeline = ConversationPipeline(
            asr,
            output,
            EventInbox(),
            None
            if self.config.purpose
            else lambda text, world: decide_conversation(self.config.llm, text, world),
            partial_transcripts=cfg.partial_transcripts,
            asr_queue_size=cfg.asr_queue_size,
            asr_queue_audio_s=cfg.asr_queue_audio_s,
        )
        return LiveVoiceLoop(
            cfg.input_device,
            cfg.input_device_name,
            vad,
            pipeline,
            self.world,
            loopback_speaker_name=cfg.loopback_speaker_name,
            input_gain=cfg.input_gain,
        )

    def poll(self, now, enabled=True):
        if self._closed:
            return WorldState(), []
        world = WorldState()
        for name in ("vision", "voice"):
            if name == "voice" and not enabled:
                starting = self._starting.get(name)
                if starting:
                    starting.cancel()
                    if starting.done.is_set():
                        self._starting.pop(name)
                if self.voice:
                    self.voice.stop()
                    self.voice = None
                self.events.drain()
                self.health[name] = {"state": "disabled"}
                continue
            if getattr(self.config, name) is None:
                self.health[name] = {"state": "pending_configuration"}
                continue
            try:
                component = getattr(self, name)
                if component is None:
                    if now < self.retry[name]:
                        continue
                    starting = self._starting.get(name)
                    if starting is not None and starting.cancelled.is_set():
                        # Finish releasing the previous model before constructing
                        # a replacement; rapid toggles must not stack workers.
                        if starting.done.is_set():
                            self._starting.pop(name)
                        self.health[name] = {"state": "stopping"}
                        continue
                    if starting is None:
                        starting = self._starting[name] = _Startup(getattr(self, "_" + name))
                    if not starting.done.is_set():
                        self.health[name] = {
                            "state": "starting",
                            "startup_elapsed_s": time.perf_counter() - starting.started,
                        }
                        continue
                    self._starting.pop(name)
                    component = starting.take()
                    setattr(self, name, component)
                    if name == "voice":
                        self.events = component.pipeline.events
                    self._startup_s[name] = starting.elapsed
                if name == "vision":
                    world = component.world()
                    # Never retain stale objects after an occluded or failed camera.
                    if world.timestamp is not None and now - world.timestamp > 3:
                        world = WorldState()
                    self.health[name] = {
                        "state": "running",
                        "frames": component.frames,
                        "startup_s": self._startup_s.get(name),
                        "semantic_accuracy": "pending_verification",
                        "tracking": getattr(component, "tracking_health", {}),
                    }
                    if getattr(component, "capture_pending", None):
                        self.health[name].update(state="pending", error=component.capture_pending)
                else:
                    component.poll()
                    self.health[name] = {
                        "state": "running",
                        **component.diagnostics,
                        "startup_s": self._startup_s.get(name),
                        "barge_in_count": component.pipeline.barge_in.interruptions,
                        "asr": component.pipeline.asr_diagnostics,
                        "tts": getattr(component.pipeline.output, "diagnostics", {}),
                        "input_signal": "observed"
                        if component.diagnostics["last_block_peak"] > 0
                        else "awaiting_signal",
                        "remote_delivery": "pending_verification",
                    }
            except Exception as exc:
                self.health[name] = {"state": "pending", "error": str(exc)[:300]}
                component = getattr(self, name)
                if component:
                    try:
                        component.stop()
                    except Exception as stop_error:
                        self.health[name]["cleanup_error"] = str(stop_error)[:200]
                setattr(self, name, None)
                self.retry[name] = now + self.config.retry_s
        return world, self.events.drain()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for starting in self._starting.values():
            try:
                starting.cancel()
            except Exception as exc:
                self.health["startup_cleanup_error"] = str(exc)[:200]
        self._starting.clear()
        for name in ("voice", "vision"):
            component = getattr(self, name)
            setattr(self, name, None)
            if component:
                try:
                    component.stop()
                except Exception as exc:
                    self.health[name] = {"state": "cleanup_error", "error": str(exc)[:200]}

    def speak(self, text, now):
        if (
            self.voice
            and now - self.last_speech >= 30
            and not self.voice.vad.active
            and not self.voice.pipeline.output.speaking
        ):
            self.voice.pipeline.output.speak(text)
            self.last_speech = now
            self.health["proactive_speech"] = {
                "state": "submitted",
                "delivery": "pending_verification",
            }
            return True
        return False

    def reply(self, text, now, *, input_context=None, wait_until_idle=False):
        if self.voice and not self.voice.vad.active:
            if not self.voice.pipeline.submit_reply(
                text, input_context, wait_until_idle=wait_until_idle
            ):
                return False
            self.last_speech = now
            self.health["conversation_reply"] = {
                "state": "submitted",
                "delivery": "pending_verification",
            }
            return True
        return False
