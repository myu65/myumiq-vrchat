"""Slow model readiness cannot occupy the executive, nor leak a cancelled session."""

from threading import Event
from types import SimpleNamespace

from myumiq_vrchat.autonomous_services import Services
from myumiq_vrchat.body import WorldState
from myumiq_vrchat.events import EventInbox


def test_voice_warmup_does_not_block_vision_or_executive_ticks():
    entered, release = Event(), Event()
    calls, ticks = [], []
    voice = SimpleNamespace(
        start=lambda: calls.append("start"),
        stop=lambda: calls.append("stop"),
        poll=lambda: None,
        diagnostics={"peak": 0.8, "last_block_peak": 0.0},
        pipeline=SimpleNamespace(
            events=EventInbox(),
            barge_in=SimpleNamespace(interruptions=0),
            asr_diagnostics={},
            output=SimpleNamespace(),
        ),
    )

    def build():
        entered.set()
        assert release.wait(2)
        return voice

    health = {}
    services = Services(SimpleNamespace(vision=True, voice=True, retry_s=1), health, WorldState)
    services.vision = SimpleNamespace(
        world=lambda: ticks.append("vision") or WorldState(), frames=0, stop=lambda: None
    )
    services._voice = build
    try:
        services.poll(0.0)
        assert entered.wait(1)
        for now in range(1, 51):
            services.poll(float(now))
            ticks.append("executive")
        assert health["voice"]["state"] == "starting"
        assert ticks.count("vision") == 51 and ticks.count("executive") == 50
        assert calls == []
        release.set()
        assert services._starting["voice"].done.wait(1)
        services.poll(51.0)
        assert services.voice is voice and services.events is voice.pipeline.events
        assert health["voice"]["state"] == "running"
        assert health["voice"]["input_signal"] == "awaiting_signal"
        assert health["voice"]["startup_s"] >= 0
    finally:
        release.set()
        services.close()
    assert calls == ["start", "stop"]


def test_disable_reenable_does_not_stack_initialization_and_close_cleans_late_result():
    entered, release, stopped = Event(), Event(), Event()
    calls = []

    def build():
        calls.append("build")
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(start=lambda: calls.append("start"), stop=stopped.set)

    services = Services(SimpleNamespace(vision=None, voice=True, retry_s=1), {}, WorldState)
    services._voice = build
    try:
        services.poll(0.0)
        assert entered.wait(1)
        job = services._starting["voice"]
        services.poll(1.0, enabled=False)
        for now in range(2, 10):
            services.poll(float(now))
        assert calls == ["build"]
        services.close()
        services.close()
    finally:
        release.set()
    assert job.done.wait(1) and stopped.is_set()
    assert calls == ["build"] and services.voice is None


def test_failed_startup_retries_only_after_deadline_and_releases_failed_component():
    calls = []
    release = Event()

    def fail():
        assert release.wait(2)
        raise RuntimeError("device unavailable")

    services = Services(SimpleNamespace(vision=None, voice=True, retry_s=5), {}, WorldState)
    services._voice = lambda: SimpleNamespace(start=fail, stop=lambda: calls.append("stop"))
    services.poll(0.0)
    job = services._starting["voice"]
    release.set()
    assert job.done.wait(1)
    services.poll(1.0)
    assert services.health["voice"]["state"] == "pending"
    assert "device unavailable" in services.health["voice"]["error"]
    assert calls == ["stop"] and services.retry["voice"] == 6.0
    services.poll(5.0)
    assert not services._starting
    services.close()
