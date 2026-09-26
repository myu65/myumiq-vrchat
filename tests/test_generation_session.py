from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from myumiq_vrchat.generation import GenerationResult, GenerationSession
from myumiq_vrchat.generation_config import LLMConfig


def test_role_session_reuses_client_and_closes_once(monkeypatch):
    created, closed = [], []

    def create(config):
        created.append(config.model)
        return SimpleNamespace(
            request=lambda *args: GenerationResult({"reply": "うん"}, 0.01),
            close=lambda: closed.append(config.model),
        )

    monkeypatch.setattr("myumiq_vrchat.generation.make_generator", create)
    session = GenerationSession(LLMConfig(base_url="http://localhost:1", model="speech"))
    assert not created
    for _ in range(2):
        assert session.request([], {}, "speech", 10).value == {"reply": "うん"}
    assert created == ["speech"] and closed == []
    session.close()
    session.close()
    assert closed == ["speech"]
    with pytest.raises(RuntimeError, match="closed"):
        session.request([], {}, "speech", 10)


def test_shutdown_never_waits_for_generation_and_still_validates_deadlines(monkeypatch):
    entered, release, disposed = Event(), Event(), Event()

    def request(*args):
        entered.set()
        assert release.wait(2.0)
        return GenerationResult({}, 2.0)  # Adapter cannot bypass the configured deadline.

    monkeypatch.setattr(
        "myumiq_vrchat.generation.make_generator",
        lambda config: SimpleNamespace(request=request, close=disposed.set),
    )
    session = GenerationSession(
        LLMConfig(base_url="http://localhost:1", model="speech", timeout_s=1.0)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(session.request, [], {}, "speech", 10)
        try:
            assert entered.wait(1.0)
            with pytest.raises(RuntimeError, match="already active"):
                session.request([], {}, "speech", 10)
            session.close()
            assert not disposed.is_set()
        finally:
            release.set()
        with pytest.raises(ValueError, match="deadline"):
            future.result(timeout=2.0)
    assert disposed.is_set()
