import time
from threading import Event
from types import SimpleNamespace

from myumiq_vrchat.exploration import ExplorationSettings
from myumiq_vrchat.exploration_runtime import ExplorationRuntime


def test_disk_stall_revokes_old_home_permission_without_blocking_executive(tmp_path):
    runner = SimpleNamespace(
        owner=SimpleNamespace(
            root=tmp_path, config=SimpleNamespace(exploration=ExplorationSettings(enabled=True))
        )
    )
    runtime = ExplorationRuntime(runner)
    runtime.gate_observation = (10.0, True)
    entered, release = Event(), Event()
    calls = []

    def slow_check():
        calls.append(1)
        entered.set()
        assert release.wait(2.0)
        return True

    runtime.gate.valid = slow_check
    try:
        assert runtime._gate_allowed(10.5, True)
        assert entered.wait(1.0)
        started = time.perf_counter()
        for _ in range(30):
            assert not runtime._gate_allowed(10.8, True)
        assert time.perf_counter() - started < 0.1 and len(calls) == 1
        future = runtime.gate_pending[0]
        release.set()
        assert future.result(timeout=1.0)
        assert not runtime._gate_allowed(11.5, True)  # Result kept its original timestamp.
        assert not runtime._gate_allowed(11.6, False)
    finally:
        release.set()
