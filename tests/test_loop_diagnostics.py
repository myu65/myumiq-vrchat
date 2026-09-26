from myumiq_vrchat.loop_diagnostics import LoopDiagnostics


def test_stall_preserves_phase_and_clock_for_output_event_correlation():
    now = [10.0]
    timing = LoopDiagnostics(clock=lambda: now[0])
    timing.enter("readback")
    now[0] += 0.7
    timing.enter("feedback_validation")
    report = timing.snapshot()
    assert report["slow_phases"] == [
        {"phase": "readback", "started": 10.0, "ended": 10.7, "duration_s": 10.7 - 10.0}
    ]
    assert report["max_duration_s"]["readback"] > 0.5
    assert report["phase"] == "feedback_validation"


def test_history_is_bounded_and_snapshots_do_not_alias_state():
    now = [0.0]
    timing = LoopDiagnostics(clock=lambda: now[0])
    before = timing.snapshot()
    for _ in range(100):
        now[0] += 0.2
        timing.enter("motor")
    assert len(timing.snapshot()["slow_phases"]) == 32
    assert before["max_duration_s"] == {}
    assert before["slow_phases"] == []
