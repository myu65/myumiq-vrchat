from myumiq_vrchat import cli


def test_startup_waits_without_starting_runtime(tmp_path, monkeypatch):
    prepared, signal = tmp_path / "prepared.json", tmp_path / "start"
    entered = []

    def release(delay):
        assert prepared.exists()
        assert not entered
        signal.touch()

    def run(*args, **kwargs):
        assert signal.exists()
        entered.append(True)
        return {"error": None}

    monkeypatch.setattr(cli.time, "sleep", release)
    monkeypatch.setattr(cli, "run_session", run)
    assert (
        cli.main(
            [
                "run",
                "--output",
                str(tmp_path / "run"),
                "--goal",
                '{"skill":"WAIT","duration_s":1}',
                "--prepared-file",
                str(prepared),
                "--start-file",
                str(signal),
            ]
        )
        == 0
    )
    assert entered == [True]


def test_startup_rejects_stale_signal(tmp_path, monkeypatch):
    signal = tmp_path / "start"
    signal.touch()
    monkeypatch.setattr(
        cli,
        "run_session",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("runtime must not start")),
    )
    assert (
        cli.main(
            [
                "run",
                "--output",
                str(tmp_path / "run"),
                "--goal",
                '{"skill":"WAIT","duration_s":1}',
                "--prepared-file",
                str(tmp_path / "ready"),
                "--start-file",
                str(signal),
            ]
        )
        == 1
    )
