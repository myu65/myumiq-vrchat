import hashlib
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_articulated_controller import controller

from myumiq_vrchat.condition_validation import main


@pytest.mark.parametrize("case", ["hold", "unreached", "wrong_rig"])
def test_condition_cli_preserves_evidence_and_rejects_unreached_goals(tmp_path, monkeypatch, case):
    motor, states, _ = controller(monkeypatch, tmp_path)
    actor = motor.actor
    start = actor.rig.forward(states[0])
    tasks = {
        "actor": str(tmp_path / "actor.pt"),
        "actor_sha256": "b" * 64,
        "rig_sha256": hashlib.sha256(actor.rig.model_dump_json().encode()).hexdigest(),
        "reference_floor": 0.0,
        "goals": {"STAND": start.model_dump(mode="json")},
    }
    if case == "wrong_rig":
        tasks["rig_sha256"] = "a" * 64
    cases = [
        {
            "id": "head-condition",
            "start": "STAND",
            "split": "heldout",
            "goal": {
                "duration_s": 1.0,
                "conditions": [
                    {
                        "part": "head",
                        "frame": "current",
                        "position": [None, None, 0.04 if case == "unreached" else 0.0],
                        "position_tolerance": 0.01,
                    }
                ],
            },
        }
    ]
    task_path, case_path, out = tmp_path / "tasks.json", tmp_path / "cases.json", tmp_path / "out"
    task_path.write_text(json.dumps(tasks), "utf-8")
    case_path.write_text(json.dumps(cases), "utf-8")
    monkeypatch.setattr("myumiq_vrchat.articulated_actor.ArticulatedActor", lambda path: actor)
    monkeypatch.setattr("myumiq_vrchat.posture_validation.fit_body", lambda *a: (states[0], {}))
    # Isolate condition acceptance from optimization: a completed dense target
    # still cannot prove the actor satisfies the original stricter requirements.
    monkeypatch.setattr(
        "myumiq_vrchat.body_conditions.complete_goal", lambda *a, **kw: (start, {"success": True})
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "condition_validation",
            "--tasks",
            str(task_path),
            "--cases",
            str(case_path),
            "--out",
            str(out),
        ],
    )
    if case == "wrong_rig":
        with pytest.raises(ValueError, match="configured actor, rig or floor"):
            main()
        assert not out.exists()
        return
    if case == "unreached":
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    else:
        main()
    report = json.loads((out / "result.json").read_text("utf-8"))
    assert report["all_accepted"] is (case == "hold")
    assert report["tasks_sha256"] == hashlib.sha256(task_path.read_bytes()).hexdigest()
    assert report["cases_sha256"] == hashlib.sha256(case_path.read_bytes()).hexdigest()
    assert report["reference_floor"] == 0.0 and not report["promoted"]
    result = report["results"][0]
    assert result["completion"]["success"]
    assert result["condition_result"]["success"] is (case == "hold")
