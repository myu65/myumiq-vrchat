import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from test_articulated_controller import controller

from myumiq_vrchat.posture_validation import main, transition
from myumiq_vrchat.whole_body import target_from_vector, vector


def test_evaluation_distinguishes_holding_from_learning_a_transition(tmp_path, monkeypatch):
    motor, states, _ = controller(monkeypatch, tmp_path)
    start = motor.rig.forward(states[0])
    monkeypatch.setattr("myumiq_vrchat.posture_validation.fit_body", lambda *a: (states[0], {}))
    result = transition(motor.actor, start, start, duration_s=1)
    assert result["accepted"] and result["settled_at_s"] >= 0.15
    shifted = vector(start)
    shifted[:, 0] += 0.4
    result = transition(motor.actor, start, target_from_vector(shifted), duration_s=1)
    assert not result["accepted"] and not result["floor_failure"]
    assert result["maximum_position_error_m"] == pytest.approx(0.4)


def test_evaluation_rejects_low_hands_even_with_feet_above_floor(tmp_path, monkeypatch):
    motor, states, _ = controller(monkeypatch, tmp_path)
    start = motor.rig.forward(states[0])
    monkeypatch.setattr("myumiq_vrchat.posture_validation.fit_body", lambda *a: (states[0], {}))
    shifted = vector(start)
    shifted[3, 2] = -0.01
    assert np.min(shifted[9:, 2]) >= 0
    with pytest.raises(ValueError, match="tracker plane"):
        transition(motor.actor, start, target_from_vector(shifted), duration_s=1)
    result = transition(motor.actor, target_from_vector(shifted), start, duration_s=1)
    assert not result["accepted"] and result["floor_failure"]


@pytest.mark.parametrize("case", ["hold", "unreached", "wrong_rig", "wrong_floor"])
def test_cli_binds_configuration_and_reports_failed_admission(tmp_path, monkeypatch, case):
    motor, states, _ = controller(monkeypatch, tmp_path)
    actor = motor.actor
    start = actor.rig.forward(states[0])
    goal = vector(start)
    if case == "unreached":
        goal[:, 0] += 0.4
    tasks = {
        "actor": str(tmp_path / "actor.pt"),
        "actor_sha256": "b" * 64,
        "rig_sha256": hashlib.sha256(actor.rig.model_dump_json().encode()).hexdigest(),
        "reference_floor": 0.0,
        "goals": {
            "STAND": start.model_dump(mode="json"),
            "CROUCH": target_from_vector(goal).model_dump(mode="json"),
        },
    }
    if case == "wrong_rig":
        tasks["rig_sha256"] = "a" * 64
    if case == "wrong_floor":
        tasks["reference_floor"] = -0.1
    path, out = tmp_path / "tasks.json", tmp_path / "evaluation"
    path.write_text(json.dumps(tasks), "utf-8")
    monkeypatch.setattr("myumiq_vrchat.articulated_actor.ArticulatedActor", lambda path: actor)
    monkeypatch.setattr("myumiq_vrchat.posture_validation.fit_body", lambda *a: (states[0], {}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "posture_validation",
            "--tasks",
            str(path),
            "--out",
            str(out),
            "--goals",
            "STAND",
            "CROUCH",
            "--duration-s",
            "1",
        ],
    )
    if case.startswith("wrong"):
        with pytest.raises(ValueError, match="evaluated configuration"):
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
    assert report["tasks_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(report["cases"]) == 4 and report["avatar_verified"] is False
