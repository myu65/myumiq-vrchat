import hashlib
from concurrent.futures import Future
from threading import Event

import pytest
from test_purposes import Services, runner

from myumiq_vrchat.capability_learning import LearningSettings
from myumiq_vrchat.purpose_runtime import PurposeRunner
from myumiq_vrchat.replay_learning_job import (
    ReplayRefinementSettings,
    make_replay_task,
    train_replay_task,
)


def config(root):
    return ReplayRefinementSettings(
        prior=root / "prior",
        train_replay=root / "train.jsonl",
        validation_replay=root / "validation.jsonl",
        reference_corpus=root / "licensed",
    )


def test_optional_penalty_keeps_existing_job_identity_but_changed_objective_is_new(tmp_path):
    settings = config(tmp_path)
    legacy = hashlib.sha256(
        settings.model_dump_json(
            exclude={"floor_weight", "reference_rehearsal", "backtracking_steps"}
        ).encode()
    ).hexdigest()
    assert make_replay_task(settings)["configuration_id"] == legacy
    adjusted = settings.model_copy(update={"floor_weight": 150.0})
    assert make_replay_task(adjusted)["configuration_id"] != legacy
    bounded = settings.model_copy(update={"backtracking_steps": 2})
    assert make_replay_task(bounded)["configuration_id"] != legacy


@pytest.mark.parametrize("eligible", [True, False])
def test_executive_automatically_queues_one_candidate_without_installing_it(
    tmp_path, monkeypatch, eligible
):
    import myumiq_vrchat.purpose_runtime as module

    owner, executive = runner(tmp_path)
    settings = executive.settings.model_copy(
        update={"learning": LearningSettings(replay_refinement=config(tmp_path))}
    )
    executive.settings = settings
    owner.config = owner.config.model_copy(update={"purpose": settings})
    jobs, pending = [], Future()
    monkeypatch.setattr(module, "background", lambda fn: jobs.append(fn) or pending)
    initial_choice = owner.choice
    executive._learning()
    assert len(jobs) == 1 and len(executive.tasks) == 1
    assert executive.tasks[0]["status"] == "running"
    for _ in range(3):
        executive._learning()
    assert len(jobs) == 1
    pending.set_result((None, dict(eligible_for_controlled_trial=eligible, scope="model_only")))
    executive._learning()
    assert executive.tasks[0]["status"] == ("candidate_ready" if eligible else "candidate_rejected")
    assert owner.choice == initial_choice
    assert "ARTICULATED_REFINEMENT" not in executive.registry.items
    executive._learning()
    assert len(jobs) == 1
    executive.close()
    restored = PurposeRunner(owner, Services())
    restored._learning()
    assert len(jobs) == 1
    assert restored.tasks[0]["status"] == executive.tasks[0]["status"]
    restored.close()


def test_cancel_only_terminates_owned_training_process(tmp_path, monkeypatch):
    import myumiq_vrchat.replay_learning_job as module

    events = []

    class Child:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def kill(self):
            events.append("killed_owned_child")

        def communicate(self):
            events.append("reaped_owned_child")
            return "", ""

    def start(command, **options):
        assert command[1:3] == ["-m", "myumiq_vrchat.experience_training"]
        assert options["env"]["OMP_NUM_THREADS"] == "1"
        assert command[command.index("--backtracking-steps") + 1] == "2"
        assert not options.get("shell")
        return Child()

    monkeypatch.setattr(module.subprocess, "Popen", start)
    cancel = Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        train_replay_task(
            config(tmp_path).model_copy(update={"backtracking_steps": 2}),
            tmp_path / "output",
            cancel,
        )
    assert events == ["killed_owned_child", "reaped_owned_child"]
