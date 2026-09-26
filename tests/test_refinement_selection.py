import copy
import json

import pytest
import torch

from myumiq_vrchat.experience_refinement import eligible_candidate, select_refinement_step


@pytest.mark.parametrize("regression", [False, True])
def test_smaller_update_must_pass_the_same_floor_and_task_gates(tmp_path, regression):
    actor = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
    prior = copy.deepcopy(actor)
    optimizer = torch.optim.Adam(actor.parameters())
    actor(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    with torch.no_grad():
        actor.weight.fill_(1.0)
    assert optimizer.state
    before = dict(floor_blocked=0, reached=20, mean_endpoint_ratio=0.8)
    reference_before = dict(
        mean_endpoint_error=0.1,
        endpoints_worse_than_hold=0,
        control_contract="static_goal_settling_v1",
        static_goals_reached=12,
    )

    def evaluate():
        step = actor.weight.item()
        return dict(
            after=dict(floor_blocked=0, reached=20, mean_endpoint_ratio=0.8 - 0.1 * step),
            reference_after=dict(
                mean_endpoint_error=0.09,
                endpoints_worse_than_hold=0,
                minimum_foot_height_m=-0.01 if regression or step > 0.25 else 0.01,
                maximum_forearm_shin_length_change_m=0.0,
                control_contract="static_goal_settling_v1",
                static_goals_reached=12,
            ),
        )

    selected, attempts = select_refinement_step(
        actor,
        prior,
        optimizer,
        evaluate,
        lambda row: eligible_candidate(
            before, row["after"], reference_before, row["reference_after"], 0.0
        ),
        3,
    )
    assert prior.weight.item() == 0.0
    if regression:
        assert [r["scale"] for r in attempts] == [1.0, 0.5, 0.25, 0.125]
        assert not selected["eligible_for_controlled_trial"]
        assert actor.weight.item() == 1.0 and optimizer.state
    else:
        assert [r["scale"] for r in attempts] == [1.0, 0.5, 0.25]
        assert selected["eligible_for_controlled_trial"]
        assert actor.weight.item() == 0.25 and not optimizer.state
    torch.save(
        dict(actor=actor.state_dict(), optimizer=optimizer.state_dict()), tmp_path / "selected.pt"
    )
    saved = torch.load(tmp_path / "selected.pt", weights_only=True)
    torch.testing.assert_close(saved["actor"]["weight"], actor.weight)
    assert bool(saved["optimizer"]["state"]) is regression


def test_default_evaluates_only_full_update_and_failure_restores_it():
    actor = torch.nn.Linear(1, 1, bias=False)
    prior = copy.deepcopy(actor)
    with torch.no_grad():
        actor.weight.add_(1.0)
    proposed = actor.weight.detach().clone()
    optimizer = torch.optim.Adam(actor.parameters())
    _, attempts = select_refinement_step(actor, prior, optimizer, lambda: {}, lambda row: False)
    assert len(attempts) == 1

    def broken():
        if not torch.equal(actor.weight, proposed):
            raise RuntimeError("evaluation failed")
        return {}

    with pytest.raises(RuntimeError, match="evaluation failed"):
        select_refinement_step(actor, prior, optimizer, broken, lambda row: False, 2)
    torch.testing.assert_close(actor.weight, proposed, atol=0, rtol=0)


def test_attempt_pose_details_do_not_overflow_the_candidate_ipc_report(tmp_path):
    from myumiq_vrchat.experience_training import save_selection_evidence

    reference = dict(
        static_goals_reached=11, minimum_foot_height_m=0.01, trials=[{"detail": "pose" * 180000}]
    )
    attempts = [
        dict(
            scale=0.5**i,
            eligible_for_controlled_trial=i == 3,
            after={"reached": 20},
            reference_after=reference,
        )
        for i in range(4)
    ]
    summaries = save_selection_evidence(tmp_path, attempts)
    assert len(json.dumps(summaries)) < 2000
    assert [r["scale"] for r in summaries] == [1.0, 0.5, 0.25, 0.125]
    for summary in summaries:
        saved = json.loads((tmp_path / summary["report"]).read_text())
        assert saved["reference_after"] == reference
        assert summary["reference_after"]["static_goals_reached"] == 11
