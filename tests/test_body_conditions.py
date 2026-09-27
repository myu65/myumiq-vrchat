# ruff: noqa: E402 -- optional whole-body dependencies
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
from test_articulated_body import fixture

from myumiq_vrchat.body import BodyCondition, BodyGoal, WorldObject, WorldState
from myumiq_vrchat.body_conditions import ConditionPreparation, complete_goal, resolve_conditions
from myumiq_vrchat.body_console import Command, submit_body_goal
from myumiq_vrchat.whole_body import vector


def goal(*conditions):
    return BodyGoal(conditions=conditions, duration_s=10)


def test_conditions_require_selected_coordinates_and_do_not_create_legacy_tasks():
    with pytest.raises(ValueError, match="empty"):
        BodyCondition(part="head")
    with pytest.raises(ValueError, match="bounded"):
        BodyCondition(part="head", position=(None, None, None))
    with pytest.raises(ValueError, match="target frame"):
        BodyCondition(part="head", frame="target", position=(0.0, 0.0, 0.0))
    c = BodyCondition(part="head", frame="current", position=(None, None, -0.1))
    assert not goal(c).tasks
    with pytest.raises(ValueError, match="combine"):
        goal(c, c)
    command = Command(session="test", issued=1.0, kind="body_goal", body_goal=goal(c))
    assert command.body_goal.conditions == (c,)
    assert not submit_body_goal(None, command.body_goal, 1.0)["accepted"]


def test_unspecified_components_are_free_and_relative_goal_binds_once():
    rig, states = fixture()
    pose = rig.forward(states[0])
    g = goal(BodyCondition(part="head", frame="current", position=(None, None, -0.1)))
    resolved = resolve_conditions(g, pose, WorldState(), 1.0)
    assert resolved.position_mask.sum() == 1 and not resolved.orientation_mask.any()
    assert resolved.desired[0, 2] == pytest.approx(pose.head.position[2] - 0.1)
    points = vector(pose)
    points[0, 0] += 0.2
    points[0, 2] -= 0.1
    from myumiq_vrchat.whole_body import target_from_vector

    assert resolved.measure(target_from_vector(points))["success"]  # free X is not a zero target
    assert not resolved.measure(pose)["success"]


@pytest.mark.parametrize(
    "source,stamp", [("vision", 1.0), ("manual", None), ("manual", 0.0), ("manual", 2.0)]
)
def test_unmeasured_stale_and_future_target_positions_are_rejected(source, stamp):
    rig, states = fixture()
    g = goal(
        BodyCondition(part="left_hand", frame="target", target="item", position=(0.0, 0.0, 0.0))
    )
    world = WorldState(
        objects=(
            WorldObject(name="item", position=(0.0, 0.2, 1.0), source=source, last_seen=stamp),
        )
    )
    with pytest.raises(ValueError, match="fresh calibrated"):
        resolve_conditions(g, rig.forward(states[0]), world, 1.0)


def test_target_motion_and_loss_invalidate_compiled_goal():
    rig, states = fixture()
    g = goal(
        BodyCondition(part="left_hand", frame="target", target="item", position=(0.0, 0.0, 0.0))
    )
    item = WorldObject(name="item", position=(0.0, 0.2, 1.0), source="fixture", last_seen=1.0)
    r = resolve_conditions(g, rig.forward(states[0]), WorldState(objects=(item,)), 1.0)
    r.validate_targets(WorldState(objects=(item.model_copy(update={"last_seen": 1.2}),)), 1.2)
    with pytest.raises(ValueError, match="changed"):
        r.validate_targets(
            WorldState(objects=(item.model_copy(update={"position": (0.1, 0.2, 1.0)}),)), 1.2
        )
    with pytest.raises(ValueError, match="missing"):
        r.validate_targets(WorldState(), 1.2)


@pytest.mark.parametrize("dx", [0.06, 0.13, -0.08])
def test_unregistered_goal_values_complete_through_one_whole_body_solver(dx):
    torch.set_num_threads(1)
    rig, states = fixture()
    pose = rig.forward(states[1])
    g = goal(
        BodyCondition(part="left_hand", frame="current", position=(dx, 0.0, 0.0)),
        BodyCondition(part="right_foot", frame="current", position=(0.0, 0.0, 0.0)),
        BodyCondition(part="left_foot", frame="current", position=(0.0, 0.0, 0.0)),
    )
    r = resolve_conditions(g, pose, WorldState(), 1.0)
    result, report = complete_goal(rig, pose, r, reference_floor=0.0, initial=states[1])
    assert report["success"] and r.measure(result)["success"]
    assert vector(result)[:, 2].min() >= 0
    # Solving a hand endpoint still preserves the connected forearm length.
    p = vector(result)
    assert np.linalg.norm(p[3, :3] - p[5, :3]) == pytest.approx(0.25)


def test_impossible_goal_and_cancellation_never_return_partial_poses():
    rig, states = fixture()
    pose = rig.forward(states[0])
    g = goal(BodyCondition(part="left_foot", position=(None, None, -0.1), position_tolerance=0.005))
    r = resolve_conditions(g, pose, WorldState(), 1.0)
    with pytest.raises((ValueError, RuntimeError), match="accepted|timed out"):
        complete_goal(rig, pose, r, reference_floor=0.0, initial=states[0])
    with pytest.raises(RuntimeError, match="cancelled"):
        complete_goal(rig, pose, r, reference_floor=0.0, initial=states[0], cancelled=lambda: True)


def test_cancelled_preparation_retains_single_slot_and_never_adopts_old_result():
    rig, states = fixture()
    pose = rig.forward(states[0])
    g = goal(BodyCondition(part="head", frame="current", position=(0.0, 0.0, 0.0)))
    jobs = []

    def submit(fn):
        f = Future()
        jobs.append((f, fn))
        return f

    controller = SimpleNamespace(rig=rig, state=states[0], reference_floor=0.0, submit=submit)
    worker = ConditionPreparation()
    assert worker.poll(controller, pose, g, WorldState(), 1.0) is None
    worker.reset()
    assert worker.poll(controller, pose, g, WorldState(), 1.1) is None
    assert len(jobs) == 1 and worker.retiring()
    jobs[0][0].set_result((pose, {"old": True}))
    assert worker.poll(controller, pose, g, WorldState(), 1.2) is None
    assert len(jobs) == 2 and worker.report is None
