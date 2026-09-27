"""Evaluate unnamed body conditions and the actual actor without device output."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import Field

from .body import BodyGoal, Frozen, Number, WorldState


class ConditionCase(Frozen):
    id: str = Field(min_length=1, max_length=80)
    start: str = Field(min_length=1, max_length=64)
    goal: BodyGoal
    world: WorldState = WorldState()
    observed_at: Number = 0.0
    split: str = Field(pattern=r"^(train|heldout)$")


def evaluate_case(actor, settings, case):
    from .body_conditions import complete_goal, resolve_conditions
    from .posture_validation import transition

    start = settings.goals[case.start]
    resolved = resolve_conditions(case.goal, start, case.world, case.observed_at)
    completed, report = complete_goal(
        actor.rig, start, resolved, reference_floor=settings.reference_floor
    )
    result = transition(
        actor,
        start,
        completed,
        duration_s=case.goal.duration_s,
        floor=settings.reference_floor,
        criterion=resolved.measure,
    )
    return {
        "id": case.id,
        "split": case.split,
        "completion": report,
        "completed_goal": completed.model_dump(mode="json"),
        "actor": result,
        "condition_result": resolved.measure(
            type(completed).model_validate_json(json.dumps(result["endpoint"]))
        ),
        "scope": "synthetic_static_conditions_not_avatar_or_contact",
    }


def main():
    from .articulated_actor import ArticulatedActor
    from .articulated_tasks import ArticulatedTasks
    from .cli import outside_repo

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    task_bytes = outside_repo(args.tasks).read_bytes()
    settings = ArticulatedTasks.model_validate_json(task_bytes)
    raw = args.cases.read_bytes()
    items = json.loads(raw)
    if not isinstance(items, list) or not 1 <= len(items) <= 64:
        raise ValueError("provide one to 64 explicit condition cases")
    cases = [ConditionCase.model_validate_json(json.dumps(c)) for c in items]
    if len({c.id for c in cases}) != len(cases) or any(
        c.start not in settings.goals for c in cases
    ):
        raise ValueError("cases require unique identities and configured starting postures")
    actor = ArticulatedActor(outside_repo(settings.actor))
    if (
        actor.manifest["sha256"] != settings.actor_sha256
        or hashlib.sha256(actor.rig.model_dump_json().encode()).hexdigest() != settings.rig_sha256
        or actor.manifest.get("reference_floor") not in (None, settings.reference_floor)
    ):
        raise ValueError("condition evaluation differs from configured actor, rig or floor")
    output = outside_repo(args.out)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        try:
            row = evaluate_case(actor, settings, case)
        except (ValueError, RuntimeError) as exc:
            row = dict(id=case.id, split=case.split, error=str(exc), actor={"accepted": False})
        results.append(row)
        print(
            json.dumps(
                {
                    "id": case.id,
                    "completion": "error" not in row,
                    "actor_accepted": row["actor"]["accepted"],
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
    report = {
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "actor_sha256": settings.actor_sha256,
        "rig_sha256": settings.rig_sha256,
        "reference_floor": settings.reference_floor,
        "results": results,
        "promoted": False,
        "all_accepted": all(row["actor"]["accepted"] for row in results),
        "scope": "synthetic_static_conditions_not_avatar_or_contact",
    }
    (output / "result.json").write_text(json.dumps(report, indent=2), "utf-8")
    if not report["all_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
