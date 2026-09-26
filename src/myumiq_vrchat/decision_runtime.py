"""High-level candidate proposals and optional reranking, outside the motor loop."""

import time
from dataclasses import asdict

from pydantic import Field, model_validator

from .body import Frozen, Number
from .decision import BackendConfig, Candidate, DecisionInput, make_scorer
from .decision_selection import SelectionSettings


class DecisionSettings(Frozen):
    backend: BackendConfig | None = None
    selection: SelectionSettings | None = None
    max_age_s: Number = Field(default=5, gt=0, le=30)
    refresh_s: Number = Field(default=5, ge=0.5, le=30)

    @model_validator(mode="after")
    def isolated(self):
        if (self.backend is None) == (self.selection is None):
            raise ValueError("configure exactly one scorer backend or Selection profile")
        if self.backend and self.backend.backend in ("qwen", "nemotron"):
            raise ValueError("run local models in the isolated decision service; use http here")
        return self


def propose(primary, world, walking):
    from .autonomous_body import Intent

    intents = [
        primary,
        Intent(skill="WAIT"),
        Intent(skill="SIT"),
        Intent(skill="STAND"),
        Intent(skill="WAVE", hand="right"),
        Intent(skill="RETURN_TO_REST"),
    ]
    if walking:
        intents.append(Intent(skill="WALK_IN_PLACE"))
    if world.objects:
        intents.append(Intent(skill="LOOK_AT", target=world.objects[0].name))
    descriptions = {
        "WAIT": "Wait quietly and listen",
        "SIT": "Sit on the floor and rest",
        "STAND": "Stand up",
        "WAVE": "Wave to greet",
        "RETURN_TO_REST": "Return to neutral pose",
        "WALK_IN_PLACE": "Practice learned walking in place",
        "LOOK_AT": "Look at the visible target",
        "REACH": "Reach to the verified nearby target",
        "CROUCH": "Crouch",
        "LIE": "Lie down",
    }
    seen, candidates = set(), []
    for intent in intents:
        identity = (intent.skill, intent.hand, intent.target)
        if identity in seen:
            continue
        try:
            intent.validate_world(world)
        except ValueError:
            continue
        seen.add(identity)
        candidates.append(
            Candidate(
                id=f"action_{len(candidates)}",
                description=descriptions[intent.skill],
                intent=intent.model_dump(mode="json"),
            )
        )
    return tuple(candidates)


def rerank(
    settings, primary, snapshot, drives, recent, body, walking, *, candidates=None, context=None
):
    from .autonomous_body import Intent

    if settings.selection is not None:
        raise ValueError("Selection requires the independent purpose executive, not chat reranking")
    world, image, captured = snapshot
    request = DecisionInput(
        world=world,
        image_base64=image,
        image_mime="image/jpeg",
        captured_at=captured,
        candidates=candidates if candidates is not None else propose(primary, world, walking),
        state={
            "drives": asdict(drives),
            "recent": recent[-5:],
            "body": body,
            "llm_proposal": primary.model_dump(mode="json"),
            "walking_available": walking,
            "goal_context": context,
        },
        instruction="Evaluate the usefulness of this action now from the image, state, drives, "
        "LLM proposal and recent outcomes. Observe without assuming contact or navigation.",
    )
    if not 0 <= time.perf_counter() - captured <= settings.max_age_s:
        raise ValueError("decision snapshot expired before scoring")
    scorer = make_scorer(settings.backend)
    try:
        result = scorer.score(request)
        selected = result.select(request, time.perf_counter(), settings.max_age_s)
        intent = Intent.model_validate(selected.intent)
        intent.validate_world(world)
        # Preserve the LLM's speech only if its original proposal wins.
        report = result.model_dump(mode="json")
        report["selected_id"] = selected.id
        report["candidates"] = [c.model_dump(mode="json") for c in request.candidates]
        return intent, report
    finally:
        if hasattr(scorer, "close"):
            scorer.close()
