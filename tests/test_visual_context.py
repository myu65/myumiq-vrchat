from types import SimpleNamespace

from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.visual_context import visual_context


def test_coherent_snapshot_excludes_stale_low_confidence_tracks():
    def obj(name, confidence, last_seen):
        return WorldObject(
            name=name,
            source="vision",
            kind="player",
            position=(1.0, 0.0, 1.0),
            confidence=confidence,
            last_seen=last_seen,
        )

    world = WorldState(
        timestamp=10.0,
        objects=(obj("good", 0.9, 10.0), obj("old", 0.9, 3.0), obj("weak", 0.1, 10.0)),
    )
    vision = SimpleNamespace(decision_snapshot=lambda: (world, "jpeg", 10.0))
    context = visual_context(vision, WorldState(), 10.1, use_image=True)
    assert [o.name for o in context.world.objects] == ["good"]
    assert context.fresh_image(10.1) == "jpeg"
    assert context.fresh_image(14.0) is None
    assert visual_context(vision, WorldState(), 14.0, use_image=True).image_base64 is None


def test_empty_scene_still_reaches_vision_model():
    world = WorldState(timestamp=5.0)
    vision = SimpleNamespace(decision_snapshot=lambda: (world, "empty-room", 5.0))
    context = visual_context(vision, world, 5.1, use_image=True)
    assert context.fresh_image(5.1) and not context.world.objects


def test_ambiguous_crossing_does_not_silently_switch_identities():
    from myumiq_vrchat.vision import YoloOnnxDetector

    detector = YoloOnnxDetector.__new__(YoloOnnxDetector)
    detector._centres = {"a": (100, 100, 0), "b": (140, 100, 0)}
    detector._next_id = 3
    assert detector._match(0, 102, 100, set()) == "a"
    assert detector._match(0, 120, 100, set()) == "3"
