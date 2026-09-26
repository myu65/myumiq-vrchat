import pytest

from myumiq_vrchat.audio import AudioFrame, BargeInCoordinator, EnergyVAD
from myumiq_vrchat.body import WorldObject, WorldState
from myumiq_vrchat.cognition import Goal, intent_schema
from myumiq_vrchat.perception import SpatialDetection, TemporalWorldTracker


def test_temporal_world_tracks_velocity_and_expires():
    tracker = TemporalWorldTracker(ttl_s=1, smoothing=0.5)
    world = tracker.update([SpatialDetection("player_1", (1, 0, 1.6), 0.9, "player")], 1.0)
    assert world.locate("player_1") == (1, 0, 1.6)
    world = tracker.update([SpatialDetection("player_1", (2, 0, 1.6), 0.8, "player")], 2.0)
    assert world.locate("player_1") == (1.5, 0.0, 1.6)
    assert world.objects[0].velocity == (0.5, 0.0, 0.0)
    assert tracker.update([], 3.01).objects == ()


def test_speech_onset_barges_in_before_utterance_completion():
    class Output:
        speaking = True
        stopped = False

        def stop(self):
            self.stopped = True
            self.speaking = False

    output = Output()
    coordinator = BargeInCoordinator(output)
    vad = EnergyVAD(threshold=0.1, release_frames=2)
    events = vad.accept(AudioFrame((0.5,) * 160, 16000, 1.0))
    assert [event.kind for event in events] == ["speech_started"]
    assert coordinator.on_event(events[0]) and output.stopped
    assert vad.accept(AudioFrame((0.0,) * 160, 16000, 1.01)) == []
    ended = vad.accept(AudioFrame((0.0,) * 160, 16000, 1.02))
    assert ended[0].kind == "speech_ended" and len(ended[0].audio) == 480


def test_bad_audio_and_detection_rejected():
    with pytest.raises(ValueError):
        EnergyVAD().accept(AudioFrame((), 16000, 0))
    with pytest.raises(ValueError):
        TemporalWorldTracker().update([SpatialDetection("", (0, 0, 0), 1)], 0)


def test_vision_geometry_can_look_but_cannot_reach():
    world = WorldState(
        objects=(
            WorldObject(
                name="person-1",
                position=(2.0, 0.0, 1.6),
                source="vision",
                kind="player",
            ),
        )
    )
    Goal(skill="LOOK_AT", target="person-1", duration_s=1).validate_world(world)
    with pytest.raises(ValueError, match="calibrated"):
        Goal(skill="REACH", hand="right", target="person-1", duration_s=1).validate_world(world)
    reach_variants = [
        variant
        for variant in intent_schema(world)["anyOf"]
        if variant["properties"]["skill"]["const"] == "REACH"
    ]
    assert reach_variants == []
