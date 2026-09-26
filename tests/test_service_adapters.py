from types import SimpleNamespace

import pytest

from myumiq_vrchat.conversation import VoiceConfig
from myumiq_vrchat.service_adapters import make_asr, make_detector, make_output, make_vad
from myumiq_vrchat.vision import VisionConfig


def test_each_installed_port_is_selected_by_config_without_loading_local_models(monkeypatch):
    asr = SimpleNamespace(transcribe=lambda audio, rate: "こんにちは")
    vad = SimpleNamespace(accept=lambda audio: [], active=False)
    output = SimpleNamespace(speak=lambda text: None, stop=lambda: None, speaking=False)
    detector = SimpleNamespace(detect=lambda image, timestamp: [])
    adapters = dict(asr=asr, vad=vad, speech_output=output, detector=detector)
    calls = []

    def entries(*, group, name):
        calls.append((group, name))
        return (SimpleNamespace(load=lambda: lambda options: adapters[group.split(".")[-1]]),)

    monkeypatch.setattr("myumiq_vrchat.adapters.entry_points", entries)
    cfg = VoiceConfig.model_validate_json("""{
        "input_device": 0, "input_device_name": "fixture in",
        "output_device": 1, "output_device_name": "fixture out",
        "asr_adapter": {"name": "asr"}, "vad_adapter": {"name": "vad"},
        "output_adapter": {"name": "tts"}}""")
    assert cfg.whisper_model is None and cfg.silero_model is None
    assert make_asr(cfg) is asr
    assert make_vad(cfg) is vad
    assert make_output(cfg) is output
    assert make_detector(VisionConfig(detector_adapter={"name": "vision"})) is detector
    assert calls == [
        ("myumiq_vrchat.asr", "asr"),
        ("myumiq_vrchat.vad", "vad"),
        ("myumiq_vrchat.speech_output", "tts"),
        ("myumiq_vrchat.detector", "vision"),
    ]


def test_incompatible_adapter_fails_instead_of_loading_another_type(monkeypatch):
    closed = []
    monkeypatch.setattr(
        "myumiq_vrchat.adapters.entry_points",
        lambda **_: (
            SimpleNamespace(
                load=lambda: lambda options: SimpleNamespace(close=lambda: closed.append(True))
            ),
        ),
    )
    with pytest.raises(TypeError, match="detector contract"):
        make_detector(VisionConfig(detector_adapter={"name": "wrong"}))
    assert closed == [True]
