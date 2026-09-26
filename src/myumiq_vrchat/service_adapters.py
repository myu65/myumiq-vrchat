"""Factories for the existing chunk-audio and image detection ports."""

from .adapters import load_adapter


def make_asr(config):
    if config.asr_adapter:
        if config.asr_adapter.name in ("local_audio_chat", "faster_whisper"):
            from .asr_adapters import builtin_asr

            return builtin_asr(config.asr_adapter)
        return load_adapter("asr", config.asr_adapter, methods=("transcribe",))
    from .audio import FasterWhisperASR

    return FasterWhisperASR(config.whisper_model)


def make_vad(config):
    if config.vad_adapter:
        adapter = load_adapter("vad", config.vad_adapter, methods=("accept",))
        if not isinstance(getattr(adapter, "active", None), bool):
            raise TypeError("VAD adapter must expose active: bool")
        return adapter
    from .audio import SileroVAD

    return SileroVAD(
        config.silero_model,
        threshold=config.vad_threshold,
        release_frames=config.vad_release_frames,
    )


def make_output(config):
    if config.output_adapter:
        adapter = load_adapter("speech_output", config.output_adapter, methods=("speak", "stop"))
        if not isinstance(getattr(adapter, "speaking", None), bool):
            raise TypeError("speech output adapter must expose speaking: bool")
        return adapter
    if config.streaming_tts_endpoint:
        from .streaming_tts import StreamingSpeechOutput

        return StreamingSpeechOutput(
            config.streaming_tts_endpoint,
            config.output_device,
            config.output_device_name,
            leading_silence_max_s=config.tts_leading_silence_max_s,
        )
    from .process_sapi import ProcessSapiSpeechOutput

    return ProcessSapiSpeechOutput(
        config.output_device,
        expected_device_name=config.output_device_name,
        voice=config.sapi_voice,
    )


def make_detector(config):
    if config.detector_adapter:
        return load_adapter("detector", config.detector_adapter, methods=("detect",))
    from .vision import TemplateTargetDetector, YoloOnnxDetector

    if config.backend == "template":
        return TemplateTargetDetector(
            config.model, name=config.target_name, confidence=config.confidence
        )
    return YoloOnnxDetector(
        config.model,
        confidence=config.confidence,
        labels=config.labels,
        player_labels=config.player_labels,
        min_player_height_fraction=config.min_player_height_fraction,
    )
