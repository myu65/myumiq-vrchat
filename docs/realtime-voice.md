# Realtime voice integration

Model setup is described in [role profiles](model-adapters.md). Speech generation,
body judgment and planning each select their own local/OpenRouter/extension adapter.
On Windows, the SoundCard loopback captures all native channels and mixes to mono
in software before 16 kHz resampling. It must not request a single capture channel:
SoundCard documents a [WASAPI single-channel corruption issue](https://github.com/bastibe/SoundCard#known-issues).
The device buffer is larger than each read block. Test the real capture path with
known spoken audio; a direct audio-file ASR test does not exercise this boundary.

Input health includes `last_block_peak`, `last_block_rms`, `capture_gap_s` and
`capture_queue_wait_s` (plus the maximum wait). `peak` remains the historical
maximum for compatibility. The input-signal indicator uses the current block;
nonzero background audio is signal, not proof that a person's speech arrived.
These scalar measurements retain no raw audio. When investigating a missed short
input, compare actual captured audio/VAD events before blaming ASR or changing
the detection threshold. Also check the injecting player's underrun report in
synthetic device tests; a file submitted for playback is not captured-input proof.

`VoiceConfig.partial_transcripts` defaults to false for chunk-based ASR. VAD onset
and active events still interrupt output and renew listening immediately. Optional
provisional transcription can be enabled on a runtime with spare capacity; it shares
the ASR worker lock and may delay final recognition. Voice health reports final
audio length, time waiting for that lock and inference duration separately.

Finalized input has a separate lifetime from reply cancellation. Completed VAD
segments are recognized in FIFO order even when another segment starts. Configure
`asr_queue_size` (default 8 waiting segments) and `asr_queue_audio_s` (default 60
seconds including in-flight audio). Overflow rejects the new segment explicitly,
increments `asr.rejected_inputs` and records an `audio_input_overflow` event; it
does not overwrite older accepted audio. This is a bounded memory queue, not a
crash-persistent audio spool. Shutdown still closes the session and discards its
unfinished work. Recognition failures are input-scoped events in purpose mode so
that the following captured segment can still be recognized.

Final events carry session/sequence/utterance IDs and audio intervals derived from
the VAD boundary timestamp and sample count. These intervals inherit capture/VAD
timestamp accuracy; they are not sample-accurate acoustic onset measurements.
The inbox retains accepted final events against provisional traffic. If all slots
contain retained events, the pipeline keeps its completed result until delivery
succeeds. The executive records delayed older inputs but only requests a reply
for the current input, checking the capture revision again before submission.
Later empty/noise segments do not automatically replay an earlier answer.
Input retention itself does not make ASR faster or supply native streaming.
Recognition health also reports `pending_input_sequence`, `pending_age_s`,
`waiting_inputs` and `buffered_audio_s`, including while a request is unfinished.

See [the runtime redesign](runtime-redesign.md) for the remaining shared playback
ledger, persistent body goals and model/runtime evaluation stages.

The shared purpose runtime uses `speech_started` before transcription to interrupt
TTS, invalidate old replies, and mark listening attention. Body goals and exploration
leases survive ordinary speech. Explicit local stop requests and existing Home,
tracking and output-expiry gates remain separate.
`speech_active` renews listening. Terminal ASR failure/no-speech/overflow releases
attention for that input only, preserving a newer speaker's listening state.
`partial_transcript` is provisional
working state; only final `utterance` enters episodic/social memory and the shared
memory used independently by the speech and body-decision models. No model inference runs in the
motor loop. Partial text never directly actuates a joint or grants a capability.

Dialogue defaults to `await_body_assessment=false` and `stream_sentences=true`.
Its structured JSON reply is read from a cancellable SSE request on the existing
role worker. A leading `reply` field can publish complete Japanese sentences before
topic/memory fields finish. Other JSON fields are never streamed to audio. A
different field order or an extension without streaming waits for the final
validated reply; it does not switch provider/model. Final schema or transport
failure discards pending text and records only the already submitted prefix.
This cannot retract a sentence already submitted before a later JSON error.

Only the executive submits sentences, through the existing input-revision check
and audio owner. It waits for that owner's `speaking` flag to clear between
sentences. Barge-in cancels pending text as well as playback and HTTP I/O, including
a request waiting for headers. Cancellation closes the local connection; it does
not prove the remote provider stopped compute or billing. Extensions may implement
nonblocking `cancel()` and `request_stream(..., on_text)`; uncooperative extensions
are bounded to two outstanding jobs and further replies coalesce until a slot
clears. Confirmed user utterances are still all retained independently.

`dialogue_sentence_submitted`, `dialogue_partial_reply` and `dialogue_reply` expose
which text was submitted, discarded or expired. These are submission records,
not acoustic playback/remote receipt evidence. Streaming still uses the configured
SAPI/neural TTS adapter; no voice model or machine profile is changed automatically.

ASR selects a built-in local audio-chat service (including Qwen3-ASR),
faster-whisper on an explicit CPU/GPU, or an installed extension. These built-ins
recognize completed segments; neither is **native Qwen3-ASR streaming**.
Chunks retain at most 8 seconds; VAD segments are bounded to
30 seconds. One partial inference is outstanding at a time; final recognition
shares its inference lock. Generation checks discard obsolete partials after onset.

## Configuring ASR and readiness

For Qwen3-ASR, run a compatible audio-enabled llama.cpp server with both the
recognizer GGUF and its matching audio projector. See the upstream
[Qwen3-ASR runtime](https://github.com/QwenLM/Qwen3-ASR),
[GGUF conversion](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF), and
[llama.cpp multimodal support](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
Bind the server to loopback, choose the GPU explicitly in the machine's startup
profile, and use the same model alias in `VoiceConfig`:

Qwen3-ASR 0.6B and 1.7B use the same adapter; each requires its matching projector.
Choose the size using recorded input from the actual capture/VAD route, including
background audio and short requests. Direct clean-file accuracy alone is insufficient.
Set `qwen3_language: "Japanese"` for a prefill-compatible runtime when Japanese is
the intended input language; omit it for automatic language detection. See
[language setup](voice-setup.md) for the prefill contract and bounded context option.

Example for the smaller model:

```json
{
  "asr_adapter": {
    "name": "local_audio_chat",
    "options": {
      "base_url": "http://127.0.0.1:18531/v1",
      "model": "qwen3-asr-0.6b",
      "text_format": "qwen3_asr",
      "timeout_s": 3.0,
      "warmup_timeout_s": 60.0,
      "max_tokens": 512
    }
  },
  "partial_transcripts": false
}
```

Merge this fragment with the input/output device and VAD configuration. Audio is
sent as in-memory mono 16 kHz PCM WAV, with no conversation history or prompt cache.
The Qwen transcript boundary is parsed explicitly; a truncated generation is an
ASR error, not a final input. Requests enforce audio length, response size and
HTTP read/total response checks. An inference timeout is visible in the input
record; it does not silently switch to a different model. Use `text_format: text`
only for an audio-chat service that returns plain transcripts. Nonlocal/custom
services require an explicitly installed adapter rather than changing this local URL.

An alternative configured inference device for faster-whisper is:

```json
{
  "asr_adapter": {
    "name": "faster_whisper",
    "options": {
      "model": "/your/local/faster-whisper-small",
      "device": "cuda",
      "device_index": 0,
      "compute_type": "float16",
      "cpu_threads": 2,
      "language": "ja"
    }
  }
}
```

The device index is relative to that process's visible GPUs. Install the runtime's
matching CUDA/cuDNN dependencies before using it; failures remain visible. CPU
profiles can choose `device: cpu` and `compute_type: int8`. The older
`whisper_model` field remains a CPU-compatible configuration path. Audio now goes
directly to the recognizer as float32 PCM instead of through a temporary WAV file.

`Services` constructs and warms voice/vision outside the executive poll loop.
Each built-in ASR performs a silent warmup before opening live capture. Health
distinguishes `starting`, `running`, and retryable `pending`, with startup duration.
The first cold decode may differ substantially from steady-state performance;
do not spend the user's first utterance on model readiness. Disabled/cancelled
startup cannot inject events into the next voice session or stack replacement
initializers. This is initialization isolation, not a process watchdog for an
arbitrarily hung third-party native library.

`VoiceConfig.streaming_tts_endpoint` enables `StreamingSpeechOutput`. Without it,
the explicitly configured SAPI output remains available. SAPI uses the selected
installed Windows voice, with one cancellation token and output stream per reply.
The configured SAPI adapter owns a separate process for native audio operations.
The playback worker alone closes that stream. Stopping posts a command without
waiting for the audio process. A heartbeat/death check reports output failure;
native cancellation cannot hold the body process's interpreter lock.
Stopping does not join that worker,
and errors from cancelled replies cannot replace the current reply's status.
This is an explicit profile choice; a model-server failure does not switch voices.
The endpoint must be
local HTTP and returns newline JSON records with base64 little-endian float32
mono PCM at 24kHz, or an `error` field. The client checks output device identity,
resamples to its native rate, buffers the first 0.5 seconds, and uses continuous
PortAudio callback playback. Cancellation is checked by the callback; queued
chunks are discarded. Underflows and time to playback are exposed in voice health.
Stopping does not join the synthesis worker. OS playback buffering can add latency;
the generation cutoff is not proof of acoustic silence at the remote client.

The current machine-local backend uses Qwen3-TTS 0.6B CustomVoice, Talker and
Predictor Q4_K_M, a separate ONNX waveform decoder, and preset Ono_Anna/Japanese.
Its model conversion scripts, binaries and local patches live outside this repo.
The model service must be loaded and warmed before enabling live voice. It returns
503 when another synthesis is active; failures remain visible rather than silently
switching voices. An already-running generation can take until its next chunk to
notice client disconnect. Service lifecycle is currently separate from body runtime.

## Live verification

1. Follow [voice setup](voice-setup.md), including VRChat microphone mode and mute.
2. Warm the model without sending its audio to the microphone.
3. Start the shared body/purpose runtime; confirm voice health and device feedback.
4. Bind a known test speaker explicitly. Visual proximity does not identify them.
5. Verify reply heard remotely, memory recall, and simultaneous body feedback.
6. Speak during a long reply; verify onset, playback interruption and listening,
   then verify only the new turn's response is delivered.
7. Stop the body session and verify restoration; stop the separately owned model
   service when no longer needed.

Useful checks include first PCM latency, ASR-final-to-reply time, uninterrupted body
heartbeats, and user-confirmed delivery. A streamed PCM response alone is not a
successful VRChat conversation test.
