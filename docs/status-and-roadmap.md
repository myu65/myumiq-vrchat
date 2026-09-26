# Integrated agent status and remaining work

This is an implementation map, not a claim that all end-to-end goals are complete.
The [2026-09-21 runtime redesign](runtime-redesign.md) defines the target boundaries
and migration gates. Input continuity is the first implemented slice; persistent
compound body goals, playback evidence and experience-driven actor promotion are
still migration work.
Machine-specific measurements, logs, model paths and live-session handoff are kept
in the parent workspace's `NEXT_SESSION.md` and `MYUMIQ_RUNBOOK.md`.

The functional goal is rough but working VRChat locomotion, conversation and
gestures, followed by repeated acquisition/refinement of new motion capabilities.
Conversational polish and perfect posture scores are not prerequisites for bounded
live functional trials. Keep posture regressions as diagnostics and retain explicit
floor/actuator bounds. Research and the remaining learning gap are documented in
[the learning approach review](learning-approach-review.md).

## Implemented boundaries

- Purpose-mode dialogue starts independently of body assessment, cancels obsolete
  HTTP streams, and submits sentences through the existing audio owner. Ordinary
  speech preserves navigation; unambiguous stop requests are handled locally.
- The existing output process composes separately leased pose, locomotion and hand
  input at nominal 60 Hz. Its repeated frames cannot renew stale producer evidence.
- Normal navigation uses continuous short leases and a 30 Hz local acceleration
  controller. The same state scales gait reference speed and controller demand;
  per-step readback waiting no longer cuts valid locomotion input. Legacy pulses
  are explicitly `pulse_test` mode. These changes require new live acceptance.
- Perception and audio events feed WorldState and the shared purpose executive.
- Working, episodic, semantic, social and capability memory supply dialogue,
  continuing purposes, planning and optional independent candidate scoring.
- With purpose + decision configured, the speech model only generates replies.
  A separate model continuously chooses body intentions from dialogue history,
  world/body state, drives and outcomes. Motor policies produce actuation.
  Chat-generated body plans remain only in the legacy no-Decision mode.
- Canonical full-body state, goals/constraints and targets are separate. Existing
  VirtualHMD/VMT output supports eleven points and posture skills.
- CC0 glTF/GLB motion import, explicit skeleton retargeting, finite and periodic
  imitation feed the same whole-body actor. Configured lessons can automatically
  train, evaluate, register and restore named gesture/dance/motion capabilities.
- Bounded mapless Home exploration combines learned gait with separate controller
  leases and fresh image observations. Pose feedback alone does not prove travel.
- Optional horizontal body facing rotates the current complete posture through
  that actor; losing a visual target holds the observed posture. Fresh centered
  images and settled body feedback complete the finite goal and retain its evidence.
- Audio and body workers run separately. Speech onset cancels stale replies
  and speech playback while retaining accepted audio for final transcription.
  Final transcripts carry input identity/capture intervals and are retained under
  event backpressure; overload is explicit. Partial ASR currently re-decodes
  bounded windows; this is not native Qwen3-ASR streaming.
- Local streamed Q4 TTS is supported through the HTTP PCM adapter. Quantized
  language components do not imply a quantized waveform decoder.
- PAMIQ records transitions with intent, policy and shared-memory references.
- Optional configured replay refinement uses PAMIQ's training-only model/trainer
  in one bounded background learning job. Confirmed device starts and a separate
  validation session compare the candidate without changing live inference weights.
- Windows Graphics Capture can target VRChat without foreground focus. Native
  capture runs in an owned process so a capture-library stall cannot hold the
  body and speech process. Missing/old frames invalidate perception.
- Optional image-aware selection receives bounded JPEGs encoded by the vision
  worker together with their original world snapshot. Scene/target consistency
  is validated; detector labels and model interpretations remain uncertain.

## Not complete

- 20 Hz actor rollout, short motion horizons, buffered interpolation/blending,
  and fast visual tracking independent of detection/semantic latency. Current
  actor reference advancement still uses confirmed readback, and navigation still
  uses the existing camera freshness gate. CPU/UDP tests do not prove avatar behavior.
- Stable person identity, depth, world localization and contact observation.
- Reliable transitions across arbitrary poses and metric navigation. Learned
  whole-body transitions exist, but supported poses and bounded Home exploration
  do not establish arbitrary world navigation or obstacle avoidance. VRChat pose
  persistence is the actuator contract; gravity balancing is not the main task.
- Open-ended goal generation in the independent decision path. Its current
  executable candidate catalogue is bounded, and reranker judgment can be wrong.
- Automatic execution of arbitrary novel capabilities, handshake, patting or
  autonomous avatar selection. Missing prerequisites become explicit learning tasks.
- Useful real-VR experience-driven policy improvement with validated before/after
  task performance. Simulated training and virtual-device tracking are separate evidence.
- Consistently grounded Japanese dialogue and goal selection; social memory requires
  explicit identity and cannot be inferred from a nearby visual detection.
- Quantified live barge-in latency and long-session robustness, native streaming ASR.
- Automatic microphone mute/device verification; remote reception still needs evidence.
- Consistent model latency across providers under simultaneous VR/model load.
  Model roles, adapters, providers and deadlines are configuration; portable setup
  profiles and role checks do not silently start an older fallback model.

## Acceptance sequence

1. Diagnose device-feedback loss without weakening watchdogs.
2. Verify conversational specificity, ownership of remembered facts and stop requests.
3. Measure actual speech interruption while body control continues.
4. Demonstrate a sustained shared-goal conversation/action session and later memory use.
5. Improve perception and evaluators before promoting more motor capabilities.
6. Close one real-experience learning loop with meaningful reward and a safe comparison.

See [architecture](architecture.md), [full body](full-body.md),
[purpose executive](goal-directed-mode.md), [shared memory](integrated-memory.md),
[decision layer](decision-layer.md), [exploration](exploration.md),
[realtime voice](realtime-voice.md) and [voice setup](voice-setup.md).
