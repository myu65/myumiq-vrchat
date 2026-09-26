# Autonomous whole-body mode

`body_console run --autonomous-config <outside-repo JSON>` loads an autonomous
executive while initially retaining manual ownership. After private-instance and
FBT setup, `body_console send --session <session> '{"kind":"autonomous"}'`
enables it. `{"kind":"manual"}` or any ordinary body command interrupts autonomy.
`stop` ends the complete body session. Existing duration and watchdog limits apply.

The JSON contains `llm` (the existing LLMConfig), `memory` (persistent JSON path),
optional `vision` and `voice` configuration paths, and `retry_s` (default 30).
Optional `decision` settings enable [candidate reranking](decision-layer.md).
Optional `purpose` settings enable [goal-directed plans, capability memory and
bounded learning tasks](goal-directed-mode.md) above those motor intents.
Purpose mode also enables [shared memory and conversation](integrated-memory.md).
Optional [private Home exploration](exploration.md) adds EXPLORE_HOME with fresh
visual feedback, expiring controller input and a private-world gate.
Configured optional services are attempted, not silently treated as verified.

## Flow

Low-rate local vision and streaming voice feed WorldState and speech events.
Drives, recent outcomes and observed head height enter a schema-constrained local
LLM request. Finite intents select procedural gaze/gesture/posture skills or a
loaded imitation walking policy. The motor loop uses full-body readback and
bounded actuation independently of model inference. PAMIQ whole-body records
include `intent_metadata`; `decisions.jsonl` connects selection, drives and memory.

Supported autonomous intents: WAIT, WAVE, LOOK_AT, REACH, RETURN_TO_REST,
WALK_IN_PLACE, CROUCH, SIT, LIE, STAND. REACH requires a calibrated manual/fixture
target. Vision supplies attention estimates, not contact geometry or identity.
WALK_IN_PLACE does not navigate through the world.

## Fault behavior and status

`status.json` reports autonomous enablement, current intent, drives, and health.
LLM failure selects an explicitly labelled `drive_fallback` intention and retries.
Unavailable vision supplies an empty world; stale targets expire. Desktop vision
requires VRChat in the foreground to avoid sensing another app. Failed audio
capture/synthesis stays pending and retries without stopping the motor loop.
Speech onset invalidates earlier cognition and interrupts speech. In purpose mode,
the shared executive redirects attention while preserving its continuing purpose;
disjoint head/hand overlays can coexist with the base body task. ASR/LLM/TTS
workers never own virtual devices.

Body tracking loss or output/watchdog failure still stops the session. A complete
executive failure is visible and disables autonomy. Ordinary manual commands
cancel autonomous control. Every session remains finite; a fresh session does
not replay old commands.

## Validation boundaries

Automatic calibration inputs are not proof of calibration success. The host
launcher must distinguish pending visual verification from device readback.
Audio processing health does not prove the VRChat microphone is unmuted or that
another player heard speech. Actual partner delivery and barge-in remain separate
validation items. Memory records elapsed actions and measured device observations,
not fabricated social success. Replay capture is not an online RL policy update;
`real_experience_rl` remains pending. General concurrent task arbitration, physical
contact/balance, navigation and robust player identity remain future work.
