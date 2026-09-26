# Running the LLM-to-body slice

For current independent conversation, decision and planning roles, start with
[portable model setup](model-adapters.md#最短の設定手順) and the
[body console](body-console.md). The `run` examples below are the original small
intention/procedural-motor slice; their neutral return is not the autonomous
whole-body loop's pose-holding behavior.

For device routing, desktop setup, and runtime microphone mute checks, see
[VRChat音声セットアップ](voice-setup.md).

The executable path is local LLM → validated intention → procedural motor policy
→ supervised device output, with PAMIQ observation/action and replay persistence.
Use the mock path before live output. This program does not install drivers,
launch SteamVR/VRChat, switch sessions, alter Windows protections, or select a GPU.

## Install without changing an existing Python environment

Python 3.12+ is required. From the repository root, with `uv` installed:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '../myumiq.local-venv'
uv sync --frozen --python 3.12 --extra dev --extra vr
$miPython = (Resolve-Path ../myumiq.local-venv/Scripts/python.exe).Path
./scripts/test-python.ps1 -Python $miPython
```

The test script allocates a fresh parent-directory temp/cache/result location on
each run, so separate Windows execution users do not share pytest temp ownership.
`vr` installs the OpenVR client binding, not an OpenVR driver. Models, downloaded
executables, credentials, host configuration and all run output belong outside
the repository. The application refuses repository-local run/config paths.

## Fixed intention without an LLM or VR

```powershell
& $miPython -m myumiq_vrchat run --output ../myumiq-runs/wave-001 `
  --goal '{"skill":"WAVE","hand":"right","duration_s":3}' --duration 8
```

The output directory must be new. Supported skills:

| Skill | Required arguments | Behavior |
| --- | --- | --- |
| `WAIT` | `duration_s` | Neutral body target |
| `WAVE` | `duration_s`, `hand` | Bounded left/right hand gesture, fingers open |
| `LOOK_AT` | `duration_s`, `target` | Limited head yaw/pitch toward explicit geometry |
| `REACH` | `duration_s`, `target`, `hand` | Reach inside a small local workspace; no root movement |

Durations are 0.2–10 seconds. After that, the policy moves back toward neutral;
it does not repeat the intention. Session duration includes LLM waiting time.
The default session is 10 seconds at 60 Hz; accepted ranges are 1–3600 seconds
and 20–120 Hz. A motor stall over 100 ms fails explicitly. Output timeouts are
independent of motor cadence. Finger values use curl 0=open / 1=fist.

## Local LLM

Run a compatible local chat-completions server separately. A CPU-only
[official llama.cpp distribution](https://github.com/ggml-org/llama.cpp/releases)
and a [Japanese LFM GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606-GGUF)
can serve the local speech role. Validate the chosen model with its actual schema.
Keep model selection and
hardware allocation in local configuration. Do not change a loaded GPU model or
terminate another workload to make room. Do not expose the server publicly.

For an already running loopback server, create `../myumiq-llm.json`, for example:

```json
{
  "base_url": "http://127.0.0.1:18487/v1",
  "model": "LiquidAI/LFM2.5-1.2B-JP-202606",
  "timeout_s": 30,
  "structured_output": true
}
```

```powershell
& $miPython -m myumiq_vrchat run --output ../myumiq-runs/llm-001 `
  --llm-config ../myumiq-llm.json --prompt '右手を3秒間振ってください。' --duration 15
```

The adapter sends a JSON schema with separate skill/argument combinations and only
known target names when `structured_output=true`. If a server lacks
that feature, explicitly set it to false; local schema validation remains mandatory.
There is no silent retry with a weaker format. Invalid JSON, invented targets,
unsupported fields, HTTP errors, truncation and timeouts stop the session. An
unsupported natural-language request may be classified as `WAIT`, visible in replay.
The LLM can only choose supported intentions; it cannot send poses, buttons,
shell commands, chat messages or instructions to other applications.

One bounded request runs off the motor thread. While waiting, the body remains
neutral. This is a single-instruction slice, not an unattended social/conversation
agent. Raw prompts and model responses are not saved by the application; the
validated intention, model name and latency are recorded. Check the server's own
logging policy separately.

For `LOOK_AT` or `REACH`, pass `--world ../myumiq-world.json`:

```json
{"objects":[{"name":"point","position":[0.3,0.3,1.4],"source":"fixture"}]}
```

This is explicit fixture geometry in the canonical stage frame, not screen
perception or an observed player. It must not be represented as a detected person.

## Inspect experience

Each run contains `session.json`, `result.json`, `output-events.jsonl`, and PAMIQ
state directories. Transitions are stored under
`states/<checkpoint>/data/experience/buffer.jsonl`:

```powershell
& $miPython -m myumiq_vrchat replay ../myumiq-runs/llm-001/states/<checkpoint>/data/experience/buffer.jsonl
```

`ReplayBuffer.load_state` validates the complete schema before replacing existing
data; it can be used for later offline training. This command reads records and
does not re-send historical actions. PAMIQ also stores its own timestamp/time
metadata in its standard format; only load checkpoints you created/trust. The CLI
does not load arbitrary PAMIQ pickle checkpoints or restore live leases.

`frames_attempted` means frames handed to local IPC, not acknowledged hardware
motion. `simulated` body signals come only from the mock sensor. `openvr_raw`
signals are device tracking, not avatar IK. The current reward measures only
device-pose progress and must not be interpreted as avatar, task or social success.
`avatar_visual_confirmation` remains false in automatic reports unless a separate
visual review records that evidence.

## Live gates and calibration

Read the parent local memo before any live operation. Required order:

1. Confirm a persistent console display and recovery route independently of RDP.
   Confirm the selected GPU using actual process/runtime/DXGI measurements.
2. Use official binary distributions of the selected Virtual HMD and VMT v0.15.
   Back up registrations/settings externally before controlled setup. Preserve
   unrelated drivers and workloads. Never build the suspended custom driver.
3. On the console display, pass the minimal Scene-client render/timing gate.
   Re-measure raw/game/render poses, VSync, frame timing and GPU/VRAM. A successful
   RDP DXGI VBlank wait alone is not this gate.
4. Establish explicit canonical-stage→driver transforms and neutral head/hand
   poses. Check VMT calibration, indices, roles, Index-compatible profile,
   buttons/axes and skeleton readback. Mode changes require the correct initial
   registration; the application will not restart SteamVR or replace devices.
5. Create a machine-local `LiveConfig`. Print its schema with
   `python -m myumiq_vrchat schema live`. Fill the separate `vmt_from_stage` and
   `hmd_from_stage` transforms, full `safe_target`, owned indices and ports.
   Set `calibrated` and `console_validated` true **only after those checks pass**.
   Identity transforms in tests are not a universal live calibration.
6. Run the producer in the same logged-in console session as SteamVR. The CLI
   rejects RDP/session 0, a missing running vrserver, a different HMD serial, or
   another device occupying the expected controller roles. It never performs a
   session switch. Verify the first neutral pose and tracking before adding goals.
7. Launch the current official VRChat in VR mode with the dedicated `--profile=1`
   and use a private empty home. Authentication is manual. Check head, hands,
   fingers and each binding visually; accepted OSC and device readback are not
   sufficient. No public chat, messages, voice or instance joining is automated.

Once those independent gates have passed:

```powershell
& $miPython -m myumiq_vrchat run --output ../myumiq-runs/live-001 `
  --live-config ../myumiq-live.json --hmd-serial '<verified HMD serial>' `
  --goal '{"skill":"WAVE","hand":"right","duration_s":2}' --duration 6
```

Add `--llm-config` and `--prompt` only after the fixed-intention live test passes.
Use `--autonomous --llm-config ...` for repeated finite-intent selection. It keeps
one LLM request outstanding and returns to WAIT while deliberating. A promoted
motor residual can be loaded with `--motor-policy <policy.json>`.
Add `--memory-state <memory.json>` to atomically persist recent outcomes and
player familiarity outside the repository; the next autonomous session includes
that memory in deliberation.
The output process exclusively owns OSC/UDP. It sends all 18 button channels,
9 trigger channels, 4 sticks and their touch/click state for each hand, plus finger
scalars and Apply; startup and shutdown also cover controls unused by this producer.
No VMT head override, global Reset, driver registration, display/GPU reassignment,
or runtime setting mutation is performed by this application.

## Stop and recovery

Ctrl+C requests PAMIQ teardown; the output process neutralizes and disables only
its owned VMT hands and sends the configured safe pose to the separate HMD. The
optional local `stop_policy: "safe_pose"` instead keeps the two owned hands
connected at the calibrated neutral pose while releasing every input. This policy
applies to startup, timeout and clean shutdown; timeout still halts the producer
lease. Use it only after verifying that neutral pose in the current environment.
The default remains `disable_owned`; global Reset is rejected by the live config.
Re-check role assignment when re-enabling a disconnected controller; do not infer
an avatar's hand assignment from valid raw tracking alone. The
HMD protocol has no sender-timeout invalidation. A new run always starts neutral
with a new lease. Pause/resume does not silently restore motion.

Producer silence/crash triggers configurable heartbeat/target deadlines and
bounded repeat stop attempts. UDP delivery is not guaranteed. Killing the whole
process tree, supervisor failure, Windows suspend or a crashed runtime can prevent
cleanup. Do not claim long-duration reliability from a short software test.

For environmental rollback, use the verified local before/after backups. Stop
only the VRChat/SteamVR/HMD/VMT/Sunshine/VDD components introduced for the trial,
in that order when applicable. Do not stop Python, WSL, VM or other GPU workloads.
Long-duration RDP-free/Moonlight-disconnect tests remain separate live gates.

## Replay motor update

Train and evaluate a WAVE residual from an existing real-device replay outside the
repository:

```powershell
& $miPython -m myumiq_vrchat train-motor <buffer.jsonl> `
  --output ../myumiq-runs/wave-right.json --skill WAVE --hand right
```

Training uses only active `openvr_raw` feedback, with a chronological 80/20 split.
It writes a `.candidate.json` when the holdout promotion threshold is not met.
Never pass a candidate to `--motor-policy` merely because training completed.

## Voice gates

The production chain is `audio capture -> Silero onset/end -> ASR -> structured
conversation decision -> cancellable TTS -> dedicated VRChat microphone route`.
Speech onset publishes a body event immediately and interrupts current output;
ASR and LLM work run outside the motor thread. Verify the dedicated audio route
with a loopback recording before selecting it in VRChat. A physical speaker plus
Stereo Mix is not considered a dedicated route and can echo other users.
For hearing, prefer `loopback_speaker_name` so the runtime captures the exact
VRChat output endpoint through WASAPI rather than a physical microphone or Stereo
Mix. `input_gain` is bounded to 0.1–16 and must be measured on the selected route.
The Silero ONNX adapter retains the required 64-sample context between 512-sample
frames; verify this before compensating for missing detections with gain.

Print `python -m myumiq_vrchat schema voice` and keep the resulting config outside
the repository. Both device indices have mandatory name checks, so Windows device
reordering fails closed instead of routing voice to a physical speaker. Add
`--voice-config <voice.json>` to an LLM run. The initial Windows vertical slice
uses cancellable Japanese SAPI Haruka output; the model boundary remains replaceable
by Qwen3-TTS after its latency and interruption behavior are measured.

## Vision gates

Print `python -m myumiq_vrchat schema vision` and keep the config outside the
repository. `--vision-config <vision.json>` captures only the visible VRChat client
rectangle and runs a small ONNX detector at a bounded 0.2–10 Hz. Its temporal
tracks update WorldState while motor output remains at 20–120 Hz. Vision geometry
is coarse monocular geometry: it can drive `LOOK_AT`, but `REACH` accepts only
calibrated fixture/manual geometry. PE-Spatial and V-JEPA remain future feature
encoders; neither alone supplies named player boxes or metric coordinates.

### Load models before starting live devices

For a supervised startup, supply both `--prepared-file <new path>` and
`--start-file <new path>` outside the repository. The CLI loads its configured
vision and speech models, writes the prepared marker, and waits at most 600
seconds for the start marker. It does not start capture or motor output during
this wait. Existing markers are rejected to prevent accidental replay of an
old start signal. Use fresh paths for every run.

The launcher should wait for model readiness, start and validate the private VR
session, release the previous pose owner, then create the start marker. The live
configuration is read after that signal. Keep the normal watchdog deadlines;
this handshake does not establish avatar tracking or guarantee scheduling latency.

`voice-ready.json` records audio-capture startup. Final `voice_diagnostics` reports
received blocks, VAD frames, post-gain peak, first/last capture timestamps, and
speech onset/end counts, without saving additional raw conversations.

In purpose mode, conversation, thought, body selection and motor execution are
independent. Speech onset interrupts output and releases controller movement;
dialogue/TTS completion is not required before the body can act. The body decision
model receives current observations plus conversation history. Requests invalidated
by a newer generation or observation are discarded. Learning proceeds separately
and installs newly acquired motions between actions.

Verify voice delivery in stages: nonzero dedicated Cable recording, VRChat's own
microphone-selection/start log, then unmuted state and reception by another test
participant. A nonzero Cable recording alone does not prove in-world delivery.
For OSC controls, confirm OSC is actually enabled in VRChat's Action Menu and
inspect the startup/runtime state; a preference write or an open UDP port alone
does not establish that input commands are accepted.

## Supervised visual gaze learning

This path is implemented but requires live calibration and before/after validation.
It is an initial learned inverse controller, not a SAC result.

1. In the approved private VR environment, choose a unique textured stationary
   object. Save its image template outside the checkout at the same client size
   used for measurement. Avoid mirrors, avatar thumbnails, repeating lamps, or
   anything that moves independently of the camera.
2. The usual launcher must pass the console/device gates and hand off its neutral
   owner. Run `scripts/collect_visual_gaze.py --live-config <config> --hmd-serial
   <serial> --template <image> --output <new-directory>`. This bounded experiment
   moves only the head within ±0.09 yaw/±0.07 pitch radians, records 40 settled
   visual transitions through the PAMIQ replay adapter, and closes its existing
   output supervisor. It does not launch or reconfigure VRChat.
3. Run `python -m myumiq_vrchat train-gaze <experience.jsonl> --output <candidate.json>`.
   Holdout prediction accuracy is reported separately from task improvement.
4. Use a vision config with `backend: "template"`, `model: <image path>`,
   `target_name: "calibration-target"`, `confidence: 0.85`, and `hz: 4`. The template
   detector rejects ambiguous matches; no detection means no valid feedback.
   Evaluate `run --gaze-policy <candidate.json>` with a finite LOOK_AT intent for
   that target, using the usual approved startup. Compare actual image error and
   convergence time against an untrained baseline over separate starting poses.
   Never call a successful fit alone a learned skill improvement.
# Conversation model evaluation

Run `python scripts/benchmark-conversation.py --url http://127.0.0.1:18082/v1
--model <server-alias> --output <new-file-outside-repo.json>` against an isolated
local server. This exercises the same conversation parser and retry path used by
the agent, including Japanese replies, hand selection, negation, unknown targets,
and unsupported skills. Inspect the actual replies as well as exact intent
accuracy; valid JSON does not prove understanding or a natural response.

Compare candidates with the same server build, quantization, thread count and
context size. Record process working set separately. The eight cases are a small
development gate, not a general Japanese benchmark. If prompts are tuned using
these cases, a fresh held-out evaluation is required before claiming selection
quality. A fast model that emits the wrong intention does not pass the gate.
