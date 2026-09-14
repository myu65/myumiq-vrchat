# AGENTS.md

This repository is the application repository for **MyuMIQ-VRChat**.

## Mission

Build a local embodied AI agent for VRChat that combines:

1. PAMIQ-based real-time sensing / acting / learning infrastructure;
2. local LLM cognition that emits structured high-level intentions;
3. a persistent Virtual Body / virtual proprioception layer;
4. replaceable motor policies for locomotion, gaze, reaching, gesture, and posture;
5. replayable experience from real VRChat sessions;
6. local, low-latency streaming voice conversation;
7. lightweight spatial and temporal perception.

The defining idea is:

> The LLM decides what it wants to do; learned motor policies learn how to do it in the world.

Do not reduce this project to a conventional voice chatbot, a state machine that only presses movement keys, or an LLM that directly emits per-frame joint/controller commands.

## Read upstream and project docs first

Before designing core runtime code, inspect the current versions of:

- https://github.com/MLShukai/pamiq-core
- https://github.com/MLShukai/pamiq-vrchat
- relevant PAMIQ recorder / curiosity examples when useful
- Valve OpenVR driver documentation and samples when working on the SteamVR backend
- current VRChat SteamVR Input / launch documentation when working on VR integration
- `docs/virtual-vr-environment.md`

Understand PAMIQ's abstractions for sensors, agents, environments, models, trainers, buffers, inference/training concurrency, model synchronization, and persistence before introducing parallel abstractions here.

Prefer using PAMIQ as a dependency. Do not fork, vendor, or substantially modify upstream PAMIQ unless there is a demonstrated blocker that cannot be solved with an application-level adapter.

## First task in a fresh implementation session

Before large implementation work:

1. inspect this repository and relevant upstream code;
2. create or update `docs/architecture.md` with the actual proposed module boundaries, data flow, coordinate frames, concurrency model, model interfaces, PAMIQ mapping, and persistence format;
3. identify the smallest vertical slice that can be executed and tested;
4. implement that slice rather than building speculative infrastructure.

Do not stop after producing a plan when implementation is possible.

## First technical feasibility milestone

Before implementing sophisticated RL, imitation learning, voice, or perception, validate the preferred embodiment path:

1. create the smallest valid OpenVR / SteamVR driver;
2. expose a Virtual HMD plus Virtual Left and Right Controllers;
3. make SteamVR detect the devices;
4. run VRChat PC through SteamVR in VR mode;
5. change head and controller poses from a small test program;
6. verify that the VRChat avatar follows them;
7. measure VRAM, GPU usage, stability, and visual quality at deliberately small render sizes / refresh rates;
8. verify that disabling or unregistering the MyuMIQ driver restores normal SteamVR/VRChat use.

Treat this as a measured feasibility spike, not an already-proven capability.

Do not make PAMIQ depend directly on OpenVR details. The SteamVR driver is a backend for the canonical Virtual Body / motor interfaces.

## Architectural boundaries

A likely package shape is:

```text
src/myumiq_vrchat/
  audio/
  perception/
  world/
  body/
  cognition/
  memory/
  behavior/
  motor/
  learning/
  backends/
    steamvr/
    vrchat_osc/
    mock/
  runtime/
```

The exact structure may change if PAMIQ's abstractions suggest a cleaner arrangement. Do not create directories merely to satisfy this sketch.

Important logical boundaries:

- **Perception**: raw screen/audio/state → structured signals / embeddings.
- **WorldState**: stable structured representation of the current environment.
- **Virtual Body / BodyState**: the agent's current self-state / virtual proprioception.
- **Drives & Memory**: persistent internal state and interaction history.
- **Cognition**: WorldState + relevant BodyState + drives + memory → validated structured intent / goal.
- **Behavior**: intent / goal → `MotionCommand`-like body-level request.
- **Motor**: BodyState + MotionCommand → `ActuationTarget`-like low-level target.
- **Backend adapters**: convert canonical actuation targets into SteamVR virtual-device state, OSC, simulator commands, etc.
- **Conversation**: streaming VAD/ASR/LLM/TTS with interruption support.
- **Learning**: replay, reward, training, evaluation, and model promotion, preferably using PAMIQ facilities where they fit.

Conversation, body-control, perception, and training loops must not unnecessarily block each other.

## Virtual Body

VRChat itself is not the canonical body state.

The Virtual Body should eventually represent concepts such as:

```text
root pose / velocity
head pose
chest / pelvis pose
left/right hand pose
left/right elbow pose
left/right knee pose
left/right foot pose
support / contact state
gaze target / direction
motion style
```

Do not pretend unobservable values are ground truth. Where useful, body signals should carry metadata such as:

```text
valid
confidence
source
timestamp
```

or an equivalent explicit representation of observed / estimated / predicted / unavailable state.

Define coordinate frames early. Keep SteamVR / VRChat axis conventions, tracker indices, OSC addresses, and Euler-specific details out of the canonical body model. Convert them at adapter boundaries.

## Current state vs desired state

Do not merge these concepts into one structure.

Keep a clear conceptual distinction between:

```text
BodyState          # what the agent believes its body is doing now
MotionCommand      # what behavior wants the body to accomplish
ActuationTarget    # what a motor policy asks the backend to realize
```

A conceptual motor interface may look like:

```text
step(body_state, motion_command, dt) -> actuation_target
```

Keep interfaces minimal and evolve them from tested use cases rather than designing a giant robotics framework in advance.

## Structured goals

LLM output consumed by control code must be schema-validated structured data, not free-form prose.

Example:

```json
{
  "skill": "APPROACH",
  "target": "player_1",
  "desired_distance": 1.2,
  "intent": "listen",
  "urgency": 0.6
}
```

A possible initial skill vocabulary includes:

```text
EXPLORE
APPROACH
FOLLOW
LOOK_AT
LISTEN
TALK
WAIT
LEAVE
FLEE
REACH
PAT_HEAD
WAVE
```

Do not make the LLM generate per-frame WASD, OSC values, joint angles, or controller poses.

## Motor skills

Avoid treating motor control as one indivisible policy. Likely skill families include:

```text
RootLocomotionPolicy
GazePolicy
ReachPolicy
GesturePolicy
PosturePolicy
```

The initial implementation may use simple procedural policies.

### Root locomotion

Start with the smallest navigation problem: approach, follow, turn, stop at a reasonable distance, and avoid obvious obstacles.

A procedural root policy should be replaceable later by a learned goal-conditioned policy.

### Whole-body motion

Do not build whole-body gait RL first.

For human-like locomotion, the intended future direction is closer to:

```text
mocap
→ imitation / motion prior
→ task-conditioned RL refinement
```

rather than unconstrained RL discovering a gait from scratch in live VRChat.

Keep the architecture compatible with future whole-body policies without implementing a large physics or mocap stack now.

## Body capabilities

The canonical Virtual Body can contain more concepts than a particular backend can actuate or observe.

Represent backend capabilities explicitly where this prevents ambiguity, e.g.:

```text
supports_root_motion
supports_gaze
supports_hand_pose
supports_skeletal_hands
supported_tracker_segments
has_observable_root_pose
supports_contact_feedback
```

Do not silently discard unsupported commands.

## VRChat / SteamVR backend direction

The preferred embodied backend is currently:

```text
Virtual Body
   ↓
SteamVRVirtualDeviceBackend
   ├─ Virtual HMD
   ├─ Virtual Left Controller
   └─ Virtual Right Controller
   ↓
SteamVR
   ↓
VRChat PC in VR mode
```

The goal is to use VRChat's normal VR IK/input path for head/hands and controller actions.

`pamiq-vrchat` remains useful for sensing, OSC and reference implementations. OSC should remain available for avatar parameters, chatbox features, auxiliary controls, fallback movement, and debugging, but the Virtual Body must not be designed around OSC.

See `docs/virtual-vr-environment.md` for environment isolation and driver-registration guidance.

## Lightweight VR rendering

Human-quality VR rendering is not a requirement.

The Virtual HMD backend should allow deliberately small render sizes and lower refresh rates for benchmarking. Do not hard-code one value before measuring stability, GPU/VRAM cost, and perception quality.

Rendering rate may be lower than motor/controller update rate.

Do not pursue fully headless/no-render operation before a minimal low-resolution VR path works reliably.

## PAMIQ mapping

PAMIQ should remain the runtime / learning substrate where practical.

Before inventing custom scheduler, replay, synchronization, or persistence machinery, map MyuMIQ concepts onto current PAMIQ abstractions such as:

```text
Agent
Environment
Interaction
DataBuffer
Model
Trainer
```

Document that mapping in `docs/architecture.md`.

The Virtual Body and SteamVR backend do not justify duplicating PAMIQ's infrastructure.

## Learning data

A replayable transition should be able to preserve the intent that caused the action and the body context in which it occurred.

Conceptually:

```text
timestamp
world_state
body_state
goal / intent
motion_command
motor action / actuation_target
reward
next_world_state
next_body_state
outcome
```

Do not build a giant storage schema before a minimal end-to-end transition exists.

Off-policy continuous-control RL such as SAC is a current candidate because real VRChat experience is expensive and replayable, but do not make SAC an immutable architectural assumption.

## Initial learning milestone

After the virtual-device feasibility spike works, prioritize this vertical slice:

1. obtain a mock or real world/body observation;
2. obtain a validated structured goal (fixed/mocked first, LLM-generated later);
3. convert it into a MotionCommand;
4. execute a procedural or learned motor policy;
5. send the target through a mock or real backend;
6. observe the result and update BodyState;
7. write a transition containing goal + BodyState to persistent replay storage;
8. load that transition back for training/testing.

Only after this loop works should the project expand aggressively into richer conversation, drives, social rewards, imitation learning, and sophisticated perception.

## Perception

Do not use a large VLM on every frame.

Current candidates include Meta PE / PE-Spatial and V-JEPA-style temporal representations. Treat them as replaceable adapters. Heavier VLM calls should be event-driven or attention-driven when semantic inspection is required.

Do not hard-code assumptions that make later encoder replacement difficult.

## Voice

Voice is local-first and streaming:

```text
VAD → streaming ASR → streaming LLM → streaming TTS → VRChat microphone
```

Barge-in is a requirement: new human speech should be able to interrupt/fade the agent's current TTS and immediately trigger listening behavior.

Speech-onset events should be usable by the body system before final transcription, for example stopping movement and looking toward a speaker.

Keep ASR/TTS backends replaceable.

## Hardware target

Primary development target:

- Windows
- RTX 2080 Ti 11 GB
- SteamVR
- VRChat PC
- Virtual HMD + Virtual Left/Right Controllers
- local inference where practical

Desktop VRChat + OSC is a fallback / debugging path, not the preferred final embodiment path.

Measure rather than guess. Record latency, throughput, GPU utilization, and peak VRAM for model and VR integrations. CPU execution is acceptable where it improves total-system latency or GPU headroom.

## Environment isolation

Do not casually mutate the user's normal VR environment.

Development automation should prefer:

- a separate VRChat `--profile` for MyuMIQ;
- idempotent SteamVR driver registration checks;
- explicit enable/register and disable/unregister steps;
- no removal or modification of unrelated third-party drivers;
- a separate Windows user only if simpler isolation proves insufficient.

Never commit credentials, session tokens, passwords, or service secrets.

A separate account/profile is only configuration isolation; it does not by itself determine whether autonomous operation is permitted by VRChat. Do not assume public/unattended bot operation is allowed without checking current platform rules.

## Testing

Real VRChat must not be required for every test.

Provide mocks or deterministic fixtures for at least:

- WorldState / target geometry;
- BodyState and body estimator inputs;
- structured goal generation;
- motor-policy behavior;
- backend action sink;
- audio events;
- replay serialization.

Test coordinate conversion and schema validation early.

For RL logic, prefer small deterministic environment tests before long training runs.

## Logging and observability

It should be possible to reconstruct why an action happened. Preserve enough structured telemetry to connect:

```text
world state
body state
internal drives
LLM structured goal
motion command
policy observation
action / actuation target
reward
outcome
latency
backend capability / failure information
```

When integrating GPU models or VR rendering, record useful VRAM / GPU measurements during benchmarks.

Avoid indiscriminately logging sensitive raw conversations or gigantic prompts when a compact structured summary is sufficient.

## Engineering rules

- Keep modules small and explicit; avoid one giant runtime file.
- Prefer typed Python and explicit schemas for cross-module data.
- Add tests with behavior-changing code.
- Do not introduce a large custom framework when PAMIQ or a small adapter is enough.
- Avoid premature distributed architecture; this is initially a single-machine system.
- Avoid silent fallback behavior that hides model, device, or capability failures.
- Keep model/backend selection configurable rather than encoded in core logic.
- Make commits coherent and explain non-obvious architectural decisions in `docs/architecture.md`.
- Update README claims when features actually become implemented; do not present planned capabilities as completed ones.

## Relationship to PAMIQ

MyuMIQ-VRChat is an independent project inspired by and built around the PAMIQ ecosystem. Do not imply that it is an official MLShukai/PAMIQ project.
