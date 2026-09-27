# MyuMIQ-VRChat

設計見直しの入口: [課題・継続する目的・会話/身体/学習の構成と移行順](docs/runtime-redesign.md)。
入力継続性、姿勢保持、経験からの候補学習の実装と、残る複合BodyGoal・自動採用を区別しています。
会話の並列化・一文単位の音声送信、60Hzの出力合成、30Hzの連続移動への変更は
[多周期構成への移行](docs/architecture.md#multi-rate-migration-2026-09-26)を参照。
actorの400ms先読み・MotionBuffer・検出から独立した高速視覚追跡も実装しました。
[実行境界と受入条件](docs/architecture.md#buffered-execution-and-visual-continuity-2026-09-27)を参照。
動作名によらず頭の高さ・手の位置などを組み合わせる
[条件付き身体目標](docs/body-conditions.md)を追加しました。現在は明示的な条件入力が入口で、
会話からの条件生成は次の段階です。
[条件を使う候補学習と動作全体の検証](docs/human-motion-practice.md)も追加しました。
学習・未見条件の比較・鏡での確認を分けます。
[ローカル学習と実機練習の反復](docs/practice-loop.md)は、回数を区切って実行できます。
候補は退行検証後に実機試験し、実機の開始姿勢を次の学習へ戻します。
採用先は練習内の設定だけで、通常モデルの設定や見た目の評価とは分けています。
視界を使う会話と行動判断、要求の採否、停止の保持は
[視界・会話・行動の接続](docs/visual-conversation-actions.md)を参照。

Current implementation boundaries and remaining acceptance work:
[integrated status and roadmap](docs/status-and-roadmap.md).

Start model configuration with [local / OpenRouter profiles](docs/model-adapters.md):
`myumiq models init` generates editable settings and `myumiq models check` tests the
configured dialogue, planning and decision contracts without operating VR devices.

**MyuMIQ-VRChat** is an experimental embodied AI agent for VRChat that combines **local LLM-driven intent**, **virtual embodiment**, **online learning**, **real-time perception**, and **low-latency voice conversation**.

> **The LLM decides what it wants to do; learned motor policies learn how to do it in the world.**

> 🚧 **Status:** the bounded LLM-to-avatar body slice has been observed in a private
> VRChat Home. Repeated autonomous finite-intent selection, simple drives and
> outcome memory are implemented. A low-rate open-vocabulary detector now feeds
> temporal WorldState, and Silero onset, configurable Qwen3-ASR/faster-whisper, structured
> conversation, cancellable local TTS and barge-in are connected off the motor
> thread. The individual models and real VRChat capture are measured; the final
> VRChat microphone round trip still requires separate in-app verification.
> Confirmed device experience can refine a separate motor candidate through
> PAMIQ; session-disjoint evaluation gates candidate eligibility. Live snapshot
> adoption is a separate explicit step.

The executable slice now connects a local LLM's validated intention to a canonical body, bounded procedural motor policy, independent output watchdog, and PAMIQ experience recording. It includes mock execution, official VMT v0.15 OSC hands/inputs/finger control, a separate VirtualHMD_OpenVR v0.1 head-pose adapter, and optional OpenVR device readback. The mock path and loopback failure cases are testable without VRChat. **Each deployment requires independent console-rendering, visible avatar-tracking and persistence checks; machine-specific evidence belongs in the parent local notes.** Custom OpenVR/display driver development is suspended. See [running the slice](docs/running.md), [VMT safety](docs/vmt-backend.md), [feasibility](docs/feasibility.md), and [architecture](docs/architecture.md).

Implemented intentions depend on the selected motor adapter and configured
capabilities. They include posture changes, WAVE, WALK_IN_PLACE, LOOK_AT,
configured `MOTION_<NAME>` gestures, bounded EXPLORE_HOME locomotion, and short
MOVE_FORWARD / TURN_LEFT / TURN_RIGHT attempts. The
legacy motor also supports nearby calibrated REACH. The LLM never generates
per-frame poses or buttons. OpenVR device feedback is not an observation of
VRChat's avatar IK or root position; person approach and contact need further sensing.

Full-body state, goals/constraints and eleven-point output are implemented, with
CC0 glTF retargeting and a small periodic imitation policy. Use the
[supervised body console](docs/body-console.md) for posture, calibration input,
learned in-place motion, status and clean shutdown. FBT calibration and avatar
motion must be verified for each deployment. Concurrent task execution,
contact-aware whole-body RL remains future work. Autonomous whole-body skill
selection is connected through [autonomous mode](docs/autonomous-mode.md);
see also [the full-body contract](docs/full-body.md).

An optional articulated actor now follows learned full-body motion sequences as
well as configured static postures. The executive can acquire a configured CC0
motion example, validate and register its prior, and restore it after restart.
Periodic and finite references use buffered joint trajectories by default and
retain the observed stopping posture. The selectable `feedback` mode remains
available for confirmed one-step replay acquisition. Configured lessons can be discovered and learned
without waiting for conversation. Per-lesson glTF/GLB sources and explicit
cross-skeleton retarget profiles support additional gestures. This is bounded
imitation acquisition; arbitrary motion invention and online task RL remain future
work. [Configured replay refinement](docs/experience-learning.md) now updates a
separate PAMIQ training model from confirmed device observations, evaluates it on
independent sessions and records ready/rejected candidates. It can run automatically
through the existing executive; it never synchronizes untested weights into live control.
Optional bounded update backtracking retains the same validation gates and saves
each rejected/accepted step for review before a later run adopts the candidate.
Controller movement can run with the learned gait while retaining separate
tracking-space and world-motion evidence. The Standard corpus has no waving clip;
the [full-body guide](docs/full-body.md) describes an actual CC0 waving source.

The articulated model supports [configurable local joint constraints](docs/joint-feasibility.md)
shared by training, tracker fitting and execution. These bound relative joint
rotations while allowing the whole body to turn or lie down. Existing unconstrained
actors require a new evaluated candidate; this does not establish live avatar IK
correctness or eliminate all self-intersection.

The optional [goal executive](docs/goal-directed-mode.md) connects free LLM purposes,
capability checks, sequential skill plans, observed outcomes and bounded imitation
learning. Missing capabilities remain explicit learning tasks; unsupported contact
and navigation are not treated as executable skills.

An optional [Decision Layer](docs/decision-layer.md) independently scores action
candidates using local Qwen/Nemotron multimodal rerankers or official TypeSafe Jev
(direct API or OpenRouter; text state only). It connects to the autonomous worker
through an isolated loopback service, preserves snapshot freshness and records
scores/selection in decision logs and references in replay. Benchmark tools cover
quality, candidate order, GPU memory and 2/4/8/16-candidate latency. Retrieval-model
judgment quality and 100 ms live latency are not established capabilities.

When both `purpose` and `decision` are configured, this separate model owns body
choice from world/body state, conversation history, drives and observed outcomes.
The chat model generates speech only; no body-command extraction runs on it.
Body decisions continue during speech and motion. The current bounded action
catalogue does not yet replace open-ended planning or supply whole-body RL.

[Role-specific model adapters](docs/model-adapters.md) let dialogue, independent
candidate selection and slow goal proposals use local chat or OpenRouter separately.
Jev keeps its own Decisions API scorer contract. Provider/model routing is fixed per
configuration with no automatic fallback. The optional `purpose.planner` proposes
goals through local capability checks and shared memory while the body decision
loop retains action ownership. Installed adapters can also replace generation,
scoring, chunk ASR, VAD, speech output and image detection. Native realtime speech
sessions remain a separate [design](docs/cognition-voice-adapters.md).

## What is MyuMIQ?

MyuMIQ is inspired by [PAMIQ](https://github.com/MLShukai/pamiq-core) and its approach to embodied, continuously learning agents.

The current working expansion of **MYUMIQ** is:

> **Model-guided Yet Unprompted Machine Intelligence with Q-functions**

- **Model-guided** — a local LLM interprets the world and forms high-level intentions.
- **Yet Unprompted** — the agent should act without waiting for a human prompt every time.
- **Machine Intelligence** — perception, memory, language, action, body state, and learning form one continuous system.
- **Q-functions** — reinforcement learning can evaluate and improve behavior from experience; the current direction includes off-policy methods such as SAC where they are appropriate.

PAMIQ explores autonomous embodied behavior driven by interaction and curiosity. MyuMIQ explores what happens when **language-model cognition and intent become part of that embodied learning loop**.

## Core idea

A conventional reinforcement-learning transition looks like:

```text
(state, action, reward, next_state)
```

MyuMIQ preserves both the LLM's intent and the agent's body state as part of experience:

```text
(world_state, body_state, llm_goal, action, reward,
 next_world_state, next_body_state, outcome)
```

For example:

```json
{
  "skill": "APPROACH",
  "target": "player_1",
  "desired_distance": 1.2,
  "intent": "listen",
  "urgency": 0.6
}
```

The LLM decides **what to do and why**. Behavior and motor policies decide **how to physically do it**.

The same physical skill can eventually develop different behavior depending on intent: `listen`, `follow`, `playful`, `start_conversation`, `avoid`, and so on.

## Virtual Body is a core concept

VRChat itself is not the agent's body model.

MyuMIQ maintains its own **Virtual Body** as a persistent estimate of self-state / virtual proprioception. It should eventually be able to represent concepts such as:

```text
root pose and velocity
head / chest / pelvis pose
left and right hand pose
elbows / knees / feet
support and contact state
gaze target
body orientation
motion style
```

Values that cannot be observed directly must be marked as estimated, predicted, or unavailable rather than silently treated as ground truth.

The Virtual Body is available to cognition, behavior, motor policies, and learning. It is not just an output buffer.

## Target architecture

```text
                 video / audio / environment state
                              │
                              ▼
                        Perception
                              │
                              ▼
                         WorldState
                              │
                ┌─────────────┴─────────────┐
                │                           │
           Memory / Drives              BodyState
                │                    virtual proprioception
                └─────────────┬─────────────┘
                              ▼
                          Local LLM
                              │
                       Structured Intent
                              │
                              ▼
                       Behavior Planner
                              │
                        MotionCommand
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 BodyState         Motor Policy
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       ActuationTarget
                              │
                     Backend Adapter
                              │
              ┌───────────────┴────────────────┐
              │                                │
     SteamVR virtual devices            OSC / fallback
     HMD + L/R controllers              auxiliary I/O
              │                                │
              └───────────────┬────────────────┘
                              ▼
                           VRChat
                              │
                     observed outcome
                              │
                              ▼
                         Body estimator
                              │
                              └────→ BodyState
```

The key boundary is:

```text
LLM → Intent → Behavior → MotionCommand → MotorPolicy → Backend
```

with a feedback path:

```text
VRChat / perception → Body estimator → BodyState
```

The LLM must not generate per-frame joint angles, controller poses, WASD values, or OSC packets.

## Motor skills

The motor layer should be decomposable rather than represented as one giant policy.

Likely skills include:

```text
RootLocomotionPolicy
GazePolicy
ReachPolicy
GesturePolicy
PosturePolicy
```

This is important for behaviors such as:

```text
APPROACH(person)
LOOK_AT(object)
FOLLOW(person)
REACH(object)
PAT_HEAD(person)
WAVE(person)
```

For example, `PAT_HEAD(person)` should become a reaching / hand-motion problem, not an LLM instruction like "move the right elbow by 15 degrees".

## Root locomotion vs whole-body motion

These are intentionally separate.

### Root locomotion

The initial navigation problem is high level:

- approach a target;
- stop at an appropriate distance;
- turn naturally;
- follow someone;
- avoid obstacles.

A procedural implementation can come first, with a replaceable RL locomotion policy later.

### Whole-body motion

Human-like gait and body motion are a later problem. The intended direction is not to discover a strange walk from scratch with unconstrained RL.

A future path may be:

```text
mocap
  ↓
imitation / motion prior
  ↓
task-conditioned RL refinement
```

The initial architecture should permit this without requiring a whole-body physics or imitation-training stack immediately.

## VRChat body backend

The preferred embodied backend is **SteamVR virtual devices**:

```text
MyuMIQ Virtual Body
        ↓
SteamVR device backend
        ├─ Virtual HMD
        ├─ Virtual Left Controller
        └─ Virtual Right Controller
        ↓
SteamVR
        ↓
VRChat PC in VR mode
```

This lets VRChat use its normal VR IK/input path for the avatar's head and hands.

OSC remains useful for auxiliary VRChat-specific features, avatar parameters, chatbox integration, debugging, and fallback control. The architecture must not tie the canonical Virtual Body to VRChat OSC addresses or SteamVR coordinate conventions.

The first implementation must treat the Virtual HMD/controller route as a **feasibility target to measure**, not as an already-proven capability.

See [docs/virtual-vr-environment.md](docs/virtual-vr-environment.md) for environment isolation, driver registration, low-resolution VR experiments, and the initial feasibility checklist.

## PAMIQ runtime

Use [pamiq-core](https://github.com/MLShukai/pamiq-core) as the primary runtime for asynchronous inference, experience collection, training, model synchronization, and persistence where practical.

Use [pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat) as a reference and dependency where its sensing / OSC components fit.

Do **not** fork or duplicate PAMIQ runtime machinery simply because MyuMIQ introduces a Virtual Body or SteamVR backend. Add application-level adapters around PAMIQ instead.

## Perception

Do not send every frame to a large VLM.

Current candidates include lightweight/pretrained visual representations such as:

- Meta PE / PE-Spatial for spatial perception;
- Meta V-JEPA-style temporal representations;
- optional heavier VLM calls only when semantic inspection is actually needed.

Perception backends should be replaceable.

A deliberately low-resolution Virtual HMD may be useful for both performance and AI perception; this must be benchmarked rather than assumed.

## Local cognition

A fast local LLM, initially in roughly the 1–3B class, should generate structured high-level goals rather than direct controls.

Example skills:

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

The LLM is responsible for **what to do and why**. Behavior and motor layers are responsible for **how to do it**.

## Reinforcement learning

The initial learning direction is **goal-conditioned, replayable experience**. Off-policy continuous-control methods such as SAC are candidates because real VRChat experience is valuable and should be reusable, but the algorithm is not an architectural requirement.

Experience should preserve at least:

```text
timestamp
world_state
body_state
intent / goal
motion command
motor action / actuation target
reward
next_world_state
next_body_state
outcome
```

A first measurable learned task can remain simple:

> Detect a target person, approach without getting stuck, stop at an appropriate distance, and face the target.

Later motor skills such as reaching or patting can be trained using the same Virtual Body abstractions.

## Real-time voice

Voice interaction should be local-first and streaming:

```text
VAD
 ↓
streaming ASR
 ↓
LLM streaming
 ↓
streaming TTS
 ↓
virtual audio device
 ↓
VRChat microphone
```

Important behavior:

- react to speech onset before ASR is complete;
- look toward the active speaker;
- allow barge-in / interruption while the agent is speaking;
- avoid a slow `ASR complete → LLM complete → TTS complete` serial pipeline.

Conversation and body-control loops must not block each other.

## Memory and drives

The agent should maintain state beyond chat history, eventually including concepts such as:

```text
boredom
curiosity
social
fatigue
confidence
tension
attention
familiarity[player]
affinity[player]
recent interactions
```

Internal state should be able to condition behavior and motion style without requiring the LLM to micromanage posture or joint motion.

## Runtime rates

Different parts of the system should run at different frequencies.

| Component | Approximate starting rate |
| --- | ---: |
| Body / controller target updates | 30–60 Hz |
| Body policy inference | 20–50 Hz |
| VAD / audio events | high frequency |
| Spatial perception | 5–10 Hz |
| Temporal visual encoder | 2–10 Hz |
| Drive updates | 1–5 Hz |
| High-level LLM goal selection | 0.2–2 Hz |
| ASR | while speech is active |
| TTS | while the agent is speaking |

Rendering rate does not need to equal motor-policy rate.

## Target hardware and environment

Initial development target:

- Windows
- NVIDIA RTX 2080 Ti 11 GB
- SteamVR
- VRChat PC
- Virtual HMD + Virtual Left/Right Controllers
- local inference where practical

The goal is **not human-quality VR rendering**. The Virtual HMD should support intentionally small render sizes / refresh rates so VRChat remains in its VR input path while preserving GPU headroom for local LLM, perception, and learning workloads.

Desktop + OSC remains a useful fallback and debugging path.

## Development roadmap

1. Validate the Virtual HMD + Virtual Left/Right Controller path in SteamVR and VRChat.
2. Benchmark deliberately low Virtual HMD render sizes / refresh rates and record GPU/VRAM/stability.
3. Define `BodyState`, coordinate frames, `MotionCommand`, `ActuationTarget`, and backend capabilities.
4. Implement a mock backend and minimal procedural root locomotion.
5. Map the body/control abstractions cleanly onto PAMIQ Agent / Environment / DataBuffer / Trainer concepts.
6. Validate a simple end-to-end `APPROACH(target)` vertical slice.
7. Persist transitions containing both the LLM goal and BodyState.
8. Add a local LLM that emits validated structured goals.
9. Introduce goal-conditioned learning where it gives measurable benefit.
10. Add streaming local ASR/TTS and barge-in.
11. Add richer perception adapters.
12. Add reach / gesture skills such as `PAT_HEAD` once free hand control is validated.

The first technical milestone is:

> **A minimal virtual HMD and two virtual controllers can drive a VRChat avatar reliably, at a measured low rendering cost, without breaking the normal VRChat/SteamVR environment.**

The first learning milestone remains:

> **An LLM-generated structured goal conditions a body policy, VRChat executes the resulting action, and the transition is stored together with the original goal and BodyState for later learning.**

## Development environment isolation

Normal VRChat use and MyuMIQ development should remain easy to separate.

Current direction:

- use a separate VRChat `--profile` for MyuMIQ;
- explicitly register / unregister or enable / disable the MyuMIQ SteamVR driver;
- keep start/stop scripts idempotent;
- avoid modifying or removing unrelated SteamVR drivers;
- use a separate Windows user only if simpler isolation proves insufficient.

Details: [docs/virtual-vr-environment.md](docs/virtual-vr-environment.md)

## Guardrails

Avoid:

- asking an LLM to generate low-level movement every frame;
- treating VRChat itself as the canonical body state;
- assuming unavailable body state is ground truth;
- hard-wiring Virtual Body concepts to one backend;
- feeding every video frame to a large VLM;
- tightly coupling the codebase to one ASR/TTS/LLM model;
- heavily modifying `pamiq-core` when an application-level adapter will work;
- training human-like gait entirely from scratch in live VRChat;
- building a giant custom scheduler / replay framework that duplicates PAMIQ;
- presenting unvalidated Virtual HMD/controller behavior as already working.

## Relationship to PAMIQ

MyuMIQ-VRChat is an independent experimental project inspired by and intended to build on the open-source PAMIQ ecosystem. It is **not an official MLShukai/PAMIQ project**.

Upstream projects:

- [MLShukai/pamiq-core](https://github.com/MLShukai/pamiq-core)
- [MLShukai/pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat)

## Platform rules

A separate profile or account can help isolate configuration, but it does not by itself determine whether a particular automated use is allowed by VRChat. Before public or unattended operation, check the current VRChat Terms of Service and platform rules.

Do not store VRChat, Steam, or other service credentials in this repository.

## Full-body development

Canonical eleven-point body state, goals/constraints and actuation are implemented,
with optional eight-tracker VMT output/readback and owned-device shutdown. A CC0
glTF import pipeline and phase-conditioned imitation model can be evaluated through
the existing watchdog and PAMIQ replay. Real VRChat eleven-point calibration and
full-body pose validation require per-deployment observation; offline motion
fitting does not prove avatar tracking or physical stability.
See [full-body documentation](docs/full-body.md).

## License

This repository is licensed under the [MIT License](LICENSE). Dependencies and official third-party driver distributions retain their own licenses.
## Autonomous whole-body operation

An optional [autonomous mode](docs/autonomous-mode.md) connects local cognition,
drives/memory, low-rate vision, voice, full-body skills and PAMIQ recording through
the existing body console. Optional service failures are explicit and recoverable;
device safety failures retain the existing stop behavior. Deployment verification
and online motor RL are reported separately.

Purpose mode uses [shared memory and dialogue](docs/integrated-memory.md):
continuing commitments, working conversation, evidence-bearing episodes,
speaker-associated knowledge and conditional capability outcomes feed the same
executive and reranker. Bounded head/hand overlays preserve disjoint body tasks.
Real partner recognition, delivered speech and general navigation are separate
validation requirements, not implied by this integration.

Optional [EXPLORE_HOME](docs/exploration.md) connects short controller turns and
advances to visual-change/novelty memory under a private-Home gate. This is bounded
mapless exploration; metric localization and obstacle-aware approach remain unavailable.
