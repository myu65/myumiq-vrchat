# MyuMIQ-VRChat

**MyuMIQ-VRChat** is an experimental embodied AI agent for VRChat that combines **local LLM-driven intent**, **virtual embodiment**, **online learning**, **real-time perception**, and **low-latency voice conversation**.

> **The LLM decides what it wants to do; learned motor policies learn how to do it in the world.**

> 🚧 **Status:** design / early implementation. The architecture below describes the intended system, not a finished feature set.

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

## License

A project license has not been selected yet.
