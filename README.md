# MyuMIQ-VRChat

**MyuMIQ-VRChat** is an experimental embodied AI agent for VRChat that combines **local LLM-driven intent**, **online reinforcement learning**, **real-time perception**, and **low-latency voice conversation**.

> **The LLM decides what it wants to do; reinforcement learning learns how to do it in the world.**

> 🚧 **Status:** design / early implementation. The architecture below describes the target system, not a finished feature set.

## What is MyuMIQ?

MyuMIQ is inspired by [PAMIQ](https://github.com/MLShukai/pamiq-core) and its approach to embodied, continuously learning agents.

The current working expansion of **MYUMIQ** is:

> **Model-guided Yet Unprompted Machine Intelligence with Q-functions**

- **Model-guided** — a local LLM interprets the world and forms high-level intentions.
- **Yet Unprompted** — the agent should act without waiting for a human prompt every time.
- **Machine Intelligence** — perception, memory, language, action, and learning form one continuous system.
- **Q-functions** — reinforcement learning evaluates and improves behavior from experience; the initial direction is an off-policy method such as SAC.

PAMIQ explores autonomous embodied behavior driven by curiosity. MyuMIQ explores what happens when **language-model cognition and intent become part of the learning loop**.

## Core idea

A conventional reinforcement-learning transition looks like:

```text
(state, action, reward, next_state)
```

MyuMIQ preserves the LLM's intent as part of the experience:

```text
(state, llm_goal, action, reward, next_state, outcome)
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

The body policy then learns **how to approach someone when the intent is to listen**. The same physical skill can eventually develop different behavior for different intentions such as `listen`, `follow`, `playful`, or `start_conversation`.

## Target architecture

```text
                       VRChat Desktop
                  ┌──────────┴──────────┐
                  │                     │
             video / audio          OSC state
                  │                     │
                  ▼                     ▼
          ┌─────────────────────────────────┐
          │            Perception           │
          │                                 │
          │ spatial perception              │
          │ temporal visual representation  │
          │ speaker direction               │
          │ streaming ASR                   │
          └───────────────┬─────────────────┘
                          ▼
                     World State
                          │
                ┌─────────┴─────────┐
                │                   │
             Memory              Drives
                │        boredom / curiosity /
                │        social / fatigue
                └─────────┬─────────┘
                          ▼
                     Local LLM
                          │
                  Structured Goal
                          │
                          ▼
                Goal-conditioned RL
                     SAC policy
                          │
               forward / strafe / turn
                          │
                          ▼
                     VRChat OSC
                          │
                          ▼
                    Real outcome
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
        Replay Buffer             Goal outcome
             │                         │
        SAC training             memory / stats
             │                         │
             └────────────┬────────────┘
                          ▼
                     next decision
```

## Intended components

### PAMIQ runtime

Use [pamiq-core](https://github.com/MLShukai/pamiq-core) as the primary runtime for asynchronous inference, experience collection, training, model synchronization, and persistence where practical.

Use [pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat) as the main reference for VRChat sensing and control. Prefer dependencies and application-level adapters over copying or forking PAMIQ code.

### Perception

Do not send every frame to a large VLM. The current direction is to combine lightweight/pretrained visual representations such as:

- Meta PE / PE-Spatial for spatial perception;
- Meta V-JEPA for temporal/world-state representation;
- optional heavier VLM calls only when semantic inspection is actually needed.

Perception backends should be replaceable through small adapter interfaces.

### Local cognition

A fast local LLM, initially in roughly the 1–3B class, should generate high-level structured goals rather than direct WASD-like controls.

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
```

The LLM is responsible for **what to do and why**. The learned body policy is responsible for **how to physically do it**.

### Reinforcement learning

The initial direction is **goal-conditioned SAC** because real VRChat experience is expensive and should be reusable through a replay buffer.

Initial continuous actions:

```text
forward   [-1, 1]
strafe    [-1, 1]
turn      [-1, 1]
look_yaw  [-1, 1]
```

Training should begin with a small, measurable task:

> Detect a target person, approach without getting stuck, stop at an appropriate distance, and face the target.

Experience gathered in actual VRChat should be persisted and reused for later training.

### Real-time voice

Voice interaction should be fully local and streaming:

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

### Memory and drives

The agent should maintain persistent state beyond chat history, initially including:

```text
boredom
curiosity
social
fatigue
familiarity[player]
affinity[player]
recent interactions
```

These values can influence LLM goals and, where useful, become part of the RL observation.

## Two learning loops

### Body learning

```text
LLM goal
   ↓
body policy
   ↓
VRChat action
   ↓
physical/social outcome
   ↓
reward
   ↓
policy update
```

This teaches the agent how to execute intentions more effectively.

### Cognitive experience

The LLM itself does not need RL fine-tuning initially. Instead, outcomes can be accumulated as statistics and memories:

```text
"approach a person standing alone" → often successful
"interrupt a busy group"          → often unsuccessful
```

Those experiences are fed back into later LLM decisions. A contextual bandit or learned goal-value model can be added later.

## Runtime rates

Different parts of the system should run at different frequencies rather than forcing everything into one frame loop.

| Component | Approximate starting rate |
| --- | ---: |
| Body policy inference | 20–50 Hz |
| VAD / audio events | high frequency |
| Spatial perception | 5–10 Hz |
| Temporal visual encoder | 2–10 Hz |
| Drive updates | 1–5 Hz |
| High-level LLM goal selection | 0.2–2 Hz |
| ASR | while speech is active |
| TTS | while the agent is speaking |

## Target hardware

Initial development target:

- Windows
- VRChat Desktop mode
- NVIDIA RTX 2080 Ti 11 GB
- local inference where practical

Small policies, VAD, memory, orchestration, and potentially a quantized LLM can run on CPU when that improves GPU headroom.

## Development roadmap

1. Validate PAMIQ + VRChat screen/audio/OSC integration.
2. Define a stable `WorldState` schema.
3. Implement a mockable fixed-goal body-control environment.
4. Persist VRChat transitions in a replay buffer.
5. Add a local LLM that emits validated structured goals.
6. Make the RL policy explicitly goal-conditioned.
7. Add streaming local ASR/TTS and barge-in.
8. Add PE-Spatial / V-JEPA perception adapters.
9. Feed accumulated outcomes back into memory and future goal selection.

The first major milestone is:

> **An LLM-generated structured goal conditions the body policy, VRChat executes the action, and the resulting transition is stored together with the original goal for learning.**

## Guardrails

Avoid:

- asking an LLM to generate low-level movement every frame;
- feeding every video frame to a large VLM;
- tightly coupling the codebase to one ASR/TTS/LLM model;
- heavily modifying `pamiq-core` when an application-level adapter will work;
- training exclusively from scratch inside live VRChat;
- fine-tuning the LLM before simpler outcome-memory approaches are tested;
- placing perception, conversation, control, and training into one giant event loop.

## Relationship to PAMIQ

MyuMIQ-VRChat is an independent experimental project inspired by and intended to build on the open-source PAMIQ ecosystem. It is **not an official MLShukai/PAMIQ project**.

Upstream projects:

- [MLShukai/pamiq-core](https://github.com/MLShukai/pamiq-core)
- [MLShukai/pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat)

## License

A project license has not been selected yet.
