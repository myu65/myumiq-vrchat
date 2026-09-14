# AGENTS.md

This repository is the application repository for **MyuMIQ-VRChat**.

## Mission

Build a local embodied AI agent for VRChat that combines:

1. PAMIQ-based real-time sensing / acting / learning infrastructure;
2. local LLM cognition that emits structured high-level intentions;
3. goal-conditioned reinforcement learning for body control;
4. persistent experience from real VRChat sessions;
5. local, low-latency streaming voice conversation;
6. lightweight spatial and temporal perception.

The defining loop is:

```text
World State
→ LLM Structured Goal
→ goal-conditioned body policy
→ VRChat action
→ outcome / reward
→ replayable experience containing the original LLM goal
→ policy + cognitive-memory improvement
```

Do not reduce this project to a conventional voice chatbot or to an LLM that directly presses movement keys.

## Read upstream first

Before designing core runtime code, inspect the current upstream implementations and examples:

- https://github.com/MLShukai/pamiq-core
- https://github.com/MLShukai/pamiq-vrchat
- relevant PAMIQ recorder / curiosity examples when useful

Understand PAMIQ's abstractions for sensors, agents, models, trainers, buffers, inference/training concurrency, model synchronization, and persistence before introducing parallel abstractions here.

Prefer using PAMIQ as a dependency. Do not fork, vendor, or substantially modify upstream PAMIQ unless there is a demonstrated blocker that cannot be solved with an application-level adapter.

## First task in a fresh implementation session

Before large implementation work:

1. inspect this repository and relevant PAMIQ upstream code;
2. create or update `docs/architecture.md` with the actual proposed module boundaries, data flow, concurrency model, model interfaces, and persistence format;
3. identify the smallest vertical slice that can be executed and tested;
4. implement that slice rather than building speculative infrastructure.

Do not stop after producing a plan when implementation is possible.

## Architectural boundaries

Keep replaceable interfaces around model-specific components. A likely package shape is:

```text
src/myumiq_vrchat/
  audio/
  perception/
  world/
  cognition/
  memory/
  behavior/
  rl/
  vrchat/
  runtime/
```

The exact structure may change if PAMIQ's abstractions suggest a cleaner arrangement.

Important logical boundaries:

- **Perception**: raw screen/audio/state → structured signals / embeddings.
- **WorldState**: stable structured representation of the current environment.
- **Drives & Memory**: persistent internal state and interaction history.
- **Cognition**: WorldState + drives + memory → validated structured goal.
- **Behavior / RL**: state + structured goal → low-level continuous control.
- **VRChat I/O**: screen/audio capture and OSC input/output.
- **Conversation**: streaming VAD/ASR/LLM/TTS with interruption support.
- **Learning**: replay, reward, training, evaluation, and model promotion.

Conversation and body-control loops must not block each other.

## Structured goals

LLM output consumed by control code must be schema-validated structured data, not free-form prose.

Initial conceptual fields:

```json
{
  "skill": "APPROACH",
  "target": "player_1",
  "desired_distance": 1.2,
  "intent": "listen",
  "urgency": 0.6
}
```

Start with a small stable skill vocabulary such as:

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

Do not make the LLM generate per-frame WASD/OSC values.

## Learning data

A transition must preserve the goal that caused the action. At minimum persist:

```text
timestamp
state
goal
action
reward
next_state
outcome
```

Where practical, keep timestamp references to recorded raw video/audio so reward or perception failures can be investigated later.

The core policy should be designed as goal-conditioned from the start:

```text
action = policy(state, goal)
```

The initial algorithm direction is off-policy continuous-control RL (currently SAC) because actual VRChat experience is valuable and should be replayable. Re-evaluate this choice based on measured behavior; do not treat SAC as immutable architecture.

## Initial milestone

Prioritize this end-to-end milestone over broad feature coverage:

1. obtain a mock or real VRChat observation;
2. obtain a validated structured goal (initially it may be fixed or mocked, then LLM-generated);
3. condition the body policy on that goal;
4. emit VRChat-compatible continuous actions;
5. observe the result;
6. write a transition including the goal to persistent replay storage;
7. load that transition back for training/testing.

Only after this loop works should the project expand into richer conversation, drives, social rewards, and sophisticated perception.

## Perception

Do not use a large VLM on every frame.

Current candidates include Meta PE / PE-Spatial and V-JEPA-style temporal representations. Treat them as replaceable adapters. Heavier VLM calls should be event-driven or attention-driven when semantic inspection is required.

Do not hard-code assumptions that make later replacement with DINO or another encoder difficult.

## Voice

Voice is local-first and streaming. The pipeline should support:

```text
VAD → streaming ASR → streaming LLM → streaming TTS → VRChat microphone
```

Barge-in is a requirement: new human speech should be able to interrupt/fade the agent's current TTS and immediately trigger listening behavior.

Speech onset events should be usable by the body system before final transcription, e.g. stop walking and look toward the speaker.

Keep ASR/TTS backends behind interfaces. Do not tightly couple the runtime to one model.

## Hardware target

Primary development target:

- Windows
- VRChat Desktop mode
- RTX 2080 Ti 11 GB
- local inference where practical

Measure rather than guess. Record latency, throughput, and peak VRAM for model integrations. CPU execution is acceptable for components where it improves total system latency or GPU headroom.

## Testing

Real VRChat must not be required for every test.

Provide mocks or deterministic fixtures for at least:

- WorldState / target geometry;
- structured goal generation;
- VRChat action sink;
- audio events;
- replay-buffer serialization.

Test schema validation and persistence early.

For RL logic, prefer small deterministic environment tests before long training runs.

## Logging and observability

It should be possible to reconstruct why an action happened. Preserve enough structured telemetry to connect:

```text
world state
internal drives
LLM structured goal
policy observation
action
reward
outcome
latency
```

When integrating GPU models, also record useful VRAM measurements during benchmarks.

Avoid indiscriminately logging sensitive raw conversations or gigantic prompts when a compact structured summary is sufficient.

## Engineering rules

- Keep modules small and explicit; avoid one giant runtime file.
- Prefer typed Python and explicit schemas for cross-module data.
- Add tests with behavior-changing code.
- Do not introduce a large custom framework when PAMIQ or a small adapter is enough.
- Avoid premature distributed architecture; this is initially a single-machine system.
- Avoid silent fallback behavior that hides model or device failures.
- Keep model/backend selection configurable rather than encoded in core logic.
- Make commits coherent and explain non-obvious architectural decisions in `docs/architecture.md`.
- Update README claims when features actually become implemented; do not present planned capabilities as completed ones.

## Relationship to PAMIQ

MyuMIQ-VRChat is an independent project inspired by and built around the PAMIQ ecosystem. Do not imply that it is an official MLShukai/PAMIQ project.
