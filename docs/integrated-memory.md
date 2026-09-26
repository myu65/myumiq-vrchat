# Shared memory, conversation and continuing purposes

Enable `purpose` in the autonomous configuration. Set `purpose.memory` to an
outside-repository SQLite path; otherwise it uses the purpose-state path with a
`.sqlite3` suffix. This database is owned by the slow executive. The motor loop
does not query SQLite or wait for model inference.

## One decision context

`agent_memory.py` keeps working conversation, focus, plan and attention;
evidence-bearing episodes; retrieved knowledge; explicitly identified partners;
and skill results grouped by conditions and observation scope. The same snapshot
is supplied to shared dialogue and independent embodied decisions. Legacy
configurations without a Decision Layer also use it for purpose generation.
PAMIQ actuation metadata includes its revision, commitment ID, topic and partner.

Purpose descriptions can continue across several finite plans. A completed plan
does not prove that a social purpose has succeeded. Visual alignment, observed
posture execution and registered learning results update the applicable criteria.
Unknown outcomes remain unknown. Repeated measured failures in the same
conditions prevent blindly repeating that step and request replanning.

On restart, historical memory remains retrievable, while commitments require
revalidation. Active plans and conversation turns are not replayed as commands.

## Conversation and body

With purpose mode enabled, the audio pipeline publishes speech-onset and
transcribed-utterance events. It does not run a second conversation agent.
Speech onset invalidates stale replies and stops queued speech. Separate bounded
planning and dialogue requests run concurrently; waiting for either does not
block the other. The shared executive commits their results and memory updates.
Dialogue uses the current purpose and memory, with explicit historical speakers.
With a Decision Layer configured, different model weights/services handle speech
and body choice. Conversation only writes utterances/replies into memory. The
body model receives that history together with current world/body state and
outcomes, including when no one is speaking. Speech output has no action field
and cannot directly set head attention, gestures or posture. New utterances
invalidate older pending body decisions; speech-model delays do not block them.

## Identity and evidence

Visual track IDs provide short-term continuity, not personal identity. Anonymous
mixed audio is never assigned to the closest visible person. An explicit known
speaker identifier or a time-limited operator association is required for social
memory. Self-reported facts are stored with source episodes and remain unverified.
LLM-proposed quotations must occur in the heard utterance.

TTS submission is not proof that another player heard the reply. Goal progress
and capability statistics retain that distinction.

## Bounded validation without a conversation partner

While purpose autonomy is active, `body_console send --session <session>` accepts:

```json
{"kind":"utterance","speaker_id":"fixture-person","text":"こんにちは。猫のミケと暮らしています。"}
```

Follow with a question about the earlier statement and inspect `dialogue_reply`
in `goals.jsonl`, memory revision/commitment in status, and replay metadata.
These are explicitly `operator_text` inputs, not evidence of working ASR or
remote microphone delivery. Use a separate fixture database to avoid mixing
test identities into normal social memory.

`associate_speaker` accepts an explicit `speaker_id` and optional currently
observed `target`. The association expires after five minutes and assumes a
controlled single-speaker test; it is not diarization.

## Remaining execution boundaries

Targeted navigation, contact and autonomous avatar-menu selection require additional
validated skills. [EXPLORE_HOME](exploration.md) supplies bounded mapless exploration.
Avatar exploration is a useful future continuing purpose:
inspect observed candidates, record preferences and unknown properties, try an
available selection skill, observe the changed avatar, and compare measured
body compatibility. Seeing an avatar thumbnail alone does not establish full-body
compatibility, performance, or that the agent can select it.
