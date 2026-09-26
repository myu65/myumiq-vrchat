# Goal-directed autonomy

Enable the optional `purpose` object in the autonomous configuration. Without it,
the existing single-intent executive remains available.

For the current separate conversation/decision/planner setup, generate a complete
configuration with [models init](model-adapters.md). The example below documents
the older no-Decision path and the reusable learning settings.

```json
{
  "llm": {"base_url": "http://127.0.0.1:18487/v1", "model": "LiquidAI/LFM2.5-1.2B-JP-202606"},
  "memory": "D:/myumiq-data/memory.json",
  "purpose": {
    "state": "D:/myumiq-data/purposes/state.json",
    "max_plan_s": 120,
    "retry_s": 10,
    "learning": {
      "motion_source": "D:/myumiq-data/motions/AnimationLibrary_Godot_Standard.gltf",
      "motion_license": "D:/myumiq-data/motions/LICENSE",
      "source_url": "https://quaternius.com/packs/universalanimationlibrary.html",
      "clip": "Walk_Loop",
      "timeout_s": 60
    }
  }
}
```

Paths must identify your own assets; the example does not download them. State and
training outputs live outside the repository. Vision, voice and decision settings
remain the same as [autonomous mode](autonomous-mode.md).

## Execution

With `decision` configured, the independent model owns body choice directly from
world/body state, dialogue history, drives, commitment and outcomes. It runs
during speech and motion, without asking the chat model to propose or extract
body actions. Its bounded executable catalogue currently replaces the free-plan
generator; open-ended planning and discovering new skills through this model
remain unfinished. See [Decision Layer](decision-layer.md).

The following legacy plan-generation path applies when `decision` is absent:

1. `purposes.py` asks the local LLM for a free purpose, reason, observable success
   description and short capability plan. Inputs include WorldState, drives,
   body summary, recent memory, capability statistics and previous goal outcomes.
2. `capabilities.py` checks adapters, prerequisites and current target geometry.
   Known actions become typed motor intents. A supported composition expands into
   steps; missing capabilities become learning tasks. Unknown names never become
   arbitrary executable code or body coordinates.
3. `purpose_runtime.py` runs the validated plan outside the motor loop. Configured
   Decision Layers use the independent path described above instead of this planner.
4. Existing motor policies, Virtual Body, actuation bounds and watchdog execute
   each step. Fresh body feedback updates capability statistics and memory.
   Failed execution ends the plan; unknown evidence is never counted as success.
5. The next LLM request includes those outcomes. `goals.jsonl` records creation,
   learning, step outcomes and completion. `decisions.jsonl` and PAMIQ
   `intent_metadata` include goal ID, purpose, step and capability.

`body_console run --autonomous-config <config> ...` prepares the executive;
`body_console send --session <session> '{"kind":"autonomous"}'` enables it after
the existing private-instance/device gates. Manual commands interrupt plans and
dialogue; speech only supersedes old replies. Autonomous planning continues
during listening and reply generation. Dialogue never applies body actions;
history is input to the separate decision model. Late responses are discarded
within their respective generation. Restart restores
statistics and verified learned artifacts, never an interrupted active plan.

## Capabilities and learning

| Request | Current behavior |
| --- | --- |
| LOOK_AT, WAVE, REACH, posture skills | Existing bounded motor adapters |
| EXPLORE | Observe up to three visible targets, least observed first; no navigation |
| EXPLORE_HOME | Optional private-Home controller exploration with fresh visual feedback and short input leases; see [exploration](exploration.md) |
| HOLD_HAND | Maintain the reaching pose at a calibrated target; no grasp/contact claim |
| TALK | Submit Japanese speech to the local voice service; delivery remains unverified |
| WALK_IN_PLACE | Loaded imitation policy, or create and execute an imitation learning task |
| HANDSHAKE | Learning task; missing grasp, contact and partner-consent evidence block execution |
| APPROACH / PAT_HEAD | Learning task with explicit missing geometry/contact/navigation requirements |
| NEW_* | Record a proposed capability requiring an adapter, evaluator and learning definition |

WALK_IN_PLACE is the executable learning vertical slice: configured CC0 glTF →
canonical retargeting → existing periodic imitation trainer → held-out
interpolation checks and bounded-target validation → policy registration →
resume the requesting plan. Training runs in a bounded child process, cancels at
shutdown and promotes at a step boundary. It does not block real-time actuation.
Repeated missing-capability requests share a task instead of spawning repeated
training. Failed jobs remain visible; changing a dataset/retrying a failed job
currently requires operator management of the persisted task.

Reference-free RL tasks identify the existing task environment/reward boundary;
VRChat-replay refinement identifies the existing REACH trainer. These routes
remain blocked until task-specific validated input and evaluation are configured.
They are not automatically launched from an arbitrary LLM request. New named
skills do not instantly acquire motor implementations.

## Evidence limits

Capability availability means an executable adapter exists. It does not establish
mastery. Success counts retain their evidence scope: simulation, device execution,
elapsed waiting, or unknown. Virtual tracker feedback is not an avatar/contact
sensor. Plan completion means observed steps completed, not that an unconstrained
natural-language goal was semantically achieved. `success_description` is retained
for judgment and auditing; it does not generate a reward function automatically.

Training evaluation is interpolation within the same motion clip, not general
walking, balance or navigation. Target familiarity currently counts observed
object IDs within a session and is not reliable player identity. Multiple steps
execute sequentially; simultaneous whole-body task arbitration remains future work.
