# Repeated local and device practice

`python -m myumiq_vrchat.practice_loop --config <external.json> --out <new-external-dir>`
runs a **finite** batch: baseline device probes, isolated PAMIQ condition training,
existing offline regression selection, restored device trials, then observed-start
feedback into the next round. It changes actual actor weights using the existing
model-based condition objective and motion rehearsal, not SAC updates or a new
real-time motor loop. Weights remain fixed throughout each device trial.

Example external configuration (absolute paths, no credentials in the repository):

```json
{
  "tasks": "/local/tasks.json",
  "cases": "/local/cases.json",
  "prior": "/local/checkpoint-directory",
  "reference_corpus": "/local/cc0-corpus",
  "adapter": ["/local/python", "/local/owned-home-adapter.py"],
  "probe_ids": ["train-lower", "train-reach", "heldout-lower"],
  "rounds": 2,
  "updates": 120,
  "learning_rate": 0.00001,
  "reference_fraction": 0.5,
  "reference_anchor_weight": 1.0,
  "training_timeout_s": 700,
  "live_timeout_s": 550,
  "cleanup_timeout_s": 90,
  "total_timeout_s": 3300
}
```

Use the `ConditionCase` format from [human motion practice](human-motion-practice.md).
Cases and live probes must include train and heldout splits. Keep heldout cases frozen. The first
adapter contract supports `tracking` / `current` conditions without target-world
bindings and body requests of 1–20 seconds. Repeated relative lowering needs an
explicit start reset; absolute height goals avoid cumulative lowering. A source
checkpoint directory contains `candidate-actor.pt` and `training-state.pt` and
must match the configured actor hash. Enable `condition_goals` in the trial tasks.

The operator supplies a host adapter using the existing private-Home launcher,
command inbox, single output owner and stop/restoration procedure. It is invoked
as an argument list (no shell), with `--operation run|cleanup --request <json>
--report <json>`. Do not configure an LLM-produced command as this adapter.

`run` receives the task file/hash, actor hash, ordered complete cases and scope.
It launches the controlled session, persists the **exact owned trial identity
before actions**, waits for verified startup, submits the ordered `body_goal`
commands and writes:

```json
{
  "error": null,
  "controlled_home": true,
  "actor_sha256": "64-character hash",
  "session": "/local/fixed-body-console-session",
  "probes": [{"case_id": "train-lower", "generation": 13}]
}
```

`cleanup` must work even after `run` fails or times out. Stop only the saved owned
trial; never follow an unrelated `active.json`. Wait for existing output cleanup
and settings/profile restoration, then write `{"restored": true, "errors": []}`
with paths to the underlying restoration evidence. Failed restoration stops the
batch. Cleanup has its own timeout after the overall deadline. The CLI does not
recover a hard machine/process crash automatically: perform that adapter's
cleanup, inspect its saved state, then start a new output directory. It never
silently replays a partly completed physical trial.

The coordinator re-reads actual `experience.jsonl` records, checks original goal,
actor, generation, raw fresh connected eleven-device observations, and separates
completion from condition accuracy. Irregular position samples are resampled at
50 ms for speed/acceleration/jerk; large gaps are missing evidence, not zero jerk.
Foot displacement and floor extrema include raw samples. A candidate needs both
the original and current offline baseline gates, plus matched nonregressing live
probes from comparable starting poses (within 3 cm / 0.15 rad), before
`selected-tasks.json` changes inside this batch. A rejected offline
candidate is **never** executed; the current model practices again instead.
Normal operator defaults remain unchanged.

Only training probe **start poses** feed the next local model rollout. The
submitted goal remains unchanged; recorded predicted horizons are not relabeled
as observed movement. The loop does not yet train against automatic mirror/video
judgments, avatar contact or human preference. Device kinematics are limited
proxies for natural movement. Actual mirror review and new heldout scenarios are
still needed before claiming human-like behavior or general skill learning.

Inspect `events.jsonl`, `status.json`, `result.json`, frozen inputs and each round's
trainer results, live measurements and cleanup evidence. Finite round/update/time
limits are configurable; no unattended endless model replacement is implied.
