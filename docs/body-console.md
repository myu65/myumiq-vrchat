# Supervised full-body operation

Console sessions have a finite duration of 1–2100 seconds. The upper bound allows
30 minutes of autonomous validation after startup and calibration. Per-action
deadlines, output leases and the independent watchdog remain unchanged.

This console controls the existing eleven-point Virtual Body. It uses the
independent output watchdog, OpenVR readback and PAMIQ replay. Audio is optional
and is not started by this console. It is an operator tool, not the autonomous
cognition loop. It does not install/register drivers or select public worlds.

## Installation and offline check

Install the `motion`, `vr` and `dev` extras using the environment procedure in
[running.md](running.md). Keep configurations, models and output outside the repo.

```powershell
& $miPython -m myumiq_vrchat.body_console run --session ../myumiq-runs/body-001 --duration 60
```

Use another terminal to control that session:

```powershell
& $miPython -m myumiq_vrchat.body_console status --session ../myumiq-runs/body-001
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-001 '{"kind":"posture","name":"crouching"}'
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-001 '{"kind":"posture","name":"sitting_floor"}'
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-001 '{"kind":"posture","name":"lying"}'
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-001 '{"kind":"posture","name":"standing"}'
& $miPython -m myumiq_vrchat.body_console stop --session ../myumiq-runs/body-001
```

Every run directory must be new. Commands expire after two seconds in transit;
session identity prevents replay into another run. Stop remains available even
when the heartbeat is stale. Button pulses last 0.05–0.5 seconds and release
without another command. Immutable command files avoid shared-file replacement
races on Windows. Invalid commands fail explicitly; they are not silently coerced.
Directory scans, stop-file checks, command reads and receipts run on an I/O
worker. Disk delays cannot stall body updates or make an expired command fresh.
Receipt backpressure defers additional commands without blocking motion.
Use `--trace-stalls` for an opt-in `stall-trace.log` containing all Python thread
stacks when a frame takes over 350 ms. This does not extend the output lease.

## Live startup

Complete the [device and private-world gates](full-body.md#devices-and-calibration)
using the host-specific launcher. That launcher must own backup/restoration,
assign eight roles, establish fresh transforms and relinquish its neutral output
owner before starting this console. No concurrent output producer is permitted.
VMT role changes can reset the temporary room transform; the host adapter must
reapply the verified transform for its exclusively owned devices.

```powershell
& $miPython -m myumiq_vrchat.body_console run `
  --session ../myumiq-runs/body-live-001 --duration 600 `
  --live-config ../myumiq-config/fullbody.json --hmd-serial '<verified serial>' `
  --policy ../myumiq-models/walk/policy.json `
  --reference-pose ../myumiq-config/calibration-tpose.json
```

`devices_valid=true` means device pose feedback. It does not assert VRChat FBT
calibration, avatar joint accuracy, contact, balance or successful walking.
Perform FBT calibration in VRChat and inspect the actual avatar separately.
The optional reference pose must be a complete canonical BodyTarget, prepared
before startup so dataset loading never blocks the motor loop.

## Calibration controls

Observe the menu and ray cursor before selecting. An off-screen menu or a ray
passing behind its plane does not establish an input binding failure. The native
desktop mouse is not required for body/menu control.

```powershell
# Open or close the left Quick Menu (short B-button pulse).
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"menu","hand":"left"}'
# Aim uses a validated canonical hand pose. Choose it from the current observed UI.
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"aim","hand":"right","pose":{"position":[0.2,-0.2,1.2],"orientation":[1,0,0,0]}}'
# Select only after the ray highlights the intended item.
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"trigger","hand":"right"}'
# Once VRChat displays calibration spheres, adopt the supplied reference pose.
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"reference"}'
# After the pose settles and alignment is inspected, confirm with both triggers.
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"trigger","hand":"both"}'
```

`view` changes only the head pose for supervised inspection. Such a diagnostic
look override is not evidence that a physically constrained pose was learned.
Manual aiming/view overrides are retained in command files; they are not LLM
decisions or RL samples with measured contact rewards.

## Learned motion and shutdown

Train a checkpoint with the [CC0 import procedure](full-body.md#cc0-motion-import-and-imitation).
No downloaded or newly trained checkpoint is automatically promoted.

```powershell
& $miPython -m myumiq_vrchat.body_console send --session ../myumiq-runs/body-live-001 '{"kind":"play"}'
& $miPython -m myumiq_vrchat.body_console stop --session ../myumiq-runs/body-live-001
```

Play executes the loaded periodic policy at quarter speed for at most twenty
seconds, then holds the last observed pose and releases controller inputs.
Manual/autonomous mode changes also hold pose. A posture/reference command
explicitly chooses a new pose. The source Walk_Loop is
in place; this is not world navigation. Stop closes owned output and readback,
saves `experience.jsonl` through PAMIQ, and writes `result.json`. Verify
`error=null`, `cleanup_errors=[]` and the host launcher's restoration result.
Stopping the console alone does not restore host settings; finish its host
wrapper too. A deadline or tracking fault also ends output safely.

## Remaining distinctions

The periodic policy still executes one clip. A separate experimental whole-body
actor can be loaded with `--tracker-policy <candidate-actor.pt>` and invoked with
`learned_pose`: `pose_goal` contains a complete canonical BodyTarget and
`goal_duration_s` is 1–20 seconds. It takes observed full-body state, goal error,
prior action and timestep and emits all 66 tracker rates together. Pose goals
cannot contain controller buttons or locomotion. This is an explicit candidate
test path; it does not replace autonomous motor policies automatically.

Goal changes preserve pose. Expiry/manual override hold the observed pose and
clear prior rates. Actual following observations populate `TrackerLearningStep`
records with a `tracker_geometry` reward and evidence reference. Simulated and
device feedback are labeled separately. Neither is avatar/world task success.

`scripts/train_tracker_actor.py` trains an actuator-space SAC candidate from CC0
pose samples; optional `--motion-prior` adds whole-body motion BC initialization
and alternating updates. Evaluation includes withheld clips and mid-motion goal
switches against pose hold. These candidates have not established natural live
avatar motion or useful real-VR policy improvement; do not equate an exported
checkpoint or demonstration loss with promotion.

An articulated candidate uses `--articulated-policy <candidate-actor.pt>`
instead. If its manifest declares a reference floor, pass the matching
`--articulated-floor <z>` in the calibrated tracking frame. This option requires
exclusive supervised ownership and cannot be combined with `--tracker-policy`
or an autonomous configuration. The model loader does not import SAC or Gym.

The articulated adapter fits latent joints to current tracker observations in
the background while holding the observed pose. It never sends that fitted
reference as a body target. Stale/failed fits keep the pose held; valid fits
allow the learned actor to apply coordinated increments to actual feedback.
Incompatible following feedback invalidates the estimate and starts another
bounded fit. Steps below the declared reference floor are rejected and reported
in `articulated_actor.error`; this guard does not establish learned floor avoidance.
Goal changes preserve state, and expiry/manual control hold the current pose.

Replay records the 290-value observation, latent joint action, physical 66-rate
action, fitting provenance and following tracker feedback. When the stored
joint estimate does not match that feedback, the raw transition remains but its
learning tuple is omitted. A mock console trial validates this path, not avatar
IK or world interaction. Keep training/fitting CPU thread counts bounded in the
launch environment (for example `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

Readback can lag output. The articulated adapter keeps one outstanding increment
and repeats its issued target for at most 350 ms while awaiting confirmation.
It does not advance the actor or revert to stale feedback during that interval.
Learning records wait for matching feedback; cancelled/unconfirmed actions retain
raw observations without a fabricated learning tuple. Holds and background fits
latch the observed pose on entry, avoiding repeated publication of older poses.

For an explicit decision-to-body trial, set `articulated_tasks` in the autonomous
configuration to a separate `ArticulatedTasks` JSON file. Its fields are `version`
(1), `actor` (the checkpoint path), `actor_sha256`, `rig_sha256`, `reference_floor`
and `goals` (named intents mapped to complete neutral-input BodyTarget poses).
The actor and rig hashes must match the loaded export. The current goal keys are
`STAND`, `CROUCH`, `SIT`, `LIE` and explicit `RETURN_TO_REST`; supply only goals
appropriate to the scene and calibrated skeleton. These names do not prove
floor contact, a usable chair or successful avatar posture.

This path requires both `purpose` and a separate `decision` model configuration.
Do not also pass a command-line actor option. At activation the goal is bound to
the current pelvis XY and heading, while height uses the declared floor frame.
The goal remains fixed during that intent. The learned actor drives all eleven
trackers, and no procedural gaze/hand overlays run in this mode. Unsupplied body
skills are unavailable; voice and separately gated controller exploration retain
their existing loops. Arbitrary headings and tracking offsets still require
evaluation with the selected policy.

The console records the actual decision reference, shared-memory context, intent
generation, task goal, estimated joints and next-feedback learning tuple.
Posture outcomes in this route compare all eleven fresh tracker poses against
the actual anchored goal, including rotation; they never compare only pelvis
height against a procedural posture. Evidence records the maximum errors and
the explicit 0.12 m / 0.35 rad tolerances. Missing or stale trackers and mismatched
goal generations remain unknown. Device tracker success does not verify avatar
IK, floor contact or world displacement.
Changing intent, expiry and manual control truncate the old task and preserve
pose. This is an opt-in candidate route, not automatic policy promotion or a
completed set of social gestures and world interactions.

Alternatively, `--components 12` fits an uncentered motion-rate basis from the
training clips and trains one SAC actor in that smaller action space. The exported
actor still emits 66 physical rates. Its manifest contains the fixed decoder;
replay also records the original latent action and decoder ID. The basis has no
pose offset, so zero latent action holds any current pose. `--components` and
`--motion-prior` are currently mutually exclusive. A learned actor may still emit
nonzero actions at its goal; this requires evaluation rather than assuming hold.

Audit the actual exported model before a candidate trial:

```powershell
& $miPython scripts/evaluate_tracker_actor.py `
  --actor ../myumiq-models/tracker/candidate-actor.pt `
  --poses ../myumiq-models/tracker/poses.json `
  --output ../myumiq-models/tracker/export-audit.json
```

Use a new output filename. This reports goal switches at 10, 20, 50 and 100 ms,
twenty-second already-at-goal drift, individual endpoints worse than pose hold,
and tracker-pair distances outside the supplied reference envelope. Failed
rollouts remain in the report. The distance diagnostic does not measure bone
length or establish avatar quality. Neither this audit nor the training command
promotes a candidate automatically.
