# Full-body contract and motion learning

See [the body console commands](body-console.md) for supervised startup, posture
control, calibration input, learned playback, status and shutdown.

## Implemented scope and live gate

The application can represent, emit and read back eleven poses. This does **not**
establish that VRChat recognizes eleven trackers or that its avatar reproduces a
pose. That requires a console/private-world session, tracker roles, calibration,
and visual confirmation. Existing three-point configuration and replay still load.

`BodyState` contains timestamped, source-labelled observations. Absent additional
trackers are unavailable, never copied from the requested pose as live feedback.
`BodyGoal` contains tasks and fixed-pose/contact constraints. `BodyTarget` contains
complete actuation poses and controller controls. Its compatibility name is
`ActuationTarget`. Existing serialized `left` and `right` fields remain; canonical
lookup uses `pose_for("left_hand")` / `signal_for("left_hand")` and corresponding
right-hand names. Canonical coordinates remain right-handed metres, forward X,
left Y, up Z, normalized active wxyz quaternions.

Parts: head, chest, pelvis, left/right hand, elbow, knee and foot. Export schemas:

```text
python -m myumiq_vrchat schema body-state
python -m myumiq_vrchat schema body-goal
python -m myumiq_vrchat schema body-target
```

Disjoint tasks can represent walking + looking + holding in one hand + gesturing
with the other. Conflicting effector ownership is rejected. Priority is metadata;
there is no implemented task arbitration solver. The initial learned policy
explicitly rejects multiple tasks and constraints rather than ignoring them.

## Devices and calibration

`LiveConfig.trackers` must contain exactly eight unique extra parts and device
indices distinct from both controllers. Every tracker and both controllers use
the same `vmt_from_stage`; the HMD uses the existing separately calibrated frame.
Each binding also carries `body_from_tracker`, a rigid mount offset. Output sends
VMT mode 7 for extra trackers, modes 5/6 for controllers, and the existing HMD
packet. Stop disables only owned devices or sends their configured safe pose.
The independent output watchdog, deadlines and explicit neutral controls remain.

VRChat's elbow trackers are above the elbow on the **upper arm**. Canonical elbow
position is the joint centre, but the Quaternius profile uses upper-arm orientation.
Calibrate the mount offset accordingly. Do not assume a zero mount offset gives
correct avatar IK. OpenVR readback removes the mount transform before storing the
anatomical pose. Device readback still does not observe avatar joints or contacts.
See [VRChat FBT documentation](https://docs.vrchat.com/docs/full-body-tracking).

Before live output:

1. Restore an operable console desktop; RDP is not the rendering gate.
2. Use the approved VirtualHMD, VMT, dedicated VRChat profile and private Home.
3. Reserve eight unused tracker indices, assign the appropriate SteamVR body roles,
   and validate all eleven device identities. Do not replace unrelated devices.
4. Calibrate one complete safe target, mount offsets, frames and an explicit
   `full_body_envelope` (at most 2.5m). The three-point envelope is insufficient for
   lying down. Never mark `calibrated`/`console_validated` true from mock results.
5. Perform VRChat FBT calibration and inspect feet, knees, pelvis, chest, elbows,
   head and hands in a mirror. Save evidence separately for each pose.
6. Release any previous output owner before the finite experiment below.

`postures.py` supplies fixed-length FK diagnostic targets for standing, crouching,
floor sitting and lying, using one fixed 1.6m body. Standing up is a bounded return
to standing. These are predicted geometric fixtures, not physics-stable actions.
The driver does not guarantee avatar contact, floor clearance, or IK convergence.

## CC0 motion import and imitation

Install the `motion` extra. `motion.py` reads dense float32 glTF LINEAR/STEP skeletal
animations, evaluates parent transforms, then converts the Quaternius Godot rig
to canonical coordinates. It normalizes scale by reference head height and
orientation by the source T-pose. This is a **specific source-rig profile**, not
general FBX ingestion or per-bone avatar morphology retargeting. CUBICSPLINE,
sparse accessors and animated matrix nodes are rejected.

Use the free Standard pack from [Quaternius Universal Animation Library](https://quaternius.com/packs/universalanimationlibrary.html),
licensed CC0. A reproducible glTF-only mirror is
[J-Ponzo, fixed revision](https://github.com/J-Ponzo/gltf-universal-animation-library/tree/e24c23cf2a1323488a3faa226ea7ea21f644b73e).
Keep the `.gltf`, referenced `.bin`, license and provenance outside this checkout.

```text
python scripts/train-whole-body.py <dataset/AnimationLibrary_Godot_Standard.gltf> --license <dataset/LICENSE> --source-url <fixed-source-url> --out <new-run-dir>
python scripts/plot-whole-body.py <run-dir>
python scripts/run-whole-body.py --policy <run-dir/policy.json> --output <new-mock-dir>
python scripts/run-whole-body.py --postures --output <new-posture-mock-dir>
```

Plotting additionally requires matplotlib. Training writes SHA256 provenance,
teacher/learned trajectories, a JSON checkpoint and held-out-time metrics.
`PeriodicImitation` learns Fourier coefficients with ridge behavior cloning.
It is a small phase-conditioned periodic motion model, not SAC or a general neural
whole-body controller. Evaluation holds out every fourth time sample from the
same clip: it measures interpolation, not unseen-motion or avatar transfer.
The default Walk_Loop is in-place; it does not implement world navigation.

`WholeBodyPolicy.step(BodyState, BodyGoal, dt) -> BodyTarget` uses observed poses
to bound each step's translation and rotation, with quarter-speed playback by
default. The current state affects this feedback guard, not the learned phase
regression. It rejects missing feedback, unsupported constraints and expired goals.
The bounded tracking error can differ from the direct model interpolation error.
There is no contact solver or exact bone-length projection after regression.

For approved live evaluation, add `--live-config <full-body-config>` and
`--hmd-serial <serial>` to the runner. It uses the existing `OutputSupervisor`,
requires complete readback, and saves through the PAMIQ `ReplayBuffer`.
Replay schema v2 records BodyGoal, state, action, next state, policy identity,
optional reward and explicit simulated/device-feedback outcome. Schema v1 remains
readable. This bounded diagnostic runner is separate from autonomous cognition;
the purpose executive can now select existing full-body skills and the loaded
imitation policy. That selection does not establish learned physical balance.

## Reference-free RL extension

`floor_sitting_reward(BodyState, contacts)` defines a first geometry component for
体育座り: low pelvis, bent/raised knees, forward knees and nearby feet. Contacts
must be supplied by an actual simulator/observer; absence prevents success.
This is an extension point, not a complete stability/collision/balance reward.

The current direction is coordinated tracker-pose rates plus distinct controller
inputs, not gravity/torque balance training. Zero tracker rates retain current
pose. The existing body loop accepts bounded 66-rate commands and records following
device feedback; a task-conditioned learner and avatar/world task evaluator are
still missing. Contact is evidence for contact-dependent tasks, not a prerequisite
for maintaining a VRChat pose. No live full-body RL improvement is claimed by
these helpers. See [the current learning audit](whole-body-learning-audit.md).

## Management and audio diagnostics

Check Sunshine service state, sunshine.exe, the Web UI at its configured bind
address and port (which may differ from localhost defaults), actual startup/log
timestamps and a fresh client-attempt log first. A stopped service is a separate
blocker from pairing, firewall or capture. After the service is running, diagnose
those remaining layers only when evidence requires it.

Separate Windows-wide remote input failure from VRChat-only input failure.
Whole-body output uses VMT/SteamVR, with no remote-keyboard dependency.

For voice, retain three separate gates: TTS-to-Cable recording level, profile=1
selecting CABLE Output, and VRChat actually recognizing microphone input. A past
successful Cable recording does not establish the current endpoint after an RDP
reconnection. Recheck availability and current VRChat microphone logs, then meter
and partner reception. Do not claim audio delivery from device enumeration alone.

On Windows the TTS playback worker initializes COM before opening WASAPI,
including when capture is already active on another thread, and releases only
the COM initialization it owns. Validate simultaneous capture and playback;
opening each endpoint separately does not establish full-duplex operation.

## Acquiring and executing motion sequences

The articulated task JSON can add a `motions` dictionary alongside its posture
`goals`. Each `WALK_IN_PLACE`, `WAVE`, or `MOTION_<NAME>` entry names a learned `policy` path,
its `sha256`, and `playback_rate` (default 0.25). WAVE additionally declares the
demonstrated `hand`; the opposite hand is unavailable until it has its own adapter.
An optional `execution_timeout_s` (1–20 seconds) sets a measured execution budget
for reference tracking and settling. Preparation keeps its separate deadline.
Finite references finish early after observing the complete sequence and a stable
endpoint; periodic gait remains bounded by the requested duration.
All eleven reference poses pass through the existing articulated actor. A clip
is anchored once to current tracking XY/heading. No joint overlay or automatic
rest pose is applied when the action ends.

For automatic acquisition, configure `purpose.learning.motion_source`,
`motion_license` and `source_url` with the licensed Quaternius Godot glTF corpus.
The existing `clip` field defaults to `Walk_Loop`. Optional `motions` entries
override it per capability:

```json
{
  "acquire_configured_motions": true,
  "motions": {
    "WALK_IN_PLACE": {"clip": "Walk_Loop", "kind": "periodic", "playback_rate": 0.25},
    "MOTION_DANCE": {"clip": "Dance_Loop", "kind": "periodic", "description": "全身で軽く踊る"}
  }
}
```

A configured WAVE lesson must refer to an actual waving demonstration, declare
its hand, and use `kind: "finite"` for a non-looping gesture. The default Standard
corpus does not contain WAVE; do not relabel an unrelated clip as a wave.
The planner can queue a supported missing capability while body selection and
conversation continue. With `acquire_configured_motions: true`, the worker also
discovers unlearned configured lessons without waiting for a conversation or LLM
proposal. Keep it false for proposal-only learning. Generic names must match
`MOTION_[A-Z][A-Z0-9_]{0,39}`. Their configured `description` is included in both
thought-model context and body selection after installation. No executable code
or arbitrary asset location comes from the model's response.
The bounded worker trains the prior, checks withheld
times and workspace/floor bounds, and installs it between actions. The executive
persists its hash and reloads verified artifacts after restart. Restored candidates
do not override an explicitly configured motion at startup. Leave its catalogue
entry absent to restore a previously learned candidate. Repeated proposals
do not create duplicate training jobs. Missing assets and failed evaluations remain
unavailable. This path also works from an installed wheel because the trainer is a
package module, `python -m myumiq_vrchat.motion_training`.

The finite prior uses Gaussian-basis regression and clamps at its learned final
frame; the periodic prior uses Fourier regression. Confirmed fresh feedback paces
phase progress. Sequence outcomes include phase coverage as well as eleven-pose
tracking error. Reaching the first or last posture alone does not verify the
motion. Loop completion requires an observed cycle and movement after entering
the starting pose; its arbitrary stopping phase has no static endpoint requirement.
Finite motions require both coverage and endpoint accuracy. Controller walking
and observed world displacement are separate from
WALK_IN_PLACE. Acquisition currently learns configured examples; online task RL,
unrestricted new-skill invention and gesture semantics require further work.

### Other licensed skeletons

A lesson can override `source`, `license`, `source_url`, and `retarget` individually.
The importer supports local-buffer glTF and GLB with an embedded binary buffer.
For a different skeleton, supply a `RetargetProfile` JSON: destination `rig`,
`reference_root`, `reference_rotations` (local wxyz), `source_nodes` mapping every
destination joint name to a source node index, `source_root`,
`source_reference_clip`, optional `source_reference_time`, a proper 3×3 coordinate
`basis`, and `translation_scale`. Optional `motion_scale` reduces the entire
demonstrated motion around that reference before fitting. Forward kinematics
retains the destination's limb proportions. This is an offline reference transfer,
not live head/hand overlays. Asset hashes and the profile hash enter the report.

For example, the official [RobotExpressive source](https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/RobotExpressive)
contains a `Wave` clip and declares CC0. It needs its own retarget profile;
the Standard corpus's Godot profile cannot interpret that skeleton directly.
Use `python -m myumiq_vrchat.motion_training <asset.glb> --license <LICENSE>
--source-url <url> --clip Wave --kind finite --retarget <profile.json> --out <new-directory>`.
Interpolation accuracy alone does not guarantee that the current motor can
follow every pose: inspect sequence coverage in simulated and device feedback.

### Facing and controller walking

The articulated catalogue's optional `facing: true` enables LOOK_AT as a whole-body
heading reference using fresh, confident image feedback. The actor moves all
joints toward a common rotation of the current posture. Lost feedback ends the
attempt holding the observed pose. Its evidence is horizontal image alignment,
not eye contact or verified vertical gaze.

With `exploration.enabled: true`, EXPLORE_HOME combines the learned WALK_IN_PLACE
reference and short controller leases. Preparation, stale feedback, preemption,
expiry and errors release the controller axes. The root inside tracking space is
not integrated from controller input. A normal gate requires private Home;
`exploration.allow_controlled_home: true` additionally permits startup-verified
owned friends Home. In that case the launcher must record `controlled_home: true`
and the exact `instance` string in `startup.json`. Any later instance change fails
the gate. The repository contains no account identifiers.

Motion references default to `motion_anchor: "session"` in the articulated task
catalogue. All motions share the first motion's tracking-space XY origin, while
using the current heading and retaining translation inside each clip. Repeated
short gait actions therefore do not accumulate endpoint error as room-scale travel.
Use `"current"` explicitly for sequences intended to start relative to the current
tracking position. This changes the goal reference only; completion still holds
the observed posture and does not reset the body.

Device acknowledgement is stricter than task completion. It checks both absolute
precision and progress from the command's starting pose. Sub-resolution commands
hold the latent joint estimate too, so dropped microsteps cannot accumulate into
an impossible fitted skeleton. Unconfirmed transitions are excluded from training.
