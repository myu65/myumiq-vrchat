# Whole-body learning audit — 2026-09-19

User correction: body control must be learned as coordinated whole-body behavior,
not expanded as separate procedural gaze/posture rules. Conversation supplies
shared history; a different decision model combines it with world/body state.

Latest correction: VRChat retains supplied tracker poses. Gravity/torque balance
training is not the primary task. Zero pose-rate action holds the current pose;
controller locomotion has a separate lease. Natural whole-body transitions and
task outcomes still require learning and avatar/world observations. MuJoCo tests
are historical experiments, not a prerequisite or a transferable motor policy.

## Current authoritative implementation

The opt-in articulated route evaluates its actual anchored goal against fresh
observations of all eleven trackers, with position and orientation errors tied
to the same intent generation. It does not reuse the procedural pelvis target
to label learned motion successful. Tracker-goal evidence is separate from
avatar IK, contact and world movement. Confirmed motor feedback also does not
establish that the independent high-level model selected a useful action.

| Path | Actual scope | Missing for whole-body RL |
| --- | --- | --- |
| `unity/MotorLab/Assets/ReachEnvironment.cs` | Three-dimensional right-hand velocity; shoulder-relative kinematics | Articulated body, coupled dynamics, support/contact observations |
| `src/myumiq_vrchat/unity_gym.py` | SAC observation 9, action 3; reach reward and episode boundaries | Whole-body task-conditioned observation/action adapter |
| `scripts/train_unity_reach.py` | SAC against isolated Unity reach environment | Whole-body trainer/evaluation/export path |
| `src/myumiq_vrchat/whole_body.py` | 11 pose outputs; periodic imitation fitted to one motion clip | State/goal-conditioned learned action and feedback adaptation |
| `floor_sitting_reward` in same file | Geometry plus caller-supplied contact success check | Actual task evaluation; missing contact remains unknown, without requiring gravity simulation |
| `src/myumiq_vrchat/replay.py:WholeBodyTransition` | Optional versioned 66-rate learning records, policy observations, reward components/provenance and terminal flags | Actual task evaluator and collector populating learning records |
| `src/myumiq_vrchat/tracker_action.py`, `body_console.py` | Bounded 11-tracker rate integration, current-pose hold and next-tick raw feedback collection | Coordinated learned policy, reward and avatar outcome verification |
| `tracker_policy.py`, `tracker_env.py`, `tracker_training.py`, `tracker_basis.py`, `scripts/train_tracker_actor.py` | One state/goal-conditioned SAC actor over 66 rates or a learned motion-rate basis; optional CC0 motion BC; ideal-actuator geometry training | Consistent goal attainment without posture distortion/hold drift, natural avatar transitions, real-world/task reward refinement |
| `scripts/evaluate_tracker_actor.py` | Exported-actor goal switches, timestep variations, twenty-second hold trials and geometric diagnostics on withheld clips | Avatar IK and real-world outcome measurement |
| `articulated_body.py`, `articulated_env.py`, `articulated_policy.py`, `articulated_training.py` | Source-skeleton joint/root rate actions, fixed limb offsets, coordinated physical speed limits, goal-anchored SAC and optional motion BC | Reliable task learning, live feedback adaptation and autonomous body-controller integration |
| `articulated_fit.py`, `articulated_controller.py`, `articulated_tasks.py` | Bounded asynchronous fit, increment-only actions, feedback invalidation and opt-in named posture tasks from the independent decision loop | Live calibration/avatar validation, spatial/gesture goals and policy promotion; fitted joints remain estimates |
| `articulated_dynamics.py`, `scripts/train_model_based_actor.py` | Differentiable ideal actuator, short-horizon policy optimization and equivalent hidden-joint augmentation | Actual avatar/world task rewards and verified transfer to VRChat |
| `body_console.py --tracker-policy` / `learned_pose` | Explicit candidate execution and next-feedback learning records; goal switches/expiry/manual hold | Autonomous policy promotion and actual VRChat outcome validation |
| `src/myumiq_vrchat/capability_learning.py` | Auto-trains WALK_IN_PLACE imitation only | Whole-body RL task routing and evaluated policy activation |

OpenVR tracking readback confirms issued tracker pose, not avatar/task achievement.
Optimizing that error alone is not learning to interact in a world. Existing
VRChat records cannot silently be relabeled as whole-body RL training episodes.

## Replacement boundary

- Executive supplies task/intent and target evidence, never a list of head/hand
  pose rules. Conversation can change intent while observation and action continue.
- One body policy receives live body state and velocities, previous action, task
  representation, and observed target/environment evidence with validity masks.
- The policy produces coordinated body actions. The actuator adapter enforces
  finite/rate/workspace limits; it does not choose social behavior.
- Training must use the VRChat actuator contract: coordinated tracker pose rates,
  distinct tracking-root and controller movement, and unchanged pose at zero rates.
  Arbitrary independent tracker displacement is insufficient evidence of natural
  whole-body control. No synthetic gravity/support problem is required.
- Record task reward, continuity/action change and constraints separately. Use
  contact evidence only when the task needs it and it is actually observable;
  missing contact or visual task evidence remains unknown.
- Goal change does not reset posture. Episode reset is a training/environment
  operation, distinct from intention completion and inference checkpoint updates.
- Use the existing PAMIQ inference/training/model/buffer boundaries; do not invent
  a second scheduler. Retain watchdog and the official device backend.

## Next concrete implementation gate

Verify the tracker-rate output and next-feedback pairing in the existing live
body loop, then connect a task evaluator with actual avatar/world evidence and a
policy learner. Evaluate changing goals from withheld starting poses against
pose-hold and existing imitation baselines. A deterministic actuator model tests
the interface only; it cannot validate VRChat IK or invent contact/task rewards.
Do not promote pose interpolation or procedural per-skill overlays as whole-body
RL. Training, policy promotion and actual VRChat improvement remain unfinished.

## Articulated candidate procedure

`scripts/train_articulated_actor.py` imports the source skeleton and trains local
joint/root rates through fixed-offset forward kinematics. It preserves the
existing physical tracker speed limits using one common action scale. No gravity
or torque simulation is involved. The root is a tracking-space coordinate;
controller input and observed world travel remain separate.

```powershell
& $miPython scripts/train_articulated_actor.py `
  --gltf ../myumiq-data/AnimationLibrary_Godot_Standard.gltf `
  --license ../myumiq-data/LICENSE `
  --output ../myumiq-models/articulated-candidate `
  --steps 20000 --components 24 --motion-prior --diverse-motion
```

Use a new output directory and an operator-supplied CC0 source. The optional
diverse set adds interaction, pickup, walking, seated conversation, kneeling,
torch idle and dance clips; withheld clips never enter the prior or basis fit.
Walking animation here does not establish world navigation. The actor sees goal
error, current trackers, previous physical rates, timestep and local joint
orientations. It emits latent rates decoded to joint motion and finally physical
tracker rates. Each replay keeps both action representations and their rig/basis
identity, plus explicit simulated joint states.

Inspect `result.json`, the BC-only export and candidate export separately.
The analytic oracle is a reachability diagnostic, never a learned-policy result.
Evaluate inferred joint states as well as original source joints: matching the
eleven trackers does not uniquely identify every internal joint. Foot height
relative to the reference plane is a geometric diagnostic, not contact feedback.
Bone length preservation alone does not guarantee natural or useful behavior.

The export has a distinct contract and uses the explicit console option
`--articulated-policy`, alongside `--articulated-floor` when its manifest requires
one. The adapter fits current observations while holding their pose, applies
increments instead of publishing a fitted reference, and invalidates estimates
that no longer agree with feedback. Its next-observation learning records label
joint state as estimated. Fresh calibration and actual avatar validation remain;
there is no automatic promotion. See `body-console.md` for the trial boundary.

## Differentiable pretraining procedure

The verified ideal actuator also supports short-horizon model-based actor
optimization. This is not a SHAC reproduction or measured VRChat learning.
SAC provides the network/container here; its critic receives no updates.
Record model-based, SAC and BC update counts separately.

```powershell
& $miPython scripts/train_model_based_actor.py `
  --source ../myumiq-models/articulated-corpus `
  --output ../myumiq-models/model-based-candidate `
  --updates 1500 --horizon 4
```

The source must be a verified CC0 corpus exported by the articulated training
script. Train on its training split only. Augmentation includes near-goal starts,
small joint perturbations and alternative hidden-joint configurations producing
the same eleven tracker poses. Goal-anchored zero output holds any current pose.
The optional floor objective concerns the declared source plane only; coefficient
and exponent are persisted, not silently changed for old exports.

`--goal-augmentation` additionally mixes in blended root/local-joint goals from
training poses only. Goal bones remain fixed, quaternions use the same hemisphere,
and sampled goals receive an explicit source-floor clearance. This is not an
inference-time correction. Measure constraint failures separately from endpoint
error; better mean reach accuracy can coexist with worse floor intrusion.

An explicit `--resume-source` restores a compatible actor and optimizer into a
new output directory. `--updates` then counts additional updates. Use
`--floor-weight`, `--floor-power` and `--learning-rate` to specify an objective
change; the result records these settings and the restarted sampling seed.
Evaluate the actual export on withheld poses, fitted latent states, alternative
hidden states and different timesteps. Check foot heights, limb lengths, goal
changes and long holds in addition to mean endpoint error. A floor rejection
in the console is a reported candidate failure, not successful learned behavior.

`--rollout-start-steps` optionally samples the current policy's visited states
before differentiating the training horizon. It is bounded to 100 steps, uses
only the training split, and preserves the preceding whole-body rates. Sampling
stops each body at its last valid state before a predicted floor intrusion; it
does not lift feet or patch individual limbs. Zero retains the prior sampler.
The result records the setting. This is an ideal-actuator training experiment,
not a new runtime policy, SAC critic update, or measured live VRChat learning.

For full-body endpoint acceptance, report the maximum position and orientation
error over all eleven trackers, not just their weighted average. Distinguish
endpoint accuracy from a valid trajectory: a later accurate endpoint does not
erase an earlier floor intrusion. Test fixed held-out goals, alternative latent
joint estimates and update intervals; freeze the case set before reading the
new candidate results. Report already-at-goal holds separately from transitions.

## Related unresolved issue

In trial164215, three distinct ASR records produced the exact model reply
`うん、覚えてますね。` with different action outputs. This is generated repetition,
not merely an old waveform replay. Current request path has no application reply
cache. Inspect/reduce historical answer anchoring and evaluate distinct inputs
with the actual model; preserve the user's low priority for wording polish.

## Sources inspected

- https://github.com/MLShukai/pamiq-core
- https://github.com/MLShukai/pamiq-vrchat
- Local files listed above and docs/virtual-vr-environment.md.

## Locomotion channels (user clarification)

Do not collapse controller locomotion and positional/tracking-space locomotion
into foot animation. The training/transfer contract must distinguish:

1. Controller input: dimensionless axis/button requests, profile/bindings, and
   input reference orientation. `exploration.drive_target` currently issues
   these in hand Controls; observed world displacement is a separate measurement.
2. Tracking-space root change: a shared rigid transform of the tracked body,
   distinct from relative joint articulation. HMD movement alone may be lean,
   posture change or translation and does not identify root motion reliably.
3. Articulation: coordinated body/joint motion relative to the tracking root.
4. World pose: independently observed avatar/world displacement with timestamp,
   coordinate frame, origin/recenter epoch and measurement validity. Unavailable
   remains unavailable, never substituted by an input integral or tracker delta.

The environment must compose transforms once and expose controller-only,
positional-only and mixed actions explicitly. It must test mixed zero/nonzero
channels, recenter events and intention changes, including double-translation
regressions. Mode is part of policy context. Do not infer mode from foot motion.
Do not add a second displacement inferred from controller input to the measured
world transform. Reward progress only from the environment's measured outcome.

### Latest recorded evidence
trial-operator-20260919-164215 finished 600.493s / 5485 transitions, error=null,
cleanup_errors=[], restoration confirmed. Offline audit found 0 records with a
non-null task reward, 0 nonzero stick commands and 99 records with a requested
head-position change. These include posture/calibration motion, not established
world walking. World displacement is unavailable. The audit JSON is in the local
trial directory. This is a concrete missing data contract, not evidence of RL.
