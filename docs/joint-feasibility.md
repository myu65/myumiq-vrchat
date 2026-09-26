# Joint feasibility for the articulated model

The skeleton's fixed bone lengths and tracker speed limits do not prevent a knee,
spine or neck from rotating into an implausible configuration. `ArticulatedRig`
therefore supports a `joint_limits` field, stored with the actor, rather than
asking the dialogue model to reason about individual joint angles.

Each non-root joint has `lower`, `upper` (three radians-valued rotation-vector
coordinates) and `max_angle`. These are relative to the parent's reference frame
in the imported rig, not SteamVR Euler angles or universal anatomical degrees.
The root entry is `null`; all other joints must have limits. The box contains zero
and is intersected with a ball of radius less than pi. Positive and negative limits
can differ. Parent-relative restrictions preserve whole-body yaw, pitch, roll and
translation. A lying body is not rejected merely because it is not upright.

The model's actuator first integrates its proposed local joint rates, maps the
proposal into the configured envelope, and applies a common speed limit to both
joints and output trackers. NumPy execution and differentiable Torch training use
the same representation. Fitting penalizes violations with a small interior margin
and accepts a solution only within the exact envelope and normal per-tracker
position/orientation tolerances. Its temporary optimizer proposals are never sent
to the actuator. This avoids hard-clipped optimizer parameters getting stuck with
zero gradient. It never overwrites the observation with a projected or neutral pose.
An infeasible start fails preparation and holds the actual readback; an impossible
goal cannot authorize an impossible next joint state.

Training interpolation and perturbations are constrained offline. Hidden-twist
augmentation is accepted only if it remains feasible; otherwise the original
sample is retained. Projecting a supposedly equivalent twist would change tracker
positions, so that shortcut is deliberately excluded. Replay records requested
`joint_action`, the common `joint_action_scale`, and `realized_joint_action` after
the constraint. Projection means requested rates times scale alone do not fully
describe the executed local joint transition.

## Preparing and training

Prepare an immutable new corpus outside the repository, from an existing verified
CC0 corpus and its original glTF (including matching buffer hashes):

```powershell
python scripts/prepare_joint_envelope.py --source <original-corpus> --gltf <source.gltf> --output <constrained-corpus>
python scripts/pretrain_articulated_prior.py --source <constrained-corpus> --output <new-prior> --policy-frame root_xy_heading_joint_world_v2 --tracker-rate-loss
```

The builder samples reference motions at 20 Hz, records the source names and
margin (default 0.1 rad), and uses only training clips unless extra reference clips
are explicitly named with `--reference-clips`. Extra reference clips cannot overlap
the held-out clips. A corpus with held-out poses outside the envelope retains those
poses and reports the coverage failure; it does not expand the limits from them.
Evaluation uses feasible starts, retains potentially unreachable goal poses, and
reports how many reference poses could not be used as starts. Check that count as
well as task error; an improvement on the supported subset is not full coverage.

Use `--heading-goal-range 1.5707963267948966` when training a prior that must turn
up to 90 degrees in either direction, and evaluate actual turning separately.
This adds both whole-posture turns and posture changes with new headings to the
offline teacher. Merely rotating start and goal together does not teach turning.
The collection range is persisted, and incompatible cached teacher data is rejected.

For a calibrated custom skeleton, supply reviewed limits in its rig JSON instead
of estimating them from these example motions. Run both training and inference
with that exact rig. The exported manifest carries
`parent_rest_rotation_vector_box_ball_v1` and the rig digest. Changing limits
changes the rig identity and requires a new evaluated candidate/task configuration.
The loader rejects inconsistent contract/digest combinations. Existing manifests
without limits retain their legacy identity and remain explicitly unconstrained;
they do not gain a claim of constrained training just by upgrading the package.

## Evidence and remaining limits

The automated checks cover adversarial repeated commands, asymmetric bounds,
root orientation freedom, exact holds, finite training gradients, NumPy/Torch
agreement, rejection of incompatible measured poses, augmentation and manifest
integrity. Also evaluate the actual posture and motion catalogue before selecting
a candidate, then check its avatar appearance in VRChat. Tracker consistency is
not direct observation of the avatar's skeleton or its IK.

This first constraint is an envelope of supported local orientations. It cannot
prove that combinations of individually allowed angles are human-like, prevent
all self-intersection, or capture pose-dependent joint limits. Those are separate
model/evaluation extensions, not something gravity simulation automatically fixes.
[Akhter and Black's pose-conditioned joint-limit work](https://is.mpg.de/ps/code/poseprior)
specifically models how admissible joint angles depend on the pose. Its code/data
are not bundled here. A whole-body motion prior remains useful alongside explicit
constraints: [Character Controllers using Motion VAEs](https://github.com/electronicarts/character-motion-vaes)
trains task control on a kinematic motion model. This implementation adds an
explicit feasibility envelope to the current joint-rate actor; it is not an
implementation of that VAE or the pose-conditioned prior.
