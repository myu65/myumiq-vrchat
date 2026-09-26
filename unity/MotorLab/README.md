# MotorLab

Minimal goal-conditioned right-hand REACH environment. This is a kinematic
end-effector experiment, not a humanoid physics model or VRChat IK replica.

1. Create a Unity 2022.3 project outside the checkout and copy `Assets` into it.
2. Build using the Unity CLI with `-batchmode -nographics -quit -projectPath
   <project> -executeMethod MotorLabBuild.Build --motorlab-build-path <exe>`.
3. Start that executable with `-batchmode -nographics --motorlab-ready <new.json>`.
4. In the application Python environment with the `learning` extra installed,
   run `python scripts/train_unity_reach.py --ready <new.json> --output <new-dir>
   --steps 30000`.

The ready file contains a local connection token; keep it outside version control.
The client closes its owned Unity player when finished. Training writes before /
after evaluations on the same held-out seeds, SAC model and replay, PAMIQ JSONL
experience, and an inference-only actor with a contract manifest.

Observations are target minus hand, hand minus shoulder, and velocity (nine
values). Actions are three normalized velocity components, bounded to 0.6 m/s
with acceleration 1.8 m/s² at 30 Hz. Canonical coordinates are forward X, left Y,
up Z. Success requires distance below 2.5 cm and speed below 0.12 m/s.

`--reach-policy <reach-actor.pt>` enables the exported actor in the application.
Validate in mock first and retain normal live gates before VRChat use. Simulation
success does not establish avatar motion or live improvement. The optimizer is
currently standalone SB3 SAC; PAMIQ persists experiences but does not yet schedule
these training updates. Raw policy observations/actions are retained separately
from the final bounded actuation target.

For deployment-interface checks, start a fresh player and run
`python scripts/evaluate_reach_transfer.py --ready <new.json> --policy
<reach-actor.pt> --output <new-result.json>`. This compares the actor through
Virtual Body with Unity at 30 Hz and evaluates a 60 Hz mock loop on the same
held-out goals. It does not connect to VRChat or establish real avatar transfer.

## Updating from recorded experience

New Unity records include the motor contract, task reward type and separate
termination/time-limit flags. `scripts/refine_reach_replay.py` reads these PAMIQ
JSONL records and creates a separate SAC candidate. Supply `--model`,
`--experience`, `--output`, and a bounded `--updates`; optional `--prior-replay`
retains a SAC replay file generated locally by the training script. Replay pickle
input is for trusted locally generated artifacts only.

Legacy records with missing learning semantics, mock samples, and live device
tracking rewards are rejected. No task success or termination is guessed from
those records. A future VRChat collector must supply verified task feedback and
the matching motor contract before its data can be used by this importer.

Updating parameters is not evidence of improved performance. Evaluate every
candidate on the same held-out goals before promotion; additional offline SAC
updates can substantially degrade the policy, even with previous replay retained.
This script never overwrites or automatically activates the baseline. Its learning
loop remains SB3, with PAMIQ experience as input; it is not a PAMIQ Trainer.
