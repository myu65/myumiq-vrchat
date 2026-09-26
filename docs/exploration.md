# Private Home exploration

`EXPLORE_HOME` adds bounded mapless locomotion to purpose mode. This is distinct
from `EXPLORE` (look at visible objects) and `WALK_IN_PLACE` (stationary imitation).
It does not make `APPROACH` or obstacle-distance sensing available.

## Configuration and operation

Add `"exploration": {"enabled": true}` to the outside-repository autonomous JSON.
Purpose mode and live vision are required. The host must produce `startup.json`
in the body session, with `private_home: true` and the current `vrchat_log` path.
The runtime verifies the latest joining and room log entries before granting
movement; an unknown/non-private/non-Home world is blocked.

For an explicitly authorized owned friends Home, additionally set
`allow_controlled_home: true`. The launcher must write `controlled_home: true`
and the exact verified instance string to `startup.json`. The current log must
continue to match that instance. The option is false by default; merely naming a
world Home does not enable it. Account identifiers belong in the host launcher,
not in repository configuration defaults.

The LLM can select EXPLORE_HOME through the normal capability/plan interface.
To give it an explicit initial exploration purpose while autonomy is enabled:

```text
python -m myumiq_vrchat.body_console send --session SESSION "{\"kind\":\"explore\"}"
```

Pass the JSON using your shell's quoting rules. `manual` and `stop` retain their
existing meanings. The optional host launcher may expose `StartExplore`/`Explore`.

## Execution and evidence

Normal operation uses `exploration.mode="continuous"`. The finite body goal
continuously renews 150 ms controller leases, without timed pulse/settle gaps.
Image-response checks run separately once per second. Official VMT's
Index-compatible profile maps thumbstick to
joystick channel **1**; channel 0 is the trackpad. No driver rebuild is required.
The local controller samples at `controller_hz` (default 30) on the existing motor
clock, with `acceleration=.9`, `deceleration=1.8`, `continuous_strength=.45` for
advance and `turn_strength=.85`. Values are normalized controller demand, not
measured metres or radians per second. Smooth/snap turning depends on the VRChat
binding; the runtime does not change that binding. For explicit operator tests,
`mode="pulse_test"` retains the old 350 ms pulse and roughly one second settle.
Stale leases/frames, explicit stop, manual mode and shutdown release input
immediately. Ordinary speech and pending high-level inference preserve the goal.

With an articulated motor, the skill also requires its learned WALK_IN_PLACE
reference. The same actor generates all body joints while controller leases
operate the separate locomotion input. Inputs remain neutral during initial pose
fitting, then use the same LocomotionState that scales the gait reference speed.
Waiting for each actor step's readback does not cancel otherwise valid navigation.
The gait reference still advances only on confirmed feedback: short actor horizons
and motion-buffer interpolation remain the next stage. Tracker root positions
are not updated from a guessed walking speed.

A low-resolution scene descriptor compares successive observation windows.
Continuous exploration alternates scanning and advancing without mandatory stops;
explicit directional goals retain their requested direction. The legacy pulse
test also keeps 64 visited views and uses novelty to choose turns. Three unchanged outcomes
disable further movement for that session. Records appear as `exploration_outcome`
in `goals.jsonl`, shared episodic/working memory, and exploration metadata on
PAMIQ body transitions. Model decisions receive this same memory.

Image change is **not** measured world displacement: animation, mirror content,
lighting and UI can also change pixels. This version has no metric map, depth,
collision sensor or guaranteed obstacle avoidance. It never reports metres
travelled, room coverage, contact success or successful person approach.

## Verification

Unit coverage includes the actual thumbstick mapping, automatic input release,
speech continuity, explicit/manual stop, stale images, no-response stop, and
private-world changes. Deployment must additionally observe actual VRChat turning
and displacement with the configured binding, rather than equating accepted OSC
packets or tracking-space poses with world movement. Live evidence belongs outside
the repository.
