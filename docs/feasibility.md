# Virtual HMD/controller feasibility spike

## Scope and validation

Application implementation and controlled validation have resumed; the historical
custom-driver build/registration commands below remain suspended. Use the
[supervised VMT output](vmt-backend.md) and [application runbook](running.md).
Console display readiness precedes any live rendering trial. Keep VRChat launch/security,
Null HMD rendering and VMT input as separate gates. Do not disable SAC, introduce
exceptions/bypasses, replace DLLs, force an older VRChat version or resume custom
driver development. Machine-specific blocker evidence remains in the parent memo.

The current spike evaluates the official VMT v0.15 binary distribution. Custom OpenVR driver development is suspended; the native driver, diagnostic probes and offline tests below are historical artifacts. Neither a successful build nor runtime input readback establishes VRChat avatar tracking or rendering quality. Validate the live gates below before claiming feasibility.

Keep machine-specific installation paths, hardware inventories, test outcomes, logs and measurements outside the repository, in a local memo and artifact directory in its parent directory. Repository documentation contains reusable design and procedures only.

## Historical custom-driver build (suspended)

From the repository root in PowerShell, with Git, CMake and Visual Studio C++ tools installed:

```powershell
git clone https://github.com/ValveSoftware/openvr.git .cache/openvr
git -C .cache/openvr checkout 0924064316de3effbcd1acf1e309182a2deb1c05
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure --no-tests=error
```

Check each exit code before continuing. Alternatively set `-DOPENVR_SDK=<absolute SDK path>`. SDK code stays outside tracked application files. The package is `build/myumiq`, with `bin/win64/driver_myumiq.dll`. The probe is `build/Release/pose_probe.exe`; its OpenVR DLL is staged alongside it.

`pose_probe --list` also prints VSync timing availability and raw/standing tracking validity. Keep software timing, runtime pose validity, rendered pose validity and observed avatar movement as separate gates. A windowed nonphysical display can expose valid tracking while rendering remains incorrect.

## VMT feasibility (current direction)

Use the [official v0.15 installer](https://github.com/gpsnmeajp/VirtualMotionTracker/releases/tag/v0.15),
not a source build. Preserve security settings and record installer, manager and driver
load outcomes separately. Retain local hashes, original runtime settings and exact
registration paths outside the repository. The application now has OSC transport,
supervision, local LLM cognition, procedural motor control and PAMIQ replay; those
software components do not themselves establish rendering or avatar feasibility.

The [VMT API](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api/) exposes OSC
pose, button/axis and skeletal inputs. Use explicitly owned indices such as 1/2
as left/right Index-compatible controllers (modes 5/6). Head pose uses a separate
VirtualHMD_OpenVR endpoint; do not create a VMT head override in the current path.
These are VMT indices, not SteamVR device indices. Device type is fixed at first
registration during a SteamVR run. Use localhost UDP and neutralize inputs on exit.

Historical Null configuration: VMT Manager's Null Driver option selects the supplied `default.vrsettings_nullhmd`
configuration, enabling **SteamVR's bundled null driver**. It does not turn a VMT
controller into an HMD. The documented `TrackingOverrides` entry maps the VMT head
tracker to `/user/head`; only head pose needs overriding, since hands are actual
VMT controllers. Confirm HMD pose readback and VRChat camera behavior independently.

Verify left/right roles, pose movement, digital press/release, analog axes and all
finger bones through SteamVR before testing the VRChat avatar. A skeletal-input
readback is evidence of runtime input, not avatar IK. Use VRChat's installed Index
bindings, the dedicated profile and a private home. Compare explicit fixed head,
hand and finger poses visually. Adoption as the standard VMTBackend requires these
VRChat checks; recording accepted OSC packets alone is insufficient.

Include a producer-silence test. Do not assume that stopping OSC automatically
releases VMT buttons/axes or invalidates its poses. The implemented adapter sends explicit
neutral values for every input it drives, with a separate liveness supervisor.
`/VMT/Reset` turns off VMT devices globally and is rejected by the shared-instance
adapter. Test scoped recovery and the separate HMD safe pose. These are adapter requirements, not grounds
for modifying or rebuilding VMT.

For an invalid VR view, compare system tracking, compositor render/game poses and
the actual camera separately. Record whether the desktop is active and whether
the test runs through Remote Desktop. Reproduce an invalid render pose in a
separate scene client before attributing it to VRChat. Compare the bundled Null
HMD with and without the tracking override; failed compositor startup is an
unmeasured comparison, not a pose result. Keep local evidence in the parent memo.

## Historical custom-driver live test sequence (suspended)

For a load failure, run `./scripts/diagnose-driver.ps1` and redirect its JSON output to the parent local artifact directory. It reads the DLL hash, Authenticode status and recent Code Integrity enforcement events at that path without loading the DLL or changing policy. Historical path matches can refer to older builds; missing events are not proof of permission. If execution is blocked, use a signing process or development environment approved for that machine before retrying. For Smart App Control, Microsoft documents [trusted-provider RSA code signing](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/code-signing-for-smart-app-control); a self-signed certificate alone does not meet that requirement.

1. Install SteamVR via Steam and locate its directory. Close VRChat and SteamVR. Record the initial `vrpathreg.exe show` output and available normal hardware. Do not enable a competing physical HMD during this experiment.
2. Run `<SteamVR>/bin/win64/vrpathreg.exe help` and confirm finddriver/adddriver/removedriver syntax against the script. Register with `./scripts/driver-registration.ps1 -Action Register -SteamVR '<SteamVR directory>'`. Running Register again should be a no-op. The script refuses another MyuMIQ path and never removes by driver name.
3. Start SteamVR explicitly. Check the log for MyuMIQ activation and three devices. Run `./build/Release/pose_probe.exe --list`. A missing device, blocked add-on, compositor failure or invalid tracking must be recorded as a failure at this gate.
4. Run `./build/Release/pose_probe.exe --wave`. It sends targets at nominal 60 Hz over ten seconds, with head yaw and right-hand vertical motion. An optional integer duration (`--wave 120`, range 5–600 seconds) supports longer observations. It reads raw tracking poses back from SteamVR, requires at least 90% valid samples for every device, head rotation matrix-element span above 0.1, and right-hand vertical span above 0.2 m. After sending stops it checks that tracking becomes invalid. It releases all controller inputs throughout. Device discovery does not require valid tracking, so the producer can recover after the timeout. Repeat as needed during later gates.
5. Launch the installed VRChat **SteamVR VR option**, using `--profile=1`, with no `--no-vr`. Confirm the launch choice and profile against the installed client. If login is awkward in VR, first log in manually using the same profile with `--no-vr`, close VRChat and reopen in VR mode. Use a controlled private test session. The tool does not automate login, conversations or public unattended operation.
6. Observe head yaw, left/right assignment and right-hand movement on an avatar mirror; record capture and any hand calibration offset. Accepted targets alone do not pass this test. The driver ships a minimal VRChat binding for both hand poses, trigger and grip in global/one-hand/menu/action-menu sets. Run `scripts/check-bindings.ps1 -ActionManifest '<VRChat>/VRChat_Data/StreamingAssets/SteamVR/actions.json'` against the installed client. Confirm SteamVR logs load this default binding without errors; existing user bindings take precedence. Locomotion, menu toggles and skeletal input are not supplied by this minimal binding.
7. Stop the probe; confirm tracking becomes invalid after 500 ms and input values return to zero. Restart it and confirm recovery. Native unit tests cover stale pose state; runtime input release needs separate checking.
8. Close VRChat and SteamVR, then run `./scripts/driver-registration.ps1 -Action Unregister -SteamVR '<SteamVR directory>'` twice. Restart the normal SteamVR/VRChat session and verify the original setup works. Save before/after driver lists; unrelated registrations must match.

If the windowed display fails to start VRCompositor, record its exact error and logs. Do not describe this as headless VR or silently switch to OSC; investigate display timing/direct-mode requirements as the next spike.

## Render and GPU measurements

The starting 512 px per eye / 60 Hz setting is an experiment, not a minimum or optimum. Edit only the staged `build/myumiq/resources/settings/default.vrsettings` while SteamVR is closed; vary `render_size` over 256, 320, 512, 768 and `refresh_hz` over 30 and 60. A fresh CMake configure re-stages source defaults. SteamVR user overrides and resolution scaling may supersede recommendations: record **actual** render dimensions and cadence from the runtime.

```powershell
./scripts/measure-gpu.ps1 -Seconds 60 -RunName baseline
# During a live run with fixed avatar/world/camera/quality:
./scripts/measure-gpu.ps1 -Seconds 60 -RunName vr-512-60
./build/Release/frame_probe.exe 60
```

These are device-wide GPU utilization and memory samples, not process attribution or exact allocation peaks. Report maximum *sampled* VRAM and sampling interval, use a comparable baseline, and list other GPU workloads. Warm up each run, then capture at least 60 seconds and repeat stable candidates. Use SteamVR frame timing for application/compositor frame time and dropped frames. Save a screenshot at each resolution and assess avatar/obstacle/text visibility; perception-model quality is deferred until a model is present.

`frame_probe` reports the active scene PID and uses counter differences over the requested interval (1–600 seconds), rejecting a changed scene or reset/nonadvancing frame counters. It prints submitted FPS, presents, dropped/reprojected counts, and average application/compositor CPU/GPU times. Counter semantics follow OpenVR; presents include reprojection and are not unique rendered frames. Its render/game pose diagnostics identify indices, not avatar observations. Run GPU sampling and frame timing concurrently, and verify the PID belongs to the intended application. Redirect results to the parent local artifact directory, not tracked docs.

| Run | Requested eye px / Hz | Actual eye px / Hz | GPU mean / max | VRAM sampled max / baseline | Frame time / drops | Head/hands | Visual quality | Stability / duration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Run identifier | Requested | Measured | Measured | Measured | Measured | Observed | Assessed | Observed |

Copy this table into the local memo for each run. Record UTC time, GPU/driver, SteamVR/VRChat versions, SDK revision, settings, world/avatar, logs, errors and restoration result there. The GPU script defaults to `../myumiq-vrchat.local-artifacts` relative to the repository root. A CSV from an idle machine does not establish rendering cost. No feasibility pass is claimed until all live gates including restoration succeed.

## Protocol and limitations

Requests are `v1 x y z qw qx qy qz trigger grip` in the canonical frame described in [architecture.md](architecture.md). Positions are bounded to ±10 m for the spike, quaternion norm squared must be within 0.01 of one, inputs are in [0,1], trailing tokens and nonfinite values are rejected. Grip uses a 0.5 threshold. HMD requests reject nonzero controller inputs. Reply `ok` acknowledges acceptance only. The wave mode sends neutral trigger/grip; a production backend remains future work.

For deterministic visual comparison, `pose_probe --targets FILE [seconds]` reads exactly three validated protocol lines (HMD, left, right), holds them at nominal 60 Hz, and checks at least 90% of returned positions match within 5 mm. Use 5–600 seconds and neutral input values for pose-only tests. This mode permits explicit trigger/grip testing in a controlled scene. Pose files and screenshots from a particular avatar/session belong in the parent local artifact directory. Fixed-position matching does not yet check full orientation error; visually inspect head orientation separately.

No skeletal hand input, locomotion policy, learned model, audio, perception, PAMIQ runtime or replay store is implemented. After live feasibility, integrate the smallest goal→motor→backend→BodyState→PAMIQ replay slice.
