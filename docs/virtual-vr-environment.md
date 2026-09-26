# Virtual VR environment

Current direction: use an RDP-independent Windows console display before resuming
live rendering validation. The application layer is implemented as described in
[running the slice](running.md). Use official VMT hands plus a separate existing
Virtual HMD for head/rendering. Do not execute the historical custom-driver build
and registration procedures below. Preserve Windows protection settings, unrelated
GPU/VM workloads and a recovery route before switching sessions. Machine setup,
backups, distribution hashes and gate results belong in the repository's parent.
RDP-only timing/render observations must be re-evaluated on the console display.

This document records the current development direction for running MyuMIQ as a VRChat player with an AI-controlled virtual body while keeping the normal VRChat / SteamVR environment easy to restore.

> Status: design and feasibility guidance. Do not treat the Virtual HMD/controller path as validated until it has been tested on the target machine.

## Goal

The preferred long-term body-output path is:

```text
MyuMIQ / PAMIQ
    ↓
Virtual Body
    ↓
Motor policies
    ↓
SteamVR device backend
    ├─ Virtual HMD
    ├─ Virtual Left Controller
    └─ Virtual Right Controller
    ↓
SteamVR
    ↓
VRChat PC in VR mode
```

This allows VRChat's normal VR IK/input path to move the avatar's head and hands instead of trying to drive every body function through desktop OSC.

OSC remains useful for avatar parameters, chatbox integration, auxiliary inputs, debugging, and fallback root control.

## Why SteamVR

VRChat's SteamVR Input 2.0 path supports controller input and skeletal hand input. SteamVR controller support, including virtual controllers, is supplied by drivers.

References:

- VRChat SteamVR Input 2.0: https://docs.vrchat.com/docs/steamvr-input-20
- Valve OpenVR driver documentation: https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md
- Valve OpenVR sample drivers: https://github.com/ValveSoftware/openvr/tree/master/samples/drivers

MyuMIQ should therefore treat the SteamVR driver as a backend for the canonical Virtual Body, not as the body model itself.

## Keep normal VRChat use isolated

The development environment should be easy to enable and disable.

Recommended separation:

```text
Normal use
  SteamVR
    MyuMIQ driver disabled / unregistered
  VRChat
    default profile or Desktop mode

MyuMIQ use
  SteamVR
    MyuMIQ virtual-device driver enabled / registered
  VRChat
    separate profile
    VR mode through SteamVR
```

VRChat officially supports `--profile=X`, where profile 0 is the default, and `--no-vr` to force Desktop mode.

Reference:

- VRChat launch options: https://docs.vrchat.com/docs/launch-options

A reasonable development convention is:

```text
normal VRChat:  --profile=0 --no-vr
MyuMIQ VRChat:  --profile=1
```

The exact launch command should be confirmed against the installed Steam/VRChat environment before automating it.

Profile 1 is a convention, not a required MyuMIQ identity. A launcher must make
the VRChat profile explicit and record the selected number. Compare a failed
test launch with the user's successful normal launch before concluding that
manual authentication is required: forcing a different profile changes the
client's login state. Do not copy credentials between profiles or silently switch
to another account. When the user chooses their usual profile, verify the actual
authenticated account and permitted Home before enabling voice/body control.
Driver/settings restoration does not imply isolation of VRChat's own profile.
Prefer a bounded normal client exit before falling back to forced termination;
record the exit mode separately from restoration results.

### Windows user separation

If profile and driver toggling prove insufficient, a dedicated Windows user is the next isolation level.

OpenVR's driver registry is stored under the current user's local application data, so a separate Windows user can also separate much of the OpenVR driver configuration.

Do not introduce a second Windows user unless testing shows it is useful; start with VRChat profiles plus explicit driver registration/removal.

## Driver registration

SteamVR ships `vrpathreg.exe` for registering external OpenVR drivers.

Typical development operations are conceptually:

```powershell
vrpathreg.exe finddriver <driver-name>
vrpathreg.exe adddriver <absolute-driver-directory>
vrpathreg.exe removedriver <absolute-driver-directory>
```

Use `vrpathreg.exe help` from the installed SteamVR version as the source of truth for exact current syntax.

Valve explicitly warns against registering the same driver more than once. Duplicate registrations can produce undefined selection behavior.

References:

- https://github.com/ValveSoftware/openvr/wiki/Local-Driver-Registration
- https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md

Development scripts should therefore be idempotent:

```text
start-myumiq-vr.ps1
  1. locate SteamVR
  2. finddriver MyuMIQ
  3. register driver only if needed
  4. start SteamVR
  5. verify virtual HMD/controllers are visible
  6. start VRChat using the MyuMIQ profile
  7. start the MyuMIQ runtime

stop-myumiq-vr.ps1
  1. stop MyuMIQ runtime
  2. close VRChat
  3. close SteamVR
  4. optionally unregister/disable the MyuMIQ driver
```

Do not implement destructive cleanup that removes unrelated third-party SteamVR drivers.

## Lightweight Virtual HMD

MyuMIQ does not need human-quality VR rendering. VRChat only needs to run through a valid VR path while MyuMIQ receives enough visual information for perception.

The Virtual HMD driver should make render resolution and refresh rate configurable and deliberately small during experiments.

Suggested benchmark matrix, not fixed defaults:

```text
render size per eye: 256, 320, 512, 768 px class
refresh:              30 and 60 Hz
```

Do not assume the lowest settings are automatically valid. Measure:

- whether SteamVR remains stable;
- whether VRChat stays in VR mode;
- GPU utilization;
- VRAM usage;
- VRChat frame time;
- captured image quality;
- downstream perception quality.

The body-control loop can run at a higher rate than rendering. For example, controller/body targets may update at 30-60 Hz while visual encoders run at only 5-10 Hz.

Avoid trying to remove rendering entirely in the first implementation. MyuMIQ still needs visual perception, and a minimally rendered VRChat view is a more useful first target than an unsupported fully headless path.

## Primary and fallback backends

The architecture should support at least the following conceptual backends:

```text
SteamVRVirtualDeviceBackend   # preferred embodied backend
VRChatOSCBackend              # fallback / auxiliary control
MockBodyBackend               # tests
SimulatorBackend              # future training environment
```

The Virtual Body must not depend on SteamVR coordinate conventions or VRChat OSC addresses. Convert coordinate systems at the backend boundary.

## Initial feasibility spike

Before building RL, LLM integration, mocap imitation, or a large body framework, validate this vertical slice:

1. Install / locate SteamVR.
2. Build the smallest valid OpenVR driver.
3. Register one Virtual HMD.
4. Register Virtual Left and Right Controllers.
5. Confirm SteamVR sees all three devices as connected.
6. Start VRChat PC through SteamVR in VR mode.
7. Move head/controller poses from a tiny test program.
8. Confirm the avatar's head and hands follow them.
9. Test basic controller input such as trigger/grip if available.
10. Optionally test skeletal hand input after pose control works.
11. Measure VRAM, GPU utilization and stability at several deliberately low render resolutions / refresh rates.
12. Confirm that stopping MyuMIQ and unregistering/disabling its driver restores normal SteamVR/VRChat use.

If the experiment fails, record exactly where it fails. Do not compensate by coupling Virtual Body directly to one workaround.

## Accounts and profiles

A separate VRChat profile can reduce configuration/login collisions, but profile separation is not a substitute for checking current VRChat platform rules.

This project should not assume that creating a second account automatically makes autonomous-agent operation permitted. Before public or unattended operation, check the current VRChat Terms of Service and platform rules and document any relevant constraints.

Do not embed credentials, session tokens, passwords, or account secrets in the repository.

## Target development machine

Initial target:

```text
Windows
NVIDIA RTX 2080 Ti 11 GB
SteamVR
VRChat PC
Virtual HMD + Virtual L/R Controllers
local PAMIQ / LLM / perception where practical
```

The practical goal is not maximum VR fidelity. It is enough rendering for perception plus reliable body input, while leaving GPU headroom for local models.

## Open questions

The first experiments should answer:

- Can VRChat run reliably with only the MyuMIQ virtual HMD/controllers and no physical HMD?
- What is the minimum stable Virtual HMD render size and refresh rate?
- What VRAM/GPU cost remains after aggressive quality reduction?
- Can controller pose updates remain smooth while rendering at a lower rate?
- Which controller profile gives the cleanest VRChat input and skeletal-hand behavior?
- What observations can be recovered as reliable virtual proprioception versus estimates?
- Does the MyuMIQ driver coexist cleanly with any physical VR hardware installed later?

Treat these as measurements, not assumptions.
