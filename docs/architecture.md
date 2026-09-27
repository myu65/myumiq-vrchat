# Architecture

## Buffered execution and visual continuity (2026-09-27)

The current articulated path uses the existing fitted actor in an optional
`execution_mode=buffered` adapter (the new task-catalogue default). One bounded
worker predicts 400 ms of joint motion at 20 Hz. The existing motor samples the
joint trajectory; constrained joints interpolate in their convex rotation-vector
envelope, preserving skeleton lengths and the initial measured tracker residual.
New horizons join at a shared future state. Delayed inference consumes the old
finite horizon, then holds; it never extrapolates. Goal cancellation invalidates
pending predictions. Fresh device observations must remain within the recently
issued per-device trajectory; lost/stalled feedback fails the action.

Predicted horizons are not confirmed training transitions. The prior `feedback`
mode remains selectable for measured one-step replay acquisition and comparison.
Task completion still requires observed posture/sequence evidence. Learned motion
reference phase is checked against actual readback, independently of predictions.

Perception retains the existing capture process and lifecycle. A 20 Hz vision
worker performs forward/backward Lucas–Kanade tracking while a single bounded
detector job runs at 2 Hz. A delayed detection is seeded on its original image and
replayed through a bounded frame history, never restamped as a current detection.
Lost/ambiguous features withdraw the track. JPEG snapshots are coherent image/world
pairs at 5 Hz; camera freshness, detector age and tracking diagnostics are separate.
OpenCV's [tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
supplies optical flow; no VLM or detector is placed on the motor clock. Existing
[PAMIQ concurrency](https://github.com/MLShukai/pamiq-core) and executive ownership
are retained, with no additional device writer or training scheduler.

Live acceptance must measure actual cadence, Home/controller continuity, autonomous
plans and local audio output. A successful device readback or screenshot change
does not establish remote audibility, collision-free navigation or avatar quality.

`VisionConfig.fast_hz` defaults to 20 (null selects the legacy detector-paced
path); `hz` remains the detector cadence. `ArticulatedTasks.execution_mode` selects
`buffered` or `feedback`. These do not change the configured model/artifact identity.
The existing producer samples the joint trajectory; the separate compositor emits
at 60 Hz. Producer/readback jitter still limits the observed pose update cadence,
so 60 Hz output alone is not a claim of 60 distinct measured poses per second.

Planner `comment` and TALK proposals enter the existing dialogue generation/output owner with
an autonomous source, never a fabricated user utterance. Human speech takes
priority; the default 45-second proactive cooldown and explicit-stop latch apply.
Both cooldown and enablement are in `DialogueSettings`. The speech model receives
the current visual snapshot and recent memory instead of directly voicing the
planner's proposal as an observed fact. Body actions continue independently.
The optional `comment` is outside the body steps, so an observation can be voiced
without replacing exploration/rest with a conversation goal. Artificial drives
separate bodily activity, changed-view observation, and recent speech interaction;
solitary movement does not satisfy social desire or establish another person's presence.

Body selection uses the same cancellable transport as dialogue. New input clears
obsolete inference and its backoff; at most two non-cooperative retired adapters
are retained before applying backpressure. Invalid decisions remain rejected;
contract failures retry after two seconds, while transport failures keep the
configured backoff. A successful finite EXPLORE_HOME interval may be followed by
another exploration decision. One-shot commands remain deduplicated, explicit
stop remains latched, and every movement still needs fresh Home/camera/output leases.

## Multi-rate migration (2026-09-26)

The next slice removes body assessment from dialogue's default critical path.
The existing executive starts speech and body decisions independently, keeps the
current body goal during listening, and cancels only obsolete reply work. Speech
uses sentence chunks from a cancellable HTTP stream; only the executive submits
them through the existing audio owner and records submitted prefixes. Confirmed
input utterances retain their existing independent memory lifetime. Extensions
without streaming/cancellation remain supported with bounded outstanding work.

The existing OutputSupervisor now owns a 60 Hz actuator compositor. Separately
leased pose, locomotion and hand inputs compose into the existing BodyTarget
protocol. Pose refresh cannot renew controls; input expiry neutralizes controls
while a fresh pose remains held. Repeated servo output retains the producer's
pose timestamp, so it cannot bypass the existing watchdog. Legacy complete-frame
producers remain compatible and renew both channels when publishing a frame.

The motor now samples a local 30 Hz locomotion controller on its existing clock.
Continuous navigation refreshes short leases without pulse/settle gaps. Velocity
requests ramp with bounded acceleration/deceleration; lost authority, explicit
stop and stale leases release inputs immediately. The same LocomotionState
controls thumbsticks and gait reference speed; per-step actor readback waiting
does not suppress otherwise valid controller input. The state records confirmed
reference phase, not an independently invented avatar gait measurement.
No tracking-space root translation is integrated from these demands.

The actor horizon and fast tracking slice above implements the subsequent migration.
Semantic vision still follows the configured planner/selection refresh as well as
new events; fully event-driven semantic scheduling remains future work. Controller
root movement remains separate from tracking-space body offsets. Rate targets are
configuration and acceptance criteria, not claims of measured real-time performance.

CPU fixtures first cover stalled body inference, cancelled HTTP requests,
sentence ordering/barge-in, input persistence and goal continuity. Later output
and tracking slices require their own deterministic tests before live acceptance.
Current continuous movement still uses finite goal deadlines and camera freshness;
it is not target following or metric navigation. Smooth versus
snap turning depends on the deployed VRChat binding; normalized yaw input does not
change that setting or establish a physical angular velocity.

## Current direction (2026-09-21)

### Visual conversation and request-grounded action

The [visual/action slice](visual-conversation-actions.md) keeps the existing executive, model workers and motor
owner. A coherent, dated camera/world snapshot is shared by dialogue and body
selection. Image input is explicitly configurable per role; missing/stale images
are reported as unavailable, never described as a current view. Image bytes stay
out of journals. Dialogue receives capability availability, the latest request's
assessment and observed execution results, not just the agent's own prior promises.

Body selection can classify an utterance as an executable request, unsupported,
ambiguous, stop, or ordinary conversation. This is a proposal validated by the
executive, not authority for speech to control the body. Dialogue defaults to
parallel execution without an assessment barrier. The older bounded wait and
neutral-acknowledgement/follow-up route requires explicit
`dialogue.await_body_assessment=true`; ordinary profiles no longer use it.
Execution evidence is never overwritten by later capability estimates or inference failures.
Unresolved or
unsupported requests must not be silently replaced with exploration. An explicit
stop persists until another executable request, while ordinary conversation does
not cancel an ongoing body purpose. Directed short forward/turn actions reuse the
existing Home gate, gait, expiring controller lease and image-response observation;
they do not imply navigation, following, metric displacement, or obstacle sensing.

Target identity remains an anonymous visual track. Low-confidence/old detections
are excluded from reasoning, and ambiguous matching cannot silently adopt a new
person. Vision-enabled selection must ground a selected target in its image.
These contracts are implemented and tested in the corresponding modules; live
avatar quality and arbitrary compound poses remain separate acceptance work.

Voice input health distinguishes historical peak amplitude from the latest block's
peak/RMS, capture interval and time waiting for VAD processing. A past loud input
must not permanently report a currently silent device as carrying signal. These
bounded scalar diagnostics retain no audio and do not change VAD/ASR decisions.

OpenVR readback enumerates device classes/serials once per observation and shares
that fresh inventory across all owned trackers and controllers. It does not cache
identities across frames. Duplicate serials, unexpected role occupants/hints and
invalid tracking retain their existing failure semantics; the selected device's
serial is checked again when creating its signal. This removes repeated full
device scans from a body sample without weakening ownership or freshness.

The existing PAMIQ replay candidate job can optionally halve its proposed actor
parameter update up to three times when the unchanged validation gates reject it.
The first eligible step is exported with all attempted evaluations; a reduced
step resets optimizer moments in both saved checkpoints. No eligible step leaves
the full proposal explicitly rejected. This bounded selection uses the configured
validation sessions, not an independent final test, and never changes live weights.
The training job, CPU budget and executive learning slot remain the same.

The body console's file-command adapter owns directory scans, command reads,
stop-file checks and receipt writes on a separate I/O thread. The motor consumes
a bounded in-memory inbox and stop latch, rechecking the original session and
two-second issue deadline at execution. Delayed disk reads cannot renew an old
command. Receipts are bounded and backpressure defers command consumption;
neither waiting for disk nor publishing stale commands belongs in a motor frame.
Shutdown stops output first and then drains the adapter with a bounded wait.
The controlled-Home log check also runs as one bounded background request. Its
result is dated at request start and expires after 750 ms; stalled disk access
therefore withdraws navigation permission without blocking speech or decisions.

Conversation retains the existing bounded ten-turn working window through both
memory retrieval and speech projection. An interrupted reply does not shorten
that input history; anonymous statements remain session evidence, not attributed
personal facts. Per-turn text bounds and motor-payload exclusion still apply.
The body console optionally records all Python thread stacks when a frame exceeds
350 ms using CPython's native fault-handler watchdog. This diagnostic is opt-in,
does not renew the output lease, and is cancelled before normal cleanup.
Observed whole-body image alignment completes the matching observe-target
commitment, just as the older gaze alignment evidence does. Device movement alone
cannot do so; both paths still require the commitment's target to match.

Body selection supports explicit image input for configured vision-capable local
or remote models. The vision worker encodes a bounded JPEG and publishes it
atomically with the world snapshot; executive/motor threads only read that pair.
Selection may use the image to reject detector false positives, never to invent
target IDs or replace device feedback. Missing-image fallback requires the
existing explicit state-only setting. Capture timestamps and candidate identity
remain unchanged, and semantic image interpretation is not player verification.
Image selection returns a short scene observation and supported target IDs before
its action. The runtime rejects IDs outside the snapshot or a selected target
absent from that evidence. These are model claims, not verified player identity;
the existing fresh device/vision conditions still govern execution.
Cancelled fitting workers retain their resource slot until they end, but their
old preparation deadline cannot fail a replacement goal. The current fit retains
its independent eight-second limit.
Each Windows SAPI reply owns its cancellation event. Starting a later reply
cannot reactivate a cancelled synthesis worker; stale cleanup/errors cannot
replace the current worker's process or failure state. Synthesis failure is
reported explicitly instead of silently producing no speech.
Reference rehearsal also samples bounded states visited by the current candidate,
using the existing non-differentiable rollout-start helper with floor checks.
This exposes late trajectory errors while retaining the recorded starts and
device residuals of the separate experience batch.

The local audio-chat adapter can pin a Qwen3-ASR language and a bounded recognition
context in configuration. Language selection uses the runtime's final-assistant
prefill contract, independently for each finalized utterance; it does not reuse
conversation history or rewrite recognized words into action commands. Automatic
language detection remains the default. A prefill-compatible server may return
either the full language boundary or only its transcript suffix.

Window capture is a configured perception adapter: the desktop route requires
foreground VRChat; the Windows Graphics Capture route targets its unique HWND
without capturing overlying desktop applications. Native callbacks copy only a
bounded latest frame, retaining its monotonic capture timestamp. Slow detection
consumes that snapshot; frozen/closed/minimized capture invalidates perception
instead of restamping an old image. Native WGC runs in a separate owned process:
a driver/library wait holding Python's GIL must not stall voice or body threads.
Only bounded latest-frame messages cross that boundary; stale capture invalidates
perception and the existing service lifecycle restarts it. Shutdown waits briefly,
then terminates only that capture child if its native cleanup does not return.

The implemented offline learning slice consumes confirmed articulated transitions from PAMIQ
replay as observed starting poses, goals, previous rates and measured timesteps.
Cases require matching actor/rig/decoder identity, fresh OpenVR feedback, finite
feasible fitted joints, and preserved tracker residuals. Whole sessions are split
between training and evaluation; adjacent samples cannot cross that boundary.
The existing differentiable actuator refines a candidate using these starts;
this is model-based optimization, not a stochastic SAC update or avatar reward.
PAMIQ's TorchTrainer, data users and training-only model handle optimization and
optimizer persistence. The candidate has no linked inference model: automatic
PAMIQ synchronization cannot change live weights. Held-out replay starts plus
licensed reference regressions must pass before any explicit snapshot adoption.
The CLI produces evaluated candidates; automatic live adoption remains
separate work. Repeated candidate tuning uses validation sessions; a fresh session
is required for the final test. No improvement is claimed from weight changes alone.
Candidate eligibility evaluates static references with the runtime's finite
completion contract: three fresh observations spanning at least 150 ms within
the existing tolerances, then hold until the next goal. It requires every such
trajectory to remain above the declared floor and no loss of completed goals.
The continuous-actor audit remains separately recorded; running an already
completed static goal indefinitely is not the executed policy. Its floor drift
still diagnoses the raw actor and is not described as repaired by goal settling.
Finite and periodic motion references require separate sequence evaluations;
static settling does not establish their safety or successful completion.
Mean endpoint improvement cannot compensate for a reference floor violation or
an increased count of endpoints worse than holding. The replay trainer's bounded
floor penalty weight is configurable; it never changes the output floor gate.
Optional reference rehearsal mixes a quarter of each training batch from the
licensed corpus's training split. These are explicitly synthetic kinematic starts,
including feasible low-clearance root translations, in a separate PAMIQ data user.
They are never counted as confirmed device experiences. The corpus's held-out split
remains evaluation-only; rehearsal aims to prevent loss of existing motion geometry
while learning a small collection of live failures.
The implementation reuses PurposeRunner's single background learning slot for
a configured replay-refinement job. Only operator-configured immutable inputs and
a fixed packaged trainer are executable; language models cannot supply paths or
commands. Its candidate/evaluation state is separate from motion availability and
does not replace the live actor. Shutdown cancels the owned bounded child process.
See [experience learning](experience-learning.md) for the packaged CLI and settings.

Repeated motor failure is evidence for avoiding autonomous retries, not permanent
capability removal. A new input utterance may request one further bounded attempt
of an otherwise executable candidate. The executive tracks attempts by utterance
and candidate independently of the bounded outcome journal; unrelated observation
updates cannot renew this allowance. Selection receives failure evidence and must
use this allowance only for a renewed human request. Capability, fitting, joint,
freshness and output gates still apply. Startup restores no input retry allowance.
Conversation receives a compact projection of shared memory (recent quoted turns,
identity, purpose, short facts and outcome summaries), never full motor conditions,
tracker controls or nested replay evidence. Detailed evidence stays in the journal.
The dialogue worker owns one lazily initialized generation session per run,
reusing its HTTP connection without changing the configured model or provider.
It admits one request at a time and retains the same response/deadline validation.
Shutdown prevents new requests and defers client cleanup until any in-flight
request returns; neither client construction nor inference blocks the executive.
Static posture goals settle on fresh confirmed tracker observations, then retain
that observed pose instead of running the actor until an arbitrary duration ends.
Three distinct samples over at least 150 ms within the existing task tolerances
complete the finite attempt; the deadline remains an upper bound.
Completion retains the validated settling evidence and its capture time. A newer
body snapshot or subsequent perception gap cannot erase the completed outcome;
the evidence belongs to that action generation and is cleared on replacement.
Whole-body facing uses the same completion rule after two distinct fresh centered
images. A brief interval waiting for the next detector result holds the observed
pose without erasing earlier image evidence; a gap between capture timestamps over
750 ms resets the consecutive-image count. Missing/stale images still cannot drive
motion or establish success. The evidence proves horizontal image alignment only.
Finite motion references use the same settling rule after the full reference
sequence has been observed. Periodic gait keeps running to its bounded deadline.
Each reference may declare an execution timeout (at most 20 seconds) for observed
feedback delays; this is selected before activation and never extended in flight.
The selection adapter labels whether a choice answers the latest input, continues
a goal, or responds to an observation. A posture accepted from a new input becomes
a continuing constraint in executive working memory. Autonomous incompatible
whole-body actions cannot replace it; a new explicit body request or manual control
can. This is the first persistent posture slice, not a compound learned body goal.

The next executable slice adds configurable local audio-chat ASR (including
Qwen3-ASR on a separately managed runtime) and configurable faster-whisper GPU
inference. Model construction/warmup runs outside the executive's poll loop, so
voice/vision startup cannot stall body decisions. Inference has bounded input,
response size and HTTP deadlines; cold readiness and steady-state latency are
reported separately. GPU selection belongs to the startup profile, never an
implicit choice of the first GPU. Native input streaming remains a separate
contract from this fast finalized-segment adapter.

Planner proposal validity depends on explicit control changes, accepted user
directives and commitment identity/status, not each finite body action or speech
onset. Body decision results revalidate the selected capability and target against
current observations; completion of a previous action does not invalidate the
whole request. Purpose changes still invalidate it. Unrelated visual identity
changes wake selection but do not discard a pending posture decision; a selected
target must still exist and meet its fresh observation gate at activation.
Completed actions remain suppressed by the
existing utterance/outcome records. These changes do not create a persistent motor
goal or remove observation freshness gates.
Terminal ASR failure/no-speech/overflow releases listening attention for that
input only. A delayed failure from an older utterance cannot release a newer
speaker's active listening state or cancel an existing body purpose.
Exploration camera loss pauses pose output and releases navigation without
cancelling the gait execution. Its original preparation/execution deadlines and
playback identity remain in force; fresh frames resume through observed body
feedback. Waiting for observation is distinct from manual cancellation and does
not count as completion or permit stale movement inputs.
The slow exploration observer also retains its finite attempt across a camera
gap. Missing frames revoke its short controller lease; an interrupted pulse
enters neutral settling and is never replayed on recovery. A fresh frame and
confirmed gait progress are required for the next pulse. Observation loss does
not repeatedly restart the initial observation interval. Permission loss or a
different body goal still resets the observer immediately.

The authoritative target boundaries, causal lifetime rules, coordinate frames,
model/runtime contracts, PAMIQ mapping and staged migration are in
[継続する目的を持つMyuMIQ：課題と移行設計](runtime-redesign.md).
Read that document first. Its implementation table distinguishes the input
continuity slice from the remaining streaming, persistent body-goal and learning
work. The sections below retain implementation detail and the history of earlier
slices; legacy procedural and independent-overlay paths are not the target design.

The first migration preserves finalized speech across later speech onsets while
cancelling obsolete replies. Input IDs and capture intervals accompany recognized
text into shared memory. Bounded queues report overload and protect finalized
input from provisional event traffic. This does not yet provide crash-persistent
audio storage, streaming generation, or a learned compound-goal motor policy.

## Joint feasibility in the whole-body model

An imported skeleton fixes bone lengths but does not by itself restrict joint
orientation. The articulated rig can now carry a versioned local rotation-vector
envelope for every non-root joint. Bounds are expressed in the existing parent's
reference frame, in radians, relative to the imported rest pose: asymmetric XYZ
bounds plus a total-angle bound below pi. The pelvis root remains unrestricted;
turning or lying down is not a joint violation. This first envelope is configurable
and estimated from explicitly selected reference motions, with recorded provenance
and margin. It is a motion-support constraint, not a universal human anatomy model
or a guarantee against self-intersection or incompatible avatar IK.

The same bounded representation belongs to the NumPy actuator, differentiable
training dynamics and tracker-to-joint estimator. The next joint proposal is
projected into the envelope before coordinated tracker speed limiting. Offline
starts/augmentations must be feasible; hidden-twist augmentation is rejected when
it leaves the envelope. Zero output holds any feasible pose. An infeasible measured
pose is never silently changed into a reference posture: constrained fitting must
meet both joint and tracker tolerances or report failure while holding readback.

The actor manifest includes the envelope contract and rig digest. Changing limits
requires a new model/corpus identity and evaluation; an old checkpoint is not
silently relabelled as trained with constraints. PAMIQ transitions retain requested
joint rates, realized joint rates and the existing observed tracker targets.
No new scheduler or output owner is introduced. Offline constraint compliance and
live avatar posture quality remain separate validation gates.
See [the configuration and validation procedure](joint-feasibility.md).

## Cognition and voice adapters (updated 2026-09-20)

See [the cognition/voice design](cognition-voice-adapters.md) for local split,
local STS/Omni, cloud Realtime STS and cloud split conversation profiles.
[Role-specific local/OpenRouter generation and installed adapter selection](model-adapters.md)
are implemented; streaming speech sessions remain proposed. MyuMIQ retains local
world/memory/Goal/Drives, decision and Planner ownership, body safety, Motor and learning.
Decision and Planner inference may use individually configured local or API models. Conversation backends
may propose high-level Intent/Goal/tool calls through a local validation broker.
One profile stays fixed per run, with no runtime model escalation or automatic
backend fallback. Official Jev is optional. The common streaming AudioSink and
proposal/tool broker described in that design are still future work.


## Motor failure diagnosis

The body console freezes each status snapshot on the motor thread and writes it
using one bounded background status writer. Slow filesystem access neither builds
a queue nor renews device heartbeats on the motor's behalf. Closing prevents late
running snapshots from replacing the final stopped state. PAMIQ replay is separate.

Temporary capture occlusion clears the current world/image while retaining the
vision detector in its existing worker. The worker retries at its configured
frame rate and reports pending until a new image arrives; it never captures a
different foreground app. Fatal capture/detector failures still use the service
restart path. This avoids repeated model construction during ordinary focus loss.

The body console records bounded in-memory phase timing for readback, validation,
commands, motor calculation, publication, replay and status writes. Status and
final results expose slow phases with the same monotonic clock as output events.
Failure captures device signals and output-owner health before cleanup. An output
owner that has already halted is reported before consequent device invalidity.
Timing performs no I/O or heartbeat renewal; existing watchdog deadlines and
fail-closed output ownership are unchanged. These diagnostics identify where a
producer stalled; they do not establish an OS/GPU scheduling root cause.

## Bounded private-Home exploration

An optional EXPLORE_HOME capability performs short controller locomotion pulses
followed by neutral-input observation. The LLM selects the purpose/skill, while
a procedural navigation policy selects bounded turn/advance actions. The existing
motor owner and watchdog remain the only device writers. A fresh visual snapshot
and a private-Home gate are required for each pulse; stale sensing, interruptions
and deadlines always release sticks. Scene descriptors retain visited-view novelty
and before/after image change as evidence, not metric position or collision truth.
Unchanged views stop forward retries and trigger observation/turning. Evidence
feeds shared episodic memory and the next high-level decision; PAMIQ records the
actual controller targets. This is mapless exploratory locomotion, not calibrated
navigation to people, contact sensing or guaranteed obstacle avoidance.

## Shared memory and continuing commitments

The purpose worker is the single writer of a shared SQLite memory store outside
the repository. Working context holds the current conversation, focus, commitment
and plan progress. An episode journal carries evidence/source IDs. Retrieved
semantic records retain their episode provenance and distinguish reported speech
from observed motor outcomes. Social records use explicitly associated identities;
visual track IDs and anonymous mixed audio never establish cross-session identity.
Capability conditions aggregate success/failure/unknown separately and retain the
body/target evidence scope. Bounded retrieval supplies the same snapshot to goal
planning, dialogue and the image candidate scorer. PAMIQ remains the motor replay
store; SQLite is application memory, not another motor/training scheduler.

A continuing commitment outlives an individual plan and survives conversation
interruptions. Step completion updates progress but is not semantic completion.
Measurable criteria (fresh centred visual observation, observed posture, recorded
dialogue turns, registered policy) gate completion. Open-ended interests require
reconsideration; an LLM cannot declare unobserved success. Restarts restore memory
and interests for fresh deliberation, never queued actuation or old dialogue replies.

In purpose mode, audio workers perform VAD/ASR and immediate TTS cancellation only.
Transcripts and observations feed one memory owner, with separate bounded pending
requests for dialogue and autonomous planning. Each request uses a detached snapshot;
completion is committed by the executive, never by inference threads. Speech only
invalidates older dialogue generations. It does not cancel autonomous planning or
prevent plan outcome observation and next-step selection. Manual preemption clears
both paths. Explicit whole-body dialogue actions arbitrate ownership through the
existing motor generation and finite action duration. Gaze/gesture overlays remain
disjoint; exploration pauses its motion lease during listening, without suspending
the overall reasoning loop. The motor loop remains independent.
Attention and hand gestures may overlay disjoint effectors of a continuing body
action; explicit priority and existing per-frame bounds resolve ownership.
Targeted root navigation remains unavailable; optional EXPLORE_HOME supplies
bounded mapless controller exploration. Looking and stationary walking with
speech do not imply navigation, partner identity or successful audio delivery.

## Purposes, capabilities and learning

An optional purpose executive sits above finite motor Intents. A local LLM emits
a free-text purpose/reason/success description and bounded capability requests;
it does not emit coordinates, executable code, asset paths or trainer commands.
The capability registry resolves supported requests into existing Intents or
reviewed compositions. Unsupported capabilities create deduplicated LearningTasks
with explicit prerequisites and data/environment blockers. Missing navigation,
contact sensing or partner agreement cannot be supplied by an LLM assertion.

The slow autonomous worker owns goal/plan progression, capability statistics and
persistent learning state. The bounded LLM/decision request is independent of
the dialogue request; manual preemption invalidates both. Independent reranking chooses only among actions
allowed by the current plan, including waiting; it cannot silently replace the
purpose. Outcomes are evaluated from fresh body/world evidence and distinguish
device execution from avatar/contact or social goal success. Unknown evidence
does not count as success. Goal/step/task IDs join existing PAMIQ intent metadata,
while a compact goal/capability history feeds the next LLM request.

Learning uses configured application adapters around existing trainers: CC0
retargeting/PeriodicImitation.fit, Unity reference-free reach SAC, and validated
VRChat REACH replay refinement. Asset locations and budgets are operator config,
never LLM output. The initial automatic path fits a periodic motion and evaluates
held-out clip times before exposing an immutable policy to the motor worker at
an intent boundary. This establishes same-clip imitation, not contact/general
motion mastery. Other routes remain blocked until their required environment or
task-feedback data exists. No new replay scheduler or PAMIQ fork is introduced.
State is persisted atomically outside the repository; interrupted plans are not
automatically replayed on restart. Existing output watchdog and safe pose remain
the sole final actuation authority.

## Candidate Decision Layer

`decision` accepts one immutable image/state snapshot and named action candidates.
Backends score each candidate against that snapshot independently; no candidate
list softmax, listwise prompt or index-dependent score. IDs restore input order.
Native scores retain their semantics (relevance logit versus Jev rubric score),
so values from different backends are not interchangeable confidence estimates.
The image timestamp is retained and expired results cannot actuate the body.

Local Qwen and Nemotron inference runs outside the motor loop and is optional.
The official Jev API currently accepts text state only; image-bearing requests
must fail explicitly unless a caller opts into a labelled state-only condition.
No invented multimodal API or implicit image omission is allowed. A loopback
service allows the autonomous process to use an isolated inference environment.
Results, selected candidate and snapshot age enter existing decision/replay
metadata. Benchmark artifacts/models and credentials remain outside the repo.

## Autonomous whole-body session

The body console remains the sole device owner. An optional executive supplies
finite high-level intents; a slow worker owns LLM requests, vision, voice, drives
and persistent memory. The motor loop never waits for model loading or inference.
It selects existing procedural skills or the learned periodic walking policy,
using measured BodyState. In purpose mode, speech onset invalidates old dialogue
and cancels speech playback; autonomous planning continues. Legacy intent mode
still preempts motion on speech. Optional service failures remain visible as pending/degraded and are
retried; body tracking and watchdog failures still stop the session.

The existing PAMIQ ReplayBuffer records whole-body transitions plus the originating
intent. Observed device statistics enter subsequent deliberation without implying
avatar/contact success. Automatic RL promotion is not part of this mode. Private
Home startup and calibration remain machine-local launcher responsibilities;
calibration commands issued and calibration visually verified are separate states.

## Full-body extension (implemented; live validation per deployment)

Keep the right-handed metre/quaternion frame. BodyState holds observed or
unavailable head, chest, pelvis, hands, elbows, knees and feet. BodyGoal contains
concurrent tasks and constraints with explicit effector ownership. BodyTarget
holds desired actuation, never inferred measurements. Preserve legacy head/hand
records while requiring every owned tracker in full-body output.

Eight additional VMT trackers share the stage transform and explicit anatomical
mounting offsets. Startup, readback, watchdog and shutdown must cover all owned
devices. VRChat FBT calibration and observed avatar poses are separate live gates.

Motion import retains source/license, skeleton mapping, units, rest-pose offsets
and timing. Retargeting produces canonical trajectories. Imitation and future
reference-free geometric/contact RL use BodyState + BodyGoal -> BodyTarget.
The existing Unity hand-only SAC scene is not a whole-body physics environment.
The initial periodic imitation model learns phase-to-pose Fourier coefficients;
BodyState supplies feedback limits, not a learned observation encoder. A bounded
runner uses the existing output supervisor and PAMIQ replay schema v2. See
[full-body contracts, execution and limits](full-body.md). Autonomous cognition
still uses the original skill set; concurrent whole-body execution is not claimed.

The supervised body console uses the same output supervisor and readback adapter.
Commands are immutable, session-bound files with short delivery deadlines, and
button pulses expire independently of later commands. Every start uses a new run
directory; old commands and active leases are never restored. Device feedback,
visual avatar verification and FBT calibration remain distinct evidence fields.
Host startup/role setup stays in the local machine wrapper; the application
console does not register drivers or change SteamVR settings.

Status: the basic LLM-to-avatar backend passed a bounded live private-Home gate on
2026-09-17. Real VRChat capture, model-backed visual detection, Japanese ASR/TTS
components and a replay policy update have been measured independently. The
single-session VRChat microphone round trip remains unverified. Current audio
endpoint availability and remote-session state must be checked independently.

## Current executable slice

The user resumed implementation and controlled live validation. Keep the native
driver suspended and use existing official VMT for hands/input, with a separate
VirtualHMD OpenVR adapter for head pose. Machine setup and evidence stay outside
the repository. The previous freeze and native descriptions below are historical.

The executable application boundaries are:

| Module | Responsibility |
| --- | --- |
| `body` | Immutable typed poses, targets, observed/estimated body state and coordinate conversion; right-handed metres, X forward/Y left/Z up, wxyz active quaternion |
| `cognition` | Local HTTP LLM client, strict structured intent schema, explicit timeout and rejection; the LLM never emits device packets |
| `motor` | Bounded WAIT/WAVE/LOOK_AT/REACH/RETURN_TO_REST policies producing complete targets, independent of LLM latency |
| `backends` | Existing lifecycle guard, complete VMT OSC serialization, separate HMD UDP adapter, mock output and OpenVR readback |
| `runtime` | PAMIQ Agent/Environment adapter and persistent transition buffer; a separate bounded output supervisor keeps polling when cognition or the producer stalls |
| `autonomy` | Slow drive update, bounded interaction memory and one outstanding local-LLM replanning request; never produces motor frames |

High-level cognition runs outside the PAMIQ interaction thread and publishes one
validated current intent. PAMIQ FixedIntervalInteraction drives the body policy.
The output supervisor owns all live actuation sockets and receives complete targets
through local bounded IPC. Heartbeat and target timestamps use the same host's
monotonic clock and retain independent deadlines. The existing VMTBackend lease
contract remains authoritative; stale, malformed or unsupported data cannot revive
expired actuation. Startup, timeout and shutdown neutralize every driven input.

Body feedback identifies its source: OpenVR device pose, procedural estimate or
unavailable. Device pose is not VRChat avatar/root observation. World targets are
explicit observations/fixtures, not inferred from successful packet sends. Intent
validation restricts target references to the current world state. Unsupported
locomotion/contact commands must fail explicitly instead of silently doing nothing.

Transitions carry schema version, time, world/body before and after, intent,
command, actuation target, outcome, latency and observation provenance. Use PAMIQ
DataBuffer/collector/persistence hooks for replay, with a small JSON export for
review. Do not persist live leases or pressed input. Mock, loopback OSC and crash
tests must run without VRChat; real avatar/console gates are reported separately.

Upstream rechecked on 2026-09-16: pamiq-core 0.6.0, revision
`265e2ca806819c410aa8b15cf921f8b59182e736`; pamiq-vrchat revision
`580d5fa891efd70b54c90edc49a315bbf0b1a102`. Use upstream as a dependency rather
than creating another interaction, replay or training framework.

## Implemented PAMIQ mapping

| Responsibility | Implementation |
| --- | --- |
| Body/world observation and actuation | `BodyEnvironment.observe/affect`; mock sensor or `OpenVRReadback` |
| Intention consumption and procedural motor | `BodyAgent.step`; asynchronous HTTP decision future |
| Body cadence | `FixedIntervalInteraction.with_sleep_adjustor` |
| Independent retained-input safety | `OutputSupervisor` process wrapping the existing `VMTBackend` guard |
| Experience collection | PAMIQ `DataCollector` / `DataUser` with `ReplayBuffer(DataBuffer)` wrapping `SequentialBuffer[str]` |
| Session lifecycle / checkpointing | PAMIQ `launch`, `LaunchConfig`, `StateStore` and buffer save/load hooks |
| Learning | Replay-trained per-skill residual candidate with chronological holdout and explicit promotion gate; a larger goal-conditioned policy/trainer is still pending |

`--autonomous` requests a new validated finite intent after the prior intent
expires. Its initial drives are curiosity, boredom, social desire and fatigue.
Only explicit identified interactions may update player familiarity. Proprioceptive
reward is the bounded reduction in head/hand target error, including head angle.
It does not claim avatar, task or social success. The reward is stored with the
transition and fed into the next deliberation summary.

`perception.TemporalWorldTracker` converts replaceable spatial detections into
time-stamped WorldObjects with smoothed position, velocity, confidence and expiry.
No encoder is implied by this boundary.

The current live detector is a customized YOLOv8s-World-v2 vocabulary exported
to ONNX (`VRChat avatar`, `humanoid robot`, `anime character`, `person`). It is a
low-rate proposal source rather than a motor-loop dependency. On the saved private
Home frame it found the mirror avatar; direct ONNX inference took about 96 ms on
CPU after model load. Small player boxes are rejected to avoid treating avatar
menu thumbnails as nearby people. PE-Spatial is a future dense feature encoder,
and V-JEPA is a future temporal representation/prediction model; neither replaces
the detector/tracker interface by itself.

The measured voice choices are Silero VAD ONNX, faster-whisper small CPU int8,
WASAPI speaker loopback, and Windows SAPI Haruka as the current low-latency output.
On a live USB-microphone utterance, fixed 8x input gain produced Silero onset/end
events and faster-whisper recognized `右手を振ってください` in 1.98 seconds.
Replaying that human utterance through the selected VRChat speaker produced the
same events and transcript through WASAPI loopback. SAPI synthesized the matching
Japanese sentence in 0.44 seconds. Qwen3-TTS 0.6B
CustomVoice was tested on an RTX 2080 Ti: FP32 used about 4.4 GiB but generated
3.28–4.88 seconds of Japanese audio in 21.4–28.7 seconds (RTF 5.9–6.5); FP16 hit
a CUDA generation assertion with both PyTorch 2.14/CUDA 13.0 and 2.9.1/CUDA 12.8.
It is therefore not the runtime default on this machine.

`audio.EnergyVAD` is a deterministic test fallback; production uses Silero. It emits `speech_started` before an
utterance is complete, and `BargeInCoordinator` immediately cancels active output.
ASR and speech output are protocols so their model workers stay outside the motor
loop.

The first replay update uses active right-WAVE transitions with real `openvr_raw`
feedback, an 80/20 chronological split and median residual fitting. A candidate is
always retained for audit, while deployment promotion requires at least 1 µm
baseline error and 1% relative improvement. This
small learner is a vertical learning gate rather than SAC. It must not be described
as avatar or social learning because the current feedback observes virtual devices.

### Visual LOOK_AT learning baseline (implementation, live evaluation pending)

`gaze.VisualGazePolicy` adds a separate learned camera-control baseline. Settled
calibration transitions contain the actual image target offset and observed
OpenVR head rotation before and after a bounded head action. They use the same
PAMIQ replay adapter, with outcome `visual_feedback` and reward equal to reduction
in normalized image-centering error. `WorldObject.image_position` preserves this
measurement separately from coarse metric position estimates. A supervised
template target is a stationary calibration fixture; its unit-depth position is
only a bearing proxy and is never valid reach geometry or player identity.

The learner fits a two-axis inverse visual response, checks a chronological
holdout, and rejects stale, simulated, singular, or insufficient-motion data.
This is least-squares system identification, not SAC and not an already-proven
motor improvement. A candidate still needs fresh live before/after task evaluation.
The fitted matrix is used only by Motor Policy for LOOK_AT; the LLM supplies the
target identity and finite goal. Each new camera observation permits one bounded
angular correction. Reusing the same frame holds that target instead of repeatedly
integrating the error at motor frequency. Missing/old image feedback is an error;
the existing supervisor remains responsible for neutralization and watchdogs.

The procedural motor currently limits successive desired targets; it is not a
learned policy or an avatar feedback controller. Real observations remain separate
from desired state and carry source/validity/confidence. Missing live head tracking
stops output; controller activation has a bounded two-second initial grace period.
Unknown controller ownership or a wrong HMD identity is a hard error.
Readback identifies the owned controllers by serial, device class and role hint.
An unavailable deprecated OpenVR role lookup does not invalidate a valid device
pose. Conflicting role ownership, duplicate identity and a wrong role hint still
stop the session. This is device proprioception, not proof of VRChat bindings.

## Coordinate and protocol boundaries

### Unity motor learning environment

`RETURN_TO_REST` is an explicit finite intention without a hand or external
target. It uses the existing bounded neutral trajectory for head and both hands.
`WAIT` and intention expiry hold observed poses and release controller inputs.
Explicit rest does not teleport poses or relax watchdog deadlines. The planner
treats explicit rest as recovery rather than active interaction.

The next motor-learning slice uses a separate Unity scene for goal-conditioned
REACH. Unity supplies resettable ground-truth end-effector/target state; Python
owns policy learning and PAMIQ experience persistence. The first scene is a
kinematic end-effector rig, not a reproduction of VRChat IK or a full-body motion
model. Its policy action is a bounded canonical hand velocity, integrated with
the same speed/acceleration contract intended for Virtual Body deployment. The
LLM still supplies only the target and finite intent.

Unity scene source lives in `unity/MotorLab`; generated projects, builds and runs
remain outside the checkout. The local transport is one reset/step response per
request on loopback TCP. It has no VRChat controls and no machine configuration
commands. Simulation steps use explicit dt, not desktop frame timing. This small
environment adapter keeps the application on its existing Python/PAMIQ runtime;
the inspected ML-Agents release23 Python interface requires Python<=3.10.12,
whereas this application and PAMIQ use Python>=3.12. The standalone experiment
uses Stable-Baselines3 SAC's training loop and replay for optimization, and the
PAMIQ replay adapter for intent-bearing experience persistence. It is not yet a
PAMIQ Trainer or concurrent live learner. Exported deterministic actors can be
loaded with `--reach-policy`; right-hand REACH uses them in the calibrated 1.6 m
body frame, while left-hand REACH remains procedural. Policy transfer, motion
realism and live experience refinement remain separate validation gates.

Unity experience now explicitly records `motor_contract`, `reward_kind`,
`terminated` and `truncated`. The offline SAC candidate updater consumes PAMIQ
records with those semantics and rejects ordinary device-progress rewards.
Missing terminal metadata in historical records is not inferred. Candidates are
saved separately and require evaluation; neither a nonzero parameter update nor
successful replay import is a promotion criterion.

Canonical frame is right-handed, metres, +X forward, +Y left, +Z up. Orientations
are normalized active wxyz quaternions from device-local to stage. The backend maps
position to OpenVR (-y,z,-x), quaternion to (w,-qy,qz,-qx), then applies the explicit
per-driver rigid calibration transform. Readback applies the inverse transform.
Canonical stage origin is a chosen floor origin; it is not an observed VRChat root.

VMT v0.15 receives `/VMT/Raw/Driver`, modes 5/6, full input snapshots and scalar
skeleton updates. The sender does not set RoomMatrix or TrackingOverrides. For the
separate VirtualHMD_OpenVR v0.1 endpoint, the payload is six little-endian doubles:
OpenVR (x,z,y) in centimetres, then Yaw, Pitch, Roll in degrees. Its inspected driver
constructs Rz(Roll) Ry(-Yaw) Rx(Pitch); the adapter inverts that rotation, including
Euler singularities. Mixed-axis and singularity round-trip tests cover this path.
Do not assume other OpenTrack/VRto3D distributions use exactly this wire convention.

## Clock, failure and persistence

Conversation memory records that a reply was planned, without claiming delivery
or a positive social outcome. A conversation event may carry an explicitly
identified `speaker_id`; only a matching player in WorldState receives a
familiarity update. The current mixed-audio pipeline supplies no speaker identity,
so proximity is not used to attribute speech. Familiarity tracks contact and
affinity tracks explicitly positive/negative interactions separately. Unknown
sentiment leaves affinity unchanged. Both maps enter subsequent autonomous LLM
context and are persisted; historical files without affinity load with an empty
affinity map. Reliable live speaker association remains unimplemented.

The producer and worker share `time.perf_counter`'s monotonic clock domain. This
avoids GetTickCount64's coarse timer resolution on Windows Python 3.12. Deadlines
use generation time, not receive time. The local IPC accepts bounded JSON datagrams
with a per-worker unpredictable token and strictly increasing sequences. Polling
occurs before each receive, so queued traffic does not starve the watchdog. No
persistent service or second general-purpose scheduler is installed.

The LLM request does not block motor or watchdog execution. Schema/HTTP failure
stops the session explicitly. Goal expiry holds observed poses; it never authorizes
continued interaction indefinitely. Controller timeouts are latched and require a
new process/lease. See [the detailed lifecycle contract](vmt-backend.md).
Late producer frames cap motor integration at 100 ms instead of generating a
large catch-up movement. Goal expiry uses full wall time; the independent output
deadlines remain unchanged.

Each schema-v1 transition includes observation/world/body, validated decision and
model latency, command, actuation target, next observation, outcome and nullable
reward. The producer serializes a validated transition before handing it to the
PAMIQ collector. Both the pending collector queue and replay retain JSON strings,
not growing graphs of live body/pose models that make cyclic GC pauses grow with
session length. Model objects are reconstructed only for explicit data access.
The replay adapter saves/loads validated JSONL through PAMIQ buffer hooks;
upstream PAMIQ retains its own checkpoint metadata. Live leases, active execution
state and model server settings are never restored from replay. Historical input
values in transition data are observations of requested actions, not startup commands.

## Realtime conversation extension

Audio capture and Silero onset detection must not wait for ASR, cognition or
waveform synthesis. Onset cancels the output generation and publishes a motor
attention event before transcription. Active events renew listening and movement
inhibition; partial transcripts are provisional working state, never social facts
or executable intentions. Final transcripts alone enter durable conversation memory.
ASR work is serialized with a latest-only partial slot and final priority, bounded
audio windows, and generation checks to discard results after interruption.
Chunked re-decoding is explicitly distinguished from a backend's native streaming.
Model inference remains outside the body loop; the existing supervisor owns devices.
Streaming TTS playback must invalidate queued chunks when interrupted, without
waiting for the producer to finish. Backend timings and device receipt are separate
from remote VRChat delivery. Model selection requires local measurements.

## Upstream sources and historical work

- [pamiq-core](https://github.com/MLShukai/pamiq-core) and [pamiq-vrchat](https://github.com/MLShukai/pamiq-vrchat): interaction, model, buffer, concurrency and persistence APIs.
- [VMT OSC API](https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api/): v0.15 roles, inputs, skeletons and retained state handling.
- [VirtualHMD_OpenVR](https://github.com/xiaofeiyu0723/VirtualHMD_OpenVR): v0.1 external head protocol; inspect official source, use official binaries only.
- [Valve OpenVR](https://github.com/ValveSoftware/openvr): driver and application pose semantics.
- [VRChat SteamVR Input 2.0](https://docs.vrchat.com/docs/steamvr-input-20) and [launch options](https://docs.vrchat.com/docs/launch-options): separate live application gates.

The earlier custom native driver/probes are historical and suspended. Their
successful builds or RDP measurements do not validate this console environment.
Voice, perception, root locomotion, social autonomy, RL/imitation and model promotion
follow the validated body slice; they are not represented by placeholder modules.

### Temporary visual feedback gaps
Vision LOOK_AT uses fresh, confident image feedback even without a fitted gaze
policy; approximate monocular positions are not absolute gaze coordinates.
During an active LOOK_AT, a missing/old/uncertain target holds the observed head
orientation instead of turning back toward neutral. A fresh observation resumes
control. LOOK_AT expiry holds the observed orientation while autonomy chooses its
next action; it does not keep chasing an expired target. Explicit RETURN_TO_REST
and subsequent body intentions retain their requested trajectories. This keeps
sensor gaps from generating contradictory head motion and does not extend target
freshness or establish person identity. Motor pending health clears on recovery.
Visual gaze failure conditions include controller version, fitted policy content,
duration, image-position bins and confidence eligibility. Ephemeral detector IDs
are excluded so a new detection number alone cannot bypass repeated-failure
suppression. Changed control/geometry may be evaluated without deleting history.
Planning schemas and EXPLORE composition expose only gaze targets eligible in the
current snapshot. Explicit LOOK_AT resolution applies the same image freshness
and confidence checks as the motor. Low-confidence detections remain world data,
but cannot become executable gaze targets. Motor-time validation remains required
because a valid planning snapshot may become stale during inference.

### Continuous posture and effector ownership
WAIT, completed intentions, policy expiry and manual/autonomous mode changes
preserve observed posture and release controller inputs. This applies to all
eleven trackers, including a gait stopped mid-cycle. VRChat pose retention does
not require a physical balance/settling controller. EXPLORE_HOME changes only
its leased controller inputs; it must not impose the standing template.
A stationary action owns only its requested effectors: LOOK_AT owns head rotation;
WAVE/REACH own the selected hand. Other parts follow the current observed pose,
not the canonical standing template. Posture requests explicitly change posture;
RETURN_TO_REST is an explicit neutral transition only while its intent is active.
This replaces per-skill expiry exceptions with a body-wide continuity contract.
Pose state is live feedback, never restored from old sessions. Invalid device
feedback, output-owner failure and session cleanup still use the existing backend
shutdown/watchdog path; ordinary intention expiry does not invoke that path.
Pose hold is an actuator contract, not evidence that the held pose is natural.
A future learned policy may choose a natural finishing motion within its intent.

### Whole-body learning direction (user correction, 2026-09-19)
The procedural stationary-continuity patches above are temporary baseline fixes,
not the intended motor architecture. See [whole-body learning audit](whole-body-learning-audit.md)
for verified gaps and the replacement boundary. A coordinated task-conditioned
whole-body policy is required; do not extend per-skill gaze/posture rules as a
substitute. Existing SAC is right-hand-only and WholeBodyPolicy is clip imitation.

### Locomotion frame contract for whole-body training
Reuse body.compose/inverse. A training state holds body-local articulation,
tracking_from_body (positional locomotion) and optional world_from_tracking
(measured/simulator world placement after controller locomotion). Tracker output
uses only tracking_from_body; world reward uses both transforms exactly once.
Controller requests remain separate dimensionless actions, never integrated into
an alleged observed transform. Recenter rebases both transforms inversely and
increments a frame epoch while preserving world poses. Unknown world placement
cannot produce a world pose or displacement reward. This contract precedes the
articulated environment; it is not an implemented learner or a VRChat observer.

### Articulated physics development environment
Use Gymnasium Humanoid-v5 / MuJoCo as a separate physics development environment,
with all 17 joint actuators and measured contact/dynamics. A task-conditioned
wrapper adds desired planar velocity and torso height to observations; changing
this task must not reset qpos/qvel. Reward components use measured velocity and
height, effort and action continuity. This is positional physical locomotion,
not a simulation of VRChat joystick locomotion. The controller/world frame
contract remains separate until the VRChat mapping is implemented. No torque
checkpoint may be sent directly to tracker outputs or marked transferable.

### Correction: VRChat actuator-space learning is the primary path
The MuJoCo gravity/torque environment above is an abandoned transfer assumption,
not a prerequisite for this application. VRChat retains tracker pose without a
balance controller. Stop support-001; do not activate its checkpoint. The learned
motor action is a coordinated vector of bounded tracker pose rates plus explicit
controller inputs. Zero pose rate retains pose; zero controller axes stop requested
locomotion. No gravity, automatic neutral posture, or torque-to-avatar equivalence
is assumed. The integration adapter is deterministic, not a procedural skill
selector. Task rewards must use observed avatar/world results; actuator readback
alone remains insufficient. The offline actuator model verifies the interface,
not VRChat IK or task success. Positional root and controller displacement remain
separate under locomotion_frames.py.

WholeBodyTransition optionally carries a TrackerLearningStep: 66 normalized pose
rates, dt, versioned policy observations before/after, terminal/truncation flags,
reward components, scope and evidence references. Actual controller requests
remain in action.left/right.controls, separate from rates. Learning records must
have matching finite reward totals and fixed observation dimensions. Legacy pose
replay remains learning=None; do not retrofit a fabricated reward or infer policy
actions from target poses. This format is not proof of reward validity: collectors
and task evaluators must provide the referenced observations.

### Tracker-rate collection through the existing body owner
The supervised console accepts one bounded 66-rate lease in the current session.
It preempts earlier manual/automatic motor work, uses the existing output owner,
and releases controller inputs. Expiry holds the last observed pose, while any
new body command cancels the rate lease. Record the exact rates and integration
dt with the submitted target, then pair it with the following motor observation,
not an immediate read directly after publication. This is raw experience only:
without a task evaluator reward/learning remain absent. No LLM emits these rates.

### State-conditioned whole-body actor and actuator-space pretraining
Add a single SAC actor over all 66 tracker rates, with current full-body poses,
shortest-arc goal pose errors, previous rates and timestep as its observation.
The goal is a full-body target in the current tracking frame, supplied by the
task adapter, never per-frame coordinates generated by a language model. Updating
the goal preserves the observed pose and policy continuity state. All tracker
positions/orientations are coupled through one network. Controller locomotion
remains a separate input channel and is neutral in this posture-training stage;
tracking-space pose change is never credited as world displacement.

Use the existing Gymnasium/SAC training boundary and PAMIQ ReplayBuffer schema,
not a new runtime scheduler. The initial environment uses the exact pose-rate
integrator and retargeted CC0 poses as start/goal samples. It has no gravity or
torque dynamics. Its rewards are explicitly tracker_geometry: target progress,
pose error, action effort/change and geometric segment distortion. This is
actuator-space pretraining only, not observed avatar IK, contact or task success.
Do not relabel historical VRChat records or auto-promote this candidate.

Export the deterministic actor separately from the trainer, with shape/contract
and hash checks. Test zero action, quaternion sign, bounded actuation and goal
changes from unfinished poses. Evaluate withheld poses and goal switches against
holding pose before controlled avatar trials. Actual VR/world feedback must later
refine and validate the same body policy; pretraining is not completion of the
requested embodied learning loop.

`tracker_training.MotionPriorSAC` optionally initializes the same actor with
velocity examples from the configured CC0 training clips, then alternates SAC
and behavioral-cloning updates. Reversed sequences are explicitly an augmentation;
withheld clips never enter the prior. Evaluate the untrained actor, BC-only actor,
SAC/BC candidate and pose hold separately. Low demonstration loss does not establish
closed-loop stability or superiority to holding pose.

An optional uncentered SVD basis of training-clip pose rates reduces the actor's
action space while preserving one coordinated 66-rate output. The decoder is
linear with a single shared rate normalization and no offset: latent zero gives
exactly zero tracker rates at every pose. It cannot restore a reference pose.
Withheld clips do not fit the basis. Measure both projection reachability and
learned closed-loop performance; explained motion variance alone does not prove
that a new target can be reached. The existing 210-observation/66-rate live
interface remains unchanged: export folds the fixed learned decoder into the
actor. Replay keeps physical rates plus original latent action and decoder ID,
so a training action is never inferred from a pose. Controller/world locomotion
is not represented by this posture decoder.

Zero decoded rates preserve pose, but this actuator invariant does not imply
that a learned actor actually chooses zero when already at its goal. Audit the
exported actor separately at multiple motor timesteps, during goal switches and
twenty-second already-at-goal trials. Preserve failed rollouts, individual cases
worse than holding, maximum tracker-pair distance deviations and hold drift;
an improved average endpoint error cannot authorize promotion. Reference tracker
distances are a geometric diagnostic, not measured bone lengths or VRChat IK.

### Articulated actuator experiment
The global rate basis does not preserve limb lengths. The next executable
training slice uses the imported CC0 skeleton's connected joints: tracking-space
pelvis translation plus local joint rotation increments, converted by forward
kinematics to the same eleven tracker targets. Retain the source's reference
offsets and calibration orientation mapping, including elbow orientation from
the upper arm. No gravity, torque or ground-contact simulation is introduced.
Root translation is tracking-space motion only; controller locomotion remains
neutral and observed world displacement remains unknown.

Goal switches change the task without resetting joint state. Zero joint/root
rates preserve the complete current articulated pose. Scale an entire action
before integration to meet the existing physical tracker velocity limits;
independent endpoint clipping must not break the skeleton. Derive physical
66-rate records from the actual integrated pair and verify that the existing
tracker integrator reproduces the result. Calibration defines body geometry,
not an automatically restored neutral pose.

Use the same Gymnasium/SAC and PAMIQ replay boundaries. Fit a low-dimensional
joint-rate basis only on training clips. The observation includes the existing
tracker/goal/rate/timestep vector plus current local joint orientations, making
the decoder's hidden joint state explicit. A deterministic policy can be anchored
at its goal by subtracting the same network's output with goal-error inputs zeroed;
this is a general equilibrium constraint, not per-skill action rules. Evaluate
it separately from an analytic joint controller and do not count that diagnostic
controller as a learned policy.

Initially this slice is offline: the rig state is known from imported motion.
Live use requires fitting and validating that state against current calibrated
tracker observations; it must never snap the avatar to an imported reference.
Skeleton geometry correctness alone does not establish natural transitions,
VRChat IK quality, full-body learning success or autonomous policy promotion.

`articulated_fit.fit_body` estimates latent joints from an explicit prior and the
eleven observed poses with a bounded CPU optimizer outside the motor loop. It
checks position/rotation residuals, rejects incompatible geometry and returns an
estimate without issuing any body command. Under-observed joints use a small
prior regularizer; they are not relabeled as measured joint angles. Continuous
feedback validation is provided by the explicit trial adapter described below.
The articulated policy uses a separate console option: its distinct observation/
latent-action contract must not be confused with a direct 66-rate actor.

Optional articulated motion BC uses training clips only, including explicitly
labeled reverse motion directions. Its targets are projected joint-rate actions
with common speed limits. Evaluate the BC-only export before SAC and retain both
results: an improved demonstration loss does not imply an improved task policy,
and SAC may worsen the warm start. No candidate is promoted by these scripts.

### Differentiable articulated policy pretraining

An optional bounded rollout-start sampler addresses the gap between source poses
and states reached by the policy itself. Before the existing differentiable
horizon, run the current deterministic whole-body actor without gradients for a
random bounded number of steps; preserve its last confirmed rates in observation.
Use only training-split starts/goals. Reject the entire candidate body increment
when any foot crosses the declared floor, retaining the last valid state for
that sample; never lift feet or alter individual limbs. The sampler is training
data generation, not runtime motion, a teacher action, or live experience. Record
its explicit setting and compare held-out endpoints, maximum per-part errors and
floor intrusion as well as the historical mean error. This does not change the
runtime actor contract, reward, acceptance thresholds, or model promotion gates.
Use the known ideal actuator to differentiate short-horizon task returns through
the existing joint-rate actor. This is model-based policy optimization in the
ideal tracker environment, not measured VRChat learning or a SHAC reproduction.
The actor still uses the same observation and exported inference contract. No
new online scheduler is introduced; retain the existing PAMIQ experience path.
Record model-based optimizer updates separately from SAC and BC updates.

First verify the batched differentiable FK, integration, rate limits, observation
and geometric reward against the established NumPy actuator. A reference-floor
penalty, when enabled for source-space pretraining, is explicitly a geometric
assumption; it does not assert observed avatar contact or general world geometry.
Persist its coefficient and exponent with the actor and replay, so resumed
training and following-observation rewards use the same objective. Legacy
artifacts without these fields retain their original quadratic penalty.
An explicit warm start validates the rig, decoder, update counter and exported
actions before restoring actor/optimizer state. Additional updates and sampling
restart are recorded separately; a new training objective never overwrites the
old candidate or its evaluation.

Augment training with observation-equivalent joint states: an unobserved joint
with one child can twist about that child's offset while applying the inverse
twist to the child. All eleven tracker positions/orientations remain unchanged.
Only eligible joints whose orientations are not tracker outputs may use this
augmentation. This addresses latent-state ambiguity without changing the body
goal or restoring a default pose. Evaluate separately on actual fitted joint
states, goal switches, already-at-goal holds and unseen source clips.

Goal augmentation may blend two training-split poses in root/local-joint space,
using shortest-hemisphere normalized quaternion interpolation. Retain rigid
offsets, and lift only the sampled training goal when its feet would be below
the declared source clearance. This is an explicit dataset augmentation, never
an inference-time body correction. Keep unaugmented goals in the mixture and
record the setting; held-out clips remain excluded from optimization.

### Explicit articulated-controller trial path
Keep the exported actor loader free of SAC/trainer imports. The supervised body
console may load it explicitly alongside a declared tracking-space reference
floor. Reuse the executive's bounded background-call helper to fit one current
body snapshot; do not block the motor loop or create another scheduler/output
owner. Hold observed pose during fitting, reject stale/failed fits, and invalidate
the estimate on manual control or incompatible feedback. At most one fitting
request may be in flight, including across cancellation.

After fitting, feed actual tracker observations and estimated joints into the
actor. Apply only the predicted FK pose increment to the currently observed
pose; never publish the fitted reference as an initial target. Check feedback
against the estimate and refit while holding if correspondence is lost. Reject
a candidate step below the declared reference plane without moving to neutral.
Tracker readback may lag output. Keep at most one outstanding pose increment;
repeat that already-issued target for a bounded feedback window instead of
reverting to an older observation or advancing hidden state again. Accept the
increment only when measured poses match the issued target. Unexpected feedback
or a deadline invalidates the estimate. Defer the learning tuple until matching
feedback arrives; on cancellation/timeout retain a raw, explicitly unconfirmed
transition. The feedback wait does not extend an intention's finite lease.
Latch a hold target from the observation on entry to holding/fitting. Repeatedly
publishing each delayed readback as a new target can echo old commands indefinitely;
holding must preserve one pose until a new intention or manual command takes over.
Goal changes retain the body estimate and previous rates; manual control and
expiry hold the current pose. These constraints are a trial adapter, not proof
that the learned policy solves its task or a replacement for model evaluation.

Replay pairs the following actual tracker observation with its stored predicted
joint estimate, explicitly labeled as estimated. If the estimate does not agree
with that feedback, keep the raw record but omit the learning tuple rather than
inventing valid hidden state. Preserve the existing owner/watchdog and finite
goal leases. Autonomous candidate selection remains a separate promotion step.

### Opt-in decision-to-articulated-task trial
An explicit `articulated_tasks` configuration may bind named high-level posture
intents to complete pose goals and one hashed candidate actor/rig. This is a task
definition, not a per-frame motion program: the same learned actor computes all
eleven tracker increments from feedback. Bind a goal once at intent activation,
preserving the observed pelvis XY and heading; never recompute an anchor each
tick or reset the body on completion. Height remains in the declared floor frame.

This trial requires the independent Decision Layer and purpose executive. Filter
its capability catalogue to the supplied learned tasks, hold and separately
gated controller exploration; unsupported body skills remain unavailable instead
of falling through to procedural head/hand overlays. Conversation remains active.
Retain the established motor thread, output owner and background fitting helper.
The console reuses next-feedback learning records, keyed to the autonomous intent
generation/deadline so a new decision, manual override or expiry truncates the
previous task. No host configuration or candidate is automatically promoted.
General spatial targets, new gestures and world navigation learning remain
required beyond the initial named-posture route.

The body console accepts a candidate only through explicit `--tracker-policy`
or `--articulated-policy` (with its declared floor when required by the model),
and `learned_pose` commands carrying an eleven-point pose goal and 1–20 second
deadline. It uses the existing output owner/watchdog. Changing the goal preserves
pose and preceding rates; manual control or expiry holds pose and clears rates.
Goal changes/preemption/time limits truncate the outgoing task's learning record.
The following actual motor observation, not the submitted target, supplies the
next state and geometry reward. Records retain simulated/device scope and do not
claim avatar outcomes. No candidate is selected by autonomous cognition yet.

### Dialogue has optional body intent and independent speech submission
Superseded by the following user correction: the speech model must not infer
body intent, even in a separate prompt. Dialogue history and WorldState are joint
inputs to a different decision model, continuously outside the speech loop.

### Authoritative independent embodied decision (latest user correction)
In purpose mode with a configured Decision Layer, that separate scorer owns body
choice directly; the chat model no longer proposes purposes, extracts commands,
or reranks body actions. The existing Qwen3-VL-Reranker service receives a bounded
snapshot of world/image, conversation history, body, drives, ongoing commitment
and observed outcomes. It scores executable registered actions plus continuing
the current action. Current capability/target checks still run on application.

Use the existing slow executive and one bounded background decision request.
Evaluate during speech generation, output and body execution; a new utterance
invalidates decisions based on older conversation context and schedules another.
Conversation only commits transcript/reply to shared memory. It never emits a
body action. On decision-service failure retain the current finite body action
and then hold pose, with an explicit unavailable state; do not fall back to the
conversation model or hard-coded social decisions. Legacy no-Decision configurations
retain the earlier purpose planner for compatibility, outside this primary path.

The current decision catalogue covers registered skills and continuation; open
ended multi-step planning and learning new skills are still unfinished. This
change establishes model/loop ownership, not full-body RL or general reasoning.

Candidate IDs in this path identify skill, hand and the current target reference,
independently of catalogue order or unrelated capability availability. A target
reference is still a detector/manual name, not verified persistent person identity.
Action records use the existing plan ID plus step index and retain the candidate,
intent generation and conversation context at activation. Their outcomes retain
that identity, evidence scope and success/unknown result. A context utterance ID
does not assert that the action fulfilled that utterance's request.

The existing executive wakes decision inference on target-set changes, external
goal/commitment updates and action outcomes, even without conversation. Changes
during inference invalidate the older result; only one request remains in flight
and the next request captures the latest state. Unchanged situations keep the
existing two-second reassessment interval. Frame timestamps and small movements
alone do not generate events; periodic reassessment and application-time target
validation remain necessary. No gaze or neutral-pose action is hard-coded by
these events; the separate decision model still chooses the body action.

A selected finite body action is an execution plan, not automatically a durable
commitment. `PurposeRunner.accept(origin='body_decision')` records the action plan
without creating, replacing or reactivating a commitment. Its interruption and
failure cannot suspend or reprioritize a broader goal. Planner-origin plans keep
the existing commitment lifecycle, and observed matching evidence may still
advance a commitment's declared success criterion. Decision snapshots retain the
commitment ID, criterion, target, progress and status, rather than a description
alone. This prevents finished actions becoming permanent inferred goals; it does
not classify a natural-language request as fulfilled without evidence.

When the last matching action was actually observed successful, the executive
suppresses immediate re-execution of that candidate under the same utterance,
world/goal stimulus and intent generation. Target geometry is checked separately
so a moved target can still require a new action. Action completion itself wakes
inference but is not a new external stimulus. Failed or unknown actions are not
marked fulfilled, and a new utterance, target set or goal can enable another
execution. The suppression event names the completed action ID; scores and model
outputs are retained unchanged.

An experimental `LocalSelection` adapter tests a separately hosted instruction
model for next-action selection, using the same world/body/history/outcome
snapshot and registered candidates. It emits only a validated candidate ID;
the executable intent still comes from the registry. Abstention is explicit.
`SelectionResult` is separate from `ScoreResult`: no artificial scores are made.
This is state-only inference, requiring an explicit opt-in for image-bearing
requests, and must use a different server and model from speech generation.
`DecisionSettings` selects exactly one scorer backend or local Selection profile.
Selection is an explicit trial option in the independent purpose executive;
existing scorer configurations remain unchanged. It requires a different local
endpoint and configured model identity from speech. These configuration checks
do not verify server weights: trial provenance records must verify the actual
model artifacts. Selection results enter the same freshness, context-generation,
capability and target validation as scorer results. Abstention leaves the current
finite action and subsequent pose hold unchanged, without invoking another model.
Legacy chat-proposal reranking does not support Selection. No automatic escalation
or default-model promotion is provided. A measured offline result permits bounded
integration trials, not a claim of general decision reliability or live success.

For the articulated trial, outcome evaluation uses the goal actually bound by
the learned motor, keyed to its intent generation. All eleven observed tracker
poses must be fresh and valid before evaluating position and orientation error.
Missing/stale feedback is unknown. Record the measured errors and tolerances;
do not compare to a legacy procedural posture or infer avatar/contact success.

The articulated controller issues only one increment until readback confirms it.
After confirmation, its next integration interval accounts for elapsed time since
the last issued increment, capped at 100 ms (the actor's tested interval range)
and the remaining command deadline. Repeated motor ticks awaiting feedback never
accumulate additional increments. Fit, goal start/end and reset discard old timing
debt; the first increment uses the current frame interval. Replay stores the
actual integration interval used for observation, FK and tracker rates together.
This compensates update cadence, not gravity, inertia or estimated world movement.

Learned posture actions separate acceptance, preparation and execution. The existing
motor owner publishes an immutable, intent-generation-keyed timing record; the
executive only reads it and never rewrites the motor's choice or clock. Preparation
holds the observed pose and has an eight-second deadline from acceptance. The
finite execution duration starts with the first issued actor command, not fitting.
Motor expiry, executive outcomes and decision context use this same timing record.
Preparation failure/timeout is an explicit execution failure, never goal success.
Cancellation or replacement invalidates outstanding fitting; its eventual result
cannot start the old action. A new action may reuse confirmed state, but cannot
inherit timing debt or an unconfirmed increment. Conversation, observation and
decision work continue through the existing loops during preparation. No new
scheduler, pose restoration or conversation authority is introduced.

Historical design, retained here only to explain the replaced implementation:
Ordinary conversation is not a request to choose a new body action. Dialogue may
return action=null; WAIT remains compatible as no new body instruction. Explicit
movement requests may return a structured intent. This lets the independently
running body purpose continue while the language model answers.

The small local model extracts an optional high-level body request in a short
separate inference over only the latest utterance and available choices. It does
not receive historical assistant replies or autonomous plans during extraction.
Speech generation then receives that proposal as data and cannot emit/overwrite
body control fields. Both run outside motor frames in the existing dialogue
worker; no direct tracker commands are generated. Extraction failures preserve
speech and become an explicit unavailable action outcome. This replaces the
unreliable combined reply-plus-optional-action grammar tested on the local model.

Revalidate an optional body intent against current observations after inference.
Unavailable/stale body targets produce a separate action outcome, not loss of a
valid reply. Log proposed action and applied intent separately; blocked actions
must not preempt the current body purpose. Speech submission retries for at most
five seconds while the same dialogue epoch remains current. New speech or manual
preemption discards queued output; successful submission is recorded once and is
not a claim that the partner heard it. This uses the existing executive tick,
output cancellation and shared-memory owner, not another scheduler.
# Role-specific inference adapters (2026-09-20)

Keep the existing PAMIQ agent/environment, motor, replay, and learning owners.
Configuration selects a fixed inference implementation for each role: `llm`
(dialogue, retained for existing configuration compatibility), `purpose.planner`
(slow goal proposal), and `decision.backend` (independent scoring) or
`decision.selection.llm` (listwise selection). Local chat remains loopback-only;
OpenRouter chat is an explicit authenticated adapter, distinct from Jev Decisions.
No failure escalates to another model or provider. Each OpenRouter profile pins
one provider and model, with bounded input/output, timeout and price ceilings.
Returned model/provider/usage are telemetry, never body authority.

The optional slow planner runs beside the existing body decision worker. It updates
only a validated commitment/proposal in shared memory; the body owner still chooses
and validates actions. One planner request is in flight, stale conversation/manual
generations expire, and a model error leaves the current goal and held pose intact.
Planner and dialogue must use separate model identities. Snapshot coordinates and
all device conversions remain unchanged; cloud responses never supply motor frames.

Small installed entry-point adapters extend JSON generation, scoring, chunk ASR,
VAD, speech output and vision detection through their actual existing contracts.
Settings select a named installed adapter and adapter-specific options; unknown
names or incompatible ports fail explicitly. Native realtime sessions need their
own future contract and are not advertised as interchangeable chunk ASR/TTS.
Persistence remains the existing local SQLite memory, goal telemetry and PAMIQ
replay; credentials are environment references only and never persisted.

### Bounded fitting and continuous audio capture

Unchanged body-decision state is reconsidered at configurable `decision.refresh_s`
intervals (five seconds by default). A new utterance, target set, commitment or
action outcome wakes the existing decision owner immediately. Motor observation,
goal progress and output freshness remain independent of that model cadence.
The interval bounds redundant inference; it does not guarantee provider latency.

The body console periodically checkpoints the existing bounded PAMIQ replay as
validated JSONL. The motor thread copies references to immutable encoded records;
one background writer performs disk I/O and atomically replaces the last complete
checkpoint. Pending writes do not queue further snapshots. Close disables late
publication before the final replay save, so an older snapshot cannot overwrite
the completed session. A native crash can lose the interval since the last save,
but no longer requires discarding the whole run. These checkpoints are recorded
experience only; resuming never replays outputs or auto-promotes model weights.

Latent fitting stops when every observed tracker meets the existing position and
orientation tolerances, including when the prior already meets them. Convergence
is checked before further backward/line-search work; final independent FK validation
still rejects an incompatible rig. Cancellation is checked before accepting any
estimate. Report elapsed time, evaluations and the reason for convergence. This
changes preparation cost, not pose commands, tolerances or the eight-second deadline.

WASAPI loopback capture only records and enqueues timestamped audio blocks. The
existing voice processing worker performs resampling, VAD and event delivery so
inference/event locks cannot directly stall device capture. The bounded queue
reports overflow explicitly; it must not silently discard speech or stretch time.
Playback retains its cancellable callback and measures both device underflows and
empty producer buffers. If synthesis runs dry, collect a new bounded prebuffer
before resuming instead of playing each tiny arriving fragment between silences.
An empty ASR result is a normal no-speech observation. Emit that outcome without
requesting dialogue or restarting capture. Actual adapter exceptions remain visible
through the existing service health/retry mechanism; stale recognition is discarded.

### Training across goal changes

Policy-rollout initialization may follow a different training-split goal before
the differentiable optimization horizon. The reached root/joints are retained;
only the previous command-rate input is cleared on goal replacement, matching the
runtime's finite-action boundary in a pose-holding actuator. Configuration controls
the probability of this sampling; zero preserves the prior training path. Sampling
never uses held-out or live evaluation goals. Record switch counts and settings in
candidate provenance. Evaluation includes repeated goals and observed reached states
in addition to the immutable held-out pose audit. These are ideal-actuator training
and offline comparisons, not live VRChat RL or avatar-quality certification.

Capability summaries and execution resolution use the same recursive dependency
check. A registered composition is unavailable while a required adapter/skill is
unavailable, unknown or cyclic. Availability is derived from current registration,
not restored statistics, and is recomputed when an adapter becomes available.
This changes advertised feasibility, not authority to bypass target/contact gates.

Stopping a voice session invalidates its pending and partial recognition generation
before waiting for capture shutdown. Late results/errors cannot publish to the shared
event inbox or speak after the session is closed. The closed pipeline rejects further
input/poll delivery; a replacement service creates a new pipeline. This reuses the
existing generation check, without forcibly terminating inference threads.

### Whole-body tolerance objective

An optional training weight adds progress and residual cost for the largest tracker
error, normalized by the unchanged 0.12 m / 0.35 rad endpoint tolerances and scaled
back to metres. This supplements the mean whole-body objective; it does not mask
trackers, change live acceptance, insert corrective poses or choose a body part.
The NumPy environment/replay and differentiable dynamics use matching terms. Record
the weight in training reports, actor manifests and replay provenance. Zero retains
the previous objective. Candidate comparison still uses both frozen endpoint audits
and floor intrusion, with no automatic promotion from a lower training loss.

### Whole-body prior and coordinate-invariant candidates

An explicit offline candidate may encode the existing observation in the current
pelvis XY/heading frame. Transform goal-error vectors, measured poses, prior rates
and the root latent quaternion together; transform the predicted root translation
and root angular rate back to tracking axes. Other joint rates stay in their parent
frames. Keep absolute height and the floor reference. Use the projected pelvis
forward axis, or its left axis when forward is near vertical. This is one common
coordinate change, not an overlay or default-pose correction. Export it inside the
model so training and runtime inference use exactly the same transformation.

This candidate requires a full-rank identity joint decoder. Its bounded deterministic
action transform is supported by differentiable policy optimization and offline
behavior cloning, not SAC's stochastic density/critic update; reject that path rather
than reporting an invalid probability. Frame identity and network width are model
provenance, and resumption must retain them. Existing tracking-frame exports remain
compatible and are never silently reinterpreted.

The optional prior learns joint-rate targets from offline trajectories between
licensed training poses. Joint-space interpolation supplies a known complete goal;
source/goal hidden-joint transformations are shared, and canonical XY/yaw changes
are common to both. Reject whole teacher samples whose bounded next pose enters the
reference floor. No held-out pose or recorded live evaluation example becomes a
training sample. The learned network still owns all runtime body rates; no solver
or teacher is installed as an automatic fallback. Record cloning and later policy
optimization separately and evaluate repeated goals with both fixed case sets.

Teacher cloning can compare the realized eleven-tracker rates after the shared
bounded kinematic actuator, instead of treating unobservable joint-rate choices
as uniquely correct labels. A small joint-rate regularizer resolves unconstrained
motion while the main error is in observed body motion. Group each training batch
by its recorded timestep; reconstruct the ideal latent state from the observation,
and use the same dynamics for teacher and student. Explicit resumption retains
the teacher dataset hash, architecture, frame, optimizer and cloning update count.

An optional cloned-policy rollout collects additional training-split states reached
by the candidate itself. Teacher labels still come from the full goal-conditioned
kinematic oracle, while previous-rate inputs reflect the actual chosen rollout
action. Sample a fixed teacher/student mixture per episode, extend goal dwell by an
explicit step count, and record rejected whole-body floor actions separately. This
is offline dataset aggregation, not online deployment or a runtime fallback. Keep
the frozen live-start and held-out audits out of both collection and optimization.

A separate explicit candidate frame removes unobservable single-child joint twists
from learned features. Keep the existing external observation/action contract, but
replace its internal joint-quaternion features with forward-kinematic joint positions
in the common pelvis frame. The network predicts every angular rate in these common
axes; rotate each learned rate into its current parent's local axes at the output.
Root rates return to tracking axes. This is a coordinate representation, not an IK
controller or a selected-joint correction: all rates still come from the network.
Equivalent hidden twists must produce identical tracker motion, including the
bounded finite-step actuator. Preserve explicit frame provenance; older model weights
cannot silently migrate into this feature representation. Compare frozen endpoints
and floors before any runtime selection.

Long teacher episodes use a shuffled cycle of the three supported timesteps, every
eight steps, rather than independent episode draws which could omit a timestep.
Preserve previous realized rates across timestep changes. Record the accepted
dataset's actual timestep counts, including for
reused datasets. This changes training coverage only, not evaluation case selection.

Heading changes are another complete-body training goal, not a gaze overlay. An
optional augmentation rotates an entire goal about its pelvis in tracking space,
preserving height and segment geometry. Include a mixture of posture-plus-heading
changes and turning the current complete pose so the network is not taught to reset
posture before turning. The learned actor still emits every joint rate. Turning in
tracking space is not controller locomotion or evidence of world displacement.
Keep this augmentation explicit in training provenance and evaluate separately on
fixed held-out turning goals before exposing it as a runtime capability.

Long model-based training runs save an explicit inference export and optimizer state
at each existing hundred-update progress boundary. Write the training state through
a temporary file, then publish checkpoint metadata only when all snapshot files are
complete. Checkpoints are unevaluated candidates, never automatically promoted.
They use the existing explicit resume-source path and retain actor frame, architecture,
cloning/update counters and source-corpus identity. Resumption restarts the documented
sampling seed; it is additional training from a saved state, not a claim of bit-exact
continuation. A process/host interruption must not be reported as completed training.

## Portable model setup and role checks

Ship editable local, OpenRouter and hybrid role profiles as package data. The setup
command expands one into the existing AutonomousConfig; it does not introduce a
second runtime configuration or select models during execution. An optional base
configuration preserves device, body policy, exploration, learning and memory paths.
Only the three model roles change. New state paths belong outside the checkout.
Local defaults use Japanese LFM speech and Gemma 4 reasoning; the OpenRouter profile
uses Gemma 4 speech/planning and Jev scoring. Model IDs, endpoints, providers,
deadlines and installed adapters remain configuration, not capability allowlists.

A separate model check executes the actual dialogue, purpose and decision contracts
against synthetic input, without opening user memory or connecting any actuator.
It reports per-role response validity, changing speech input, candidate identity and
deadline failures; it never equates these probes with avatar or microphone success.
Credentials come only from each adapter's configured environment variable. Failure
does not replace a model, disable a role or launch a historical model automatically.

## Learned motion sequences through the articulated actor

An optional hash-bound motion catalogue supplies learned eleven-pose trajectories
to the same articulated actor used for static goals. These are reference motions,
not tracker commands: the actor retains ownership of every joint rate. Bind one
common XY/heading transform at action acceptance and retain it for the entire
sequence. Never re-anchor each sample to the moving pelvis. Completion, timeout
and cancellation retain the observed posture, without a default-pose transition.

Periodic priors represent loops; finite basis-regression priors clamp at their
last learned frame and never wrap. Advance reference phase only from confirmed,
fresh body feedback within the existing pose tolerances. Cap each phase increment
so missed motor frames cannot skip a gesture. Record phase coverage and completion
separately from endpoint error. A stationary body at a clip's final pose is not
evidence that the motion was performed. Tracking-root changes, controller inputs
and measured world displacement remain separate channels.

The existing bounded capability-learning worker trains configured licensed clips,
evaluates withheld clip times, and installs the resulting prior between actions.
The executive persists artifact hashes and restores verified artifacts, not saved
availability claims. With an articulated motor, install into its motion catalogue
instead of the legacy walking slot; registry refreshes must preserve this result.
The slow planner may queue supported missing capabilities without taking the body
away from the independent decision loop. This is imitation acquisition from a
configured corpus, not unrestricted motion invention or online task RL. Reuse
PAMIQ observation/replay metadata and the existing worker; add no new scheduler.

A loop request succeeds when fresh confirmed observations cover a whole cycle and
show actual motion after entering the first reference pose. Its arbitrary stopping
phase has no static endpoint requirement; keep endpoint error as a diagnostic.
Finite motions still require their endpoint as well as sequence coverage. Neither
criterion proves avatar semantics, contact or world displacement. Include the
reference hash, playback settings and control contract in failure-condition memory,
so a changed reference does not inherit an unrelated task's execution block.
# Motion acquisition across skeletons and locomotion

Motion lessons may select their own local CC0 source and explicit retarget profile.
The offline retarget adapter transfers world rotation changes relative to a named
source reference onto a declared articulated reference pose. Forward kinematics
uses the destination rig's proportions. The profile records node indices (glTF
names need not be unique), coordinate basis and translation scale. It emits all
eleven poses for the existing imitation trainer; the live actor still owns every
joint rate. GLB is a container variant of the same motion adapter.

SAPI utterances own immutable cancellation tokens and their own output streams.
The configured SAPI adapter runs those native synthesis/playback operations in an
owned spawned process. The parent only posts the latest bounded command and reads
generation-tagged health; stop never waits for native audio. A stalled or crashed
audio process is reported by a heartbeat/death check without blocking the body
thread. Service teardown closes this process and does not replay pending speech.
Cancellation signals the worker without joining it or closing a global PortAudio
stream from another thread. The playback worker opens, aborts and closes its stream
before releasing COM; late worker errors cannot overwrite a newer reply's status.
This prevents concurrent stream-close paths without assuming every native crash
has the same cause. Remote acoustic interruption still requires measurement.

Controller locomotion remains separate from tracking-space pose generation.
Exploration may run a learned gait while bounded, expiring controller leases move
the avatar. Only observed image response is presently available; it is never a
metric world-position reward. The exploration gate may explicitly allow the
startup-verified owned friends Home, and rejects any subsequent instance change.
An explicitly authorized Friends+ trial additionally requires the launcher's
`allow_friends_plus_home: true` session marker in `startup.json`; enabling controlled
Home exploration alone does not allow Friends+. The launcher must verify the owned
Home and record its exact instance before writing this marker. The runtime continues
to require the same instance, Home room, and fresh visual input on every check.
The capability also requires a fresh visual frame. Losing capture cancels the
current exploratory gait as well as its controller lease and holds the observed
whole-body pose; dialogue and other independent intentions remain available.
The motor checks frame age independently of slow decision work so an earlier
exploration choice cannot keep moving after visual feedback has disappeared.
Missing configured motion lessons are learned by the existing PurposeRunner worker,
evaluated and installed between actions. Dialogue does not schedule motor frames.

Optional body facing produces horizontal image-based heading references by rotating
the current complete posture about its pelvis. It shares the articulated actor and
fresh feedback/deadline contracts with other tasks. It does not overlay a separate
head controller, reset the posture, or claim that horizontal alignment is eye contact.
When that visual target temporarily disappears, the actor holds the observed pose
without issuing motion from stale geometry. The same target can resume within a
1.5-second grace period; prolonged loss fails the task, and the original action
deadline is not extended. Freshness errors include age and confidence for diagnosis.

Completed one-shot body actions are associated with the current utterance ID, not
detector IDs or planner revisions. Retain their successful outcomes for that turn
and remove duplicate candidates while preserving world-facing targets and ongoing
exploration. New utterances, including identical text, receive a new identity.
Historical turns remain context, never fresh commands. Failure memory uses the
latest consecutive observed failures under the same control conditions: an old
success must not exempt a now-failing action from its retry bound.

Latent joint fitting minimizes excess error against each tracker's unchanged
position/angular acceptance limits. A low average error is insufficient when one
tracker misses its bound. Fit-contract identity is part of failure conditions;
changing that estimator does not erase prior evidence or claim a new actor.
Device confirmation uses a separate 10 micrometre / 10 microradian comparison.
The prior frame must not acknowledge a submillimetre pending command: doing so
advances latent joints before the device moves and accumulates estimation drift.
The same confirmation contract gates learning replay; fitting/task tolerances and
the existing feedback deadline remain unchanged.
For commands below that absolute confirmation tolerance, also require the observed
pose to approach the issued target relative to its starting error. Commands below
output resolution hold both measured and latent state; discarded microsteps must
not accumulate only inside the model.
Apply that comparison per tracker, since HMD and VMT packets can arrive in different
frames; a large confirmed head change does not confirm a small pending foot change.

In controller locomotion mode, motion references share one session tracking-space
XY origin. Rebinding each short gait to its imperfect endpoint integrates tracking
error into room-scale travel, independently of the controller's world movement.
Keep heading current and clip-relative translation intact. A configured `current`
motion anchor remains available for deliberate relative tracking-space sequences.
The origin affects goals only; it never publishes a neutral pose on completion.
When a bent pelvis points its forward axis vertically, use its orthogonal lateral
axis to define horizontal task heading. Such a pose is not invalid tracking and
must not prevent the next motion or posture request from being attempted.
