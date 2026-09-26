# Independent candidate Decision Layer

With `purpose` and `decision` configured, the separate Decision Layer directly
chooses body intentions from world state, body, drives, conversation history and
observed outcomes. The conversation model generates speech only. Motor policies
generate trajectories. Inference never runs in the motor/device thread.
Legacy single-intent mode retains the earlier proposal/reranking path.

## Interface and backends

`DecisionInput` contains a coherent `WorldState`, additional state (drives, recent
outcomes, body summary), image bytes encoded as base64, capture time and named
`Candidate` objects. Each candidate has a description and structured intent.
`Scorer.score(request)` returns one finite `CandidateScore` per candidate, in
input order. Selection checks snapshot age and uses stable IDs for ties.

| Backend | Input | Score |
| --- | --- | --- |
| `qwen` | Image + state + individual action | yes-minus-no relevance logit |
| `nemotron` | Image + state + individual action | classifier relevance logit |
| `jev` | State + individual action | ordered rubric expectation / 2 |
| `jev_openrouter` | Same official Jev, through OpenRouter | ordered rubric expectation / 2 |
| `http` | Same contract, local service transport | upstream semantics preserved |

There is no softmax across candidates and no listwise prompt. Local models use
one candidate per sequence, batched for throughput. Candidate-as-query and
observation-as-document are used by both local models. FP16 may cause small score
changes when padding/batch composition changes; the benchmark measures this.
Scores are not calibrated probabilities of action success or interchangeable
across backends. Retrieval models require validation for embodied decisions.

Official Jev is text-only. Image-bearing requests require explicit
`allow_state_only: true`; its result always reports `image_used: false`.
It does not receive image bytes disguised as text. Use perception-derived state
when selecting this backend. Visual-only cases cannot be fairly scored as Jev
vision capability.

## Installation and service

Use a separate Python environment with the `decision` extra. On the RTX 2080 Ti,
install a compatible CUDA PyTorch wheel (validated baseline: torch 2.8.0/cu126,
torchvision 0.23.0), FP16 and SDPA. BF16/FlashAttention2 are not assumed.
Download model snapshots outside the repository; review NVIDIA's custom model
code before using its `trust_remote_code` loader. No downloads occur at runtime.

Validated snapshot revisions:

- Qwen: `4bd860ac4f15ad1897a214615cccc700f8f71818`
- NVIDIA: `0c733be3b6451289f4288c0b170376d1fbeda55a`

Example local backend config (outside the repository):

```json
{
  "backend": "qwen",
  "model_path": "D:/models/qwen-reranker",
  "device": "cuda:0",
  "expected_device_name": "NVIDIA GeForce RTX 2080 Ti",
  "max_pixels": 65536,
  "batch_size": 8,
  "max_length": 2048
}
```

Select CUDA by GPU UUID using `CUDA_VISIBLE_DEVICES`; CUDA indices can differ
from `nvidia-smi` indices. `expected_device_name` rejects a mismatched GPU before
loading weights. Keep other GPU workloads running.

```powershell
& $decisionPython -m myumiq_vrchat.decision_service --config $backendConfig --port 18520
```

The service binds only to 127.0.0.1, serializes inference, limits request size,
rejects silent text truncation and accepts bytes rather than remote image URLs.
`GET /health` confirms model readiness; `POST /score` accepts `DecisionInput` JSON.

For official Jev through OpenRouter, use:

```json
{
  "backend": "jev_openrouter",
  "api_key_env": "OPENROUTER_API_KEY",
  "jev_model": "typesafe/jev-1.13",
  "allow_state_only": true
}
```

The endpoint is `https://openrouter.ai/api/alpha/decisions`, not chat completions.
Use an OpenRouter key only with OpenRouter. The direct TypeSafe backend uses
`https://api.typesafe.ai/v1/systemone`, `TYPESAFE_API_KEY`, and `jev-latest`.
No keys belong in config files, logs or source. A local launcher can read an
existing pi OpenRouter credential and pass it through the child environment;
MyuMIQ itself does not search credential stores.

## Autonomous integration

Add to the existing autonomous configuration:

```json
"decision": {
  "backend": {"backend": "http", "endpoint": "http://127.0.0.1:18520", "timeout_s": 5},
  "max_age_s": 5
}
```

In purpose mode, `embodied_decision.py` submits a bounded world/image + shared
conversation/body/memory snapshot roughly every two seconds, with at most one
request in flight. It evaluates available body actions and continuing the current
action while both speech and body execution proceed. Transcripts invalidate old
conversation snapshots; manual changes invalidate pending actions. Current target
and capability checks run again before activation. Continuing or expiry keeps
the current pose. A missing image is explicit state-only inference.

Failures are reported as unavailable: finish the current finite action, then
hold pose. There is no conversation-model fallback in this path. No `decision`
configuration retains the legacy chat-based purpose planner. The current
catalogue is a bounded baseline, not open-ended goal generation or whole-body RL.

`goals.jsonl` records request context, candidate IDs/intents, raw scores, selected ID, capture
time, modality and latency. PAMIQ intent metadata and interaction memory include
a compact backend/model/selection reference tied to the applied action, unaffected
by later scoring. `decisions.jsonl` records activation. Existing watchdog, bounded motor
step, finite session and safe shutdown behavior remain in place.

## Reproduce comparisons

```powershell
& $decisionPython -m myumiq_vrchat.decision_benchmark `
  --config $backendConfig --cases $cases --output $results --repeats 5
```

Cases are JSON objects with `id`, `request`, optional local `image` path and
predeclared `acceptable` candidate IDs. The first case must contain 16 distinct
candidates. Quality uses the original ordering; reverse and seeded shuffle check
top-1 stability and maximum score change. Latency uses 2 warmups followed by
repeated 2/4/8/16-candidate requests. Output preserves every raw result.

End-to-end measurements include file read, base64, validation, image decode,
preprocessing, GPU inference and result transfer, or HTTP for Jev. They exclude
model load and a new live camera capture. Service transport adds another measured
`transport_end_to_end` field. GPU figures are PyTorch peak allocated/reserved
memory, not total Windows GPU consumption. No image/embedding cache is used.

Keep machine-specific results outside the repository. Distinguish controlled
state scenarios over real captured images from fresh live-world validation.
Small hand-authored cases are smoke tests, not a general accuracy benchmark.

## Explicit local Selection trials

The independent purpose executive also accepts a local listwise Selection profile.
Configure exactly one of `backend` (independent scoring) or `selection`:

```json
{
  "selection": {
    "llm": {
      "base_url": "http://127.0.0.1:18488/v1",
      "model": "decision-model",
      "timeout_s": 15
    },
    "allow_state_only": true
  },
  "max_age_s": 5
}
```

This is the `decision` section of an autonomous configuration with `purpose`
enabled. The model ID must match the separately launched local decision server;
speech requires a different endpoint and configured model identity. Record actual
weight hashes to verify the roles: aliases alone do not prove different weights.
No server startup, automatic switching or default promotion is implied.

Selection defaults to `use_image=false`. With `allow_state_only=true`, it then
omits any supplied raw image and records `image_used=false`. For a configured
vision-capable model, set `use_image=true` to send the coherent JPEG and structured
state together. The same `allow_state_only` flag explicitly permits fallback when
no current image is available; without it, image-enabled selection requires an
image. Live JPEG encoding runs on the vision worker, with a maximum 640-pixel side
and 80,000 base64 characters; capture timestamps are preserved. Unsupported model
modalities fail normally, without silently retrying another model. The model
returns one registered candidate
ID or null for abstention. `SelectionResult` contains no fabricated scores and is
not accepted by the scoring benchmark above. Evaluate it separately for ordering,
abstention, completed requests, new requests, target grounding and model failures.
The same executive freshness, capability, context and completion checks apply.
With images, the structured response also includes a short `scene` observation
and `visible_target_ids`. The runtime rejects a chosen target absent from that
list or any invented ID. This consistency check cannot make a weak vision model
accurate: evaluate false detector labels on empty rooms and decorated scenes.
Provider support for text-only structured output does not establish image support.

Sources: [Qwen model](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B),
[NVIDIA model](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2),
[Jev state modalities](https://docs.typesafe.ai/concepts/state),
[OpenRouter OpenAPI](https://openrouter.ai/openapi.json).
