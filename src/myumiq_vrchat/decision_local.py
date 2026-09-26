"""Isolated FP16 multimodal inference; each sequence contains one candidate only."""

import base64
import io
import json
import time

from .decision import CandidateScore, ScoreResult


class LocalReranker:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoProcessor

        if config.model_path is None:
            raise ValueError("a reviewed local model snapshot is required")
        self.config = config
        self.torch = torch
        self.device = torch.device(config.device)
        if self.device.type != "cuda":
            raise ValueError("local reranker requires an explicit CUDA device")
        if (
            config.expected_device_name is not None
            and torch.cuda.get_device_name(self.device) != config.expected_device_name
        ):
            raise ValueError("CUDA device does not match expected_device_name")
        path = str(config.model_path)
        common = dict(local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa")
        if config.backend == "qwen":
            from transformers import Qwen3VLForConditionalGeneration

            lm = Qwen3VLForConditionalGeneration.from_pretrained(path, **common)
            self.processor = AutoProcessor.from_pretrained(
                path, local_files_only=True, padding_side="left"
            )
            vocab = self.processor.tokenizer.get_vocab()
            self.head = (
                (lm.lm_head.weight[vocab["yes"]] - lm.lm_head.weight[vocab["no"]])
                .detach()
                .clone()
                .to(self.device)
            )
            self.model = lm.model.to(self.device).eval()
        else:
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    path, trust_remote_code=True, **common
                )
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(
                path,
                trust_remote_code=True,
                local_files_only=True,
                max_input_tiles=config.max_tiles,
                rerank_max_length=config.max_length,
            )
        self.model.config.use_cache = False

    def _inputs(self, request, candidates, image):
        observation = json.dumps(
            {"world": request.world.model_dump(mode="json"), "state": request.state},
            ensure_ascii=False,
            sort_keys=True,
        )
        # Both backends receive candidate-as-query, observation-as-document.
        queries = [
            request.instruction
            + " Candidate action: "
            + c.description
            + " Intent: "
            + json.dumps(c.intent, ensure_ascii=False, sort_keys=True)
            for c in candidates
        ]
        if self.config.backend == "nemotron":
            return self.processor.process_queries_documents_crossencoder(
                [{"question": q, "doc_text": observation, "doc_image": image} for q in queries],
                truncation=False,
                padding=True,
                return_tensors="pt",
            )
        from qwen_vl_utils import process_vision_info

        pairs = []
        for query in queries:
            content = [
                {
                    "type": "text",
                    "text": "<Instruct>: Determine whether the observed situation supports taking the candidate action."
                    + "\n<Query>:"
                    + query
                    + "\n<Document>:",
                }
            ]
            if image is not None:
                content.append(
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 4096,
                        "max_pixels": self.config.max_pixels,
                    }
                )
            content.append({"type": "text", "text": observation})
            pairs.append(
                [
                    {
                        "role": "system",
                        "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                        'Note that the answer can only be "yes" or "no".',
                    },
                    {"role": "user", "content": content},
                ]
            )
        texts = self.processor.apply_chat_template(
            pairs, tokenize=False, add_generation_prompt=True
        )
        images, _ = process_vision_info(pairs, image_patch_size=16)
        return self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            do_resize=False,
            return_tensors="pt",
        )

    def score(self, request):
        from PIL import Image

        torch = self.torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        image = None
        if request.image_base64 is not None:
            with Image.open(io.BytesIO(base64.b64decode(request.image_base64))) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("image is too large")
                image = source.convert("RGB")
        scores, preprocessing, forward = [], 0.0, 0.0
        with torch.inference_mode():
            for offset in range(0, len(request.candidates), self.config.batch_size):
                candidates = request.candidates[offset : offset + self.config.batch_size]
                t = time.perf_counter()
                inputs = self._inputs(request, candidates, image)
                if inputs["input_ids"].shape[1] > self.config.max_length:
                    raise ValueError(
                        "decision input exceeds token limit; truncation is not allowed"
                    )
                inputs = {
                    k: (
                        v.to(self.device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in inputs.items()
                }
                torch.cuda.synchronize(self.device)
                preprocessing += time.perf_counter() - t
                t = time.perf_counter()
                output = self.model(**inputs)
                if self.config.backend == "qwen":
                    logits = output.last_hidden_state[:, -1] @ self.head
                else:
                    logits = output.logits.reshape(-1)
                values = logits.float().cpu().tolist()
                torch.cuda.synchronize(self.device)
                forward += time.perf_counter() - t
                scores.extend(
                    CandidateScore(id=c.id, score=v)
                    for c, v in zip(candidates, values, strict=True)
                )
        return ScoreResult(
            backend=self.config.backend,
            model=self.config.model_path.name,
            captured_at=request.captured_at,
            scores=tuple(scores),
            image_used=image is not None,
            semantics="relevance_logit",
            timings_ms={
                "end_to_end": (time.perf_counter() - started) * 1000,
                "preprocessing": preprocessing * 1000,
                "forward": forward * 1000,
            },
            metadata={
                "device": torch.cuda.get_device_name(self.device),
                "dtype": "float16",
                "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20,
                "max_pixels": self.config.max_pixels,
                "max_tiles": self.config.max_tiles,
                "batch_size": self.config.batch_size,
                "image_cache": False,
            },
        ).validate_request(request)
