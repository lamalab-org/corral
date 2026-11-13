"""Independent async clients for native KnowLine choices and Gemma chat evidence.

Servers remain external. Importing this module loads no model or serving library.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from corral.epistemic_analysis.budget import RequestBudgetExceeded, record_request
from corral.epistemic_analysis.context import current_context
from corral.epistemic_analysis.scheduling import rolling_map
from corral.epistemic_analysis.schema import content_hash
from corral.epistemic_analysis.storage import atomic_save, read_record, save_record
from corral.epistemic_analysis.trace_check_explanations import EXPLANATION_PROMPT

from .base import record_model
from .jev import JevError, validate_answers

NATIVE_GRAPH_VERSION = "knowline-native-graph-1"


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    revision: str
    endpoint: str
    token_endpoint: str
    context_tokens: int = 131072
    output_tokens: int = 1
    timeout: float = 900
    retries: int = 2
    concurrency: int = 1
    scoring: dict = field(default_factory=dict)
    metadata_endpoint: str = ""

    def __post_init__(self):
        if (
            self.provider not in {"knowline", "gemma"}
            or not self.model
            or not self.revision
        ):
            raise ValueError("Provider, model, and pinned revision are required")
        for value in (self.endpoint, self.token_endpoint, self.metadata_endpoint):
            if not value:
                continue
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
            ):
                raise ValueError(
                    "Endpoints must be HTTP(S) URLs without embedded credentials or query parameters"
                )
        if not self.endpoint or not self.token_endpoint:
            raise ValueError("Inference and native token endpoints are required")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        for name in ("context_tokens", "output_tokens", "concurrency"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.retries) is not int or not 0 <= self.retries <= 10:
            raise ValueError("retries must be between 0 and 10")
        if self.output_tokens >= self.context_tokens:
            raise ValueError("Output allowance must leave room for input")

    @property
    def fingerprint(self):
        return content_hash(asdict(self))


class Provider:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout, transport=transport)
        self.semaphore = asyncio.Semaphore(config.concurrency)
        # CPU tokenization can prepare the next request while the GPU runs the
        # current one. Keep inference concurrency unchanged (especially for the
        # KnowLine server's shared prefix cache).
        self.preparation_concurrency = 4
        self.token_semaphore = asyncio.Semaphore(self.preparation_concurrency)
        self.metadata = None
        self._token_counts = OrderedDict()

    async def close(self):
        await self.client.aclose()

    async def request(self, url, body=None):
        gate = (
            self.token_semaphore
            if url == self.config.token_endpoint
            else self.semaphore
        )
        async with gate:
            for attempt in range(self.config.retries + 1):
                try:
                    response = await (
                        self.client.get(url)
                        if body is None
                        else self.client.post(url, json=body)
                    )
                    response.raise_for_status()
                    return response.json()
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    # A lost inference response does not mean the server stopped
                    # computing. Blind replay can fill the queue with duplicates.
                    if url == self.config.endpoint and isinstance(
                        exc,
                        httpx.ReadTimeout
                        | httpx.ReadError
                        | httpx.WriteTimeout
                        | httpx.WriteError
                        | httpx.RemoteProtocolError,
                    ):
                        raise
                    retryable = not isinstance(
                        exc, httpx.HTTPStatusError
                    ) or exc.response.status_code in {408, 429, 500, 502, 503, 504}
                    if not retryable or attempt == self.config.retries:
                        if isinstance(exc, httpx.HTTPStatusError):
                            raise httpx.HTTPStatusError(
                                f"{exc}\nProvider response: {exc.response.text[:4000]}",
                                request=exc.request,
                                response=exc.response,
                            ) from exc
                        raise
                    await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError("Exhausted provider attempts")

    async def preflight(self):
        """Verify an advertised revision when configured; record all server metadata."""
        if not self.config.metadata_endpoint:
            raise ValueError(
                "A model metadata endpoint is required to verify the served revision"
            )
        metadata = await self.request(self.config.metadata_endpoint)
        if self.config.provider == "gemma":
            match = next(
                (
                    m
                    for m in metadata.get("data", [])
                    if m.get("id") == self.config.model
                ),
                None,
            )
            if match is None or self.config.revision not in {
                match.get("root"),
                match.get("revision"),
                str(match.get("root", "")).rstrip("/").rsplit("/", 1)[-1],
            }:
                raise ValueError(
                    "Gemma served model/revision differs from configuration"
                )
            if (
                match.get("max_model_len", self.config.context_tokens)
                < self.config.context_tokens
            ):
                raise ValueError("Configured Gemma context exceeds the server limit")
        else:
            if (
                metadata.get("revision") != self.config.revision
                or metadata.get("scoring") != self.config.scoring
            ):
                raise ValueError(
                    "KnowLine served revision/scoring differs from configuration"
                )
            if metadata.get("context_tokens", 0) < self.config.context_tokens:
                raise ValueError("Configured KnowLine context exceeds the server limit")
        self.metadata = metadata
        return metadata

    async def count(self, body):
        payload = (
            body
            if self.config.provider == "knowline"
            else {k: body[k] for k in ("model", "messages", "chat_template_kwargs")}
        )
        # Fit checks and inference use identical bodies. Reuse the server's native
        # count without retaining large traces or tokenizing them twice. Preserve
        # object order: it can change native prompt rendering.
        key = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).digest()
        if key in self._token_counts:
            self._token_counts.move_to_end(key)
            return self._token_counts[key]
        result = await self.request(self.config.token_endpoint, payload)
        count = result.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid native token count")
        self._token_counts[key] = count
        if len(self._token_counts) > 1024:
            self._token_counts.popitem(last=False)
        return count

    async def fits(self, body):
        return (
            await self.count(body) + self.config.output_tokens
            <= self.config.context_tokens
        )

    async def infer(self, body):
        count = await self.count(body)
        if count + self.config.output_tokens > self.config.context_tokens:
            raise ValueError("Native context budget exceeded; source was not truncated")
        started = time.monotonic()
        response = await self.request(self.config.endpoint, body)
        return {
            "elapsed_s": time.monotonic() - started,
            "response": response,
            "native_input_tokens": count,
            "request": body,
            "request_hash": content_hash(body),
            "provider": asdict(self.config),
            "server_metadata": self.metadata,
        }


class KnowLineClient(Provider):
    def __init__(self, config, *, preparation_buffer=0, request_timeout=None, **kwargs):
        if config.provider != "knowline":
            raise ValueError("KnowLineClient requires the knowline provider")
        if type(preparation_buffer) is not int or not 0 <= preparation_buffer <= 64:
            raise ValueError("preparation_buffer must be between 0 and 64")
        if request_timeout is not None and (
            not math.isfinite(request_timeout) or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be positive and finite")
        super().__init__(config, **kwargs)
        self.preparation_buffer = preparation_buffer
        self.request_timeout = request_timeout
        if request_timeout is not None:
            self.client.timeout = httpx.Timeout(request_timeout)
        self._buffer_ready = not preparation_buffer

    async def preflight(self):
        metadata = await super().preflight()
        if self.preparation_buffer:
            backend = metadata.get("inference_backend", {})
            limit = backend.get("adapter", {}).get("max_concurrency")
            if (
                backend.get("backend") != "vllm-native-choice-1"
                or type(limit) is not int
                or not 0 < limit <= self.config.concurrency
            ):
                raise ValueError(
                    "Preparation buffering requires a verified native vLLM server "
                    "with an inference limit no higher than configured concurrency"
                )
            self.semaphore = asyncio.Semaphore(
                self.config.concurrency + self.preparation_buffer
            )
            self._buffer_ready = True
        return metadata

    async def infer(self, body):
        if not self._buffer_ready:
            raise ValueError("Preparation buffering requires server preflight")
        exchange = await super().infer(body)
        if self.preparation_buffer or self.request_timeout is not None:
            # Operational lookahead does not change scoring or invalidate labels.
            # The adapter admits only its independently verified inference limit.
            exchange["client_scheduling"] = {
                "preparation_buffer": self.preparation_buffer,
                "request_limit": self.config.concurrency + self.preparation_buffer,
                "server_inference_limit": (self.metadata or {})
                .get("inference_backend", {})
                .get("adapter", {})
                .get("max_concurrency"),
            }
            if self.request_timeout is not None:
                exchange["client_scheduling"]["request_timeout_seconds"] = (
                    self.request_timeout
                )
        return exchange

    def body(self, definition, state, *, partial=False):
        return {
            "model": self.config.model,
            "state": state,
            "questions": {definition.name: definition.question(partial=partial)},
        }

    def parse(self, response, definition, threshold):
        answer = response.get("answers", {}).get(definition.name)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Missing native KnowLine choice")
        probabilities, choice = answer.get("probabilities"), answer.get("choice")
        allowed = {"present", "absent", "uncertain"}
        if (
            choice not in allowed
            or not isinstance(probabilities, dict)
            or set(probabilities) != allowed
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in probabilities.values()
            )
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=0.01)
        ):
            raise ValueError("Invalid option probabilities or model-assigned N/A")
        maximum = max(probabilities.values())
        if probabilities[choice] < maximum:
            raise ValueError("Choice is inconsistent with maximum option probability")
        return {
            "status": choice if maximum >= threshold else "uncertain",
            "choice": choice,
            "probabilities": probabilities,
            "maximum_option_probability": maximum,
            "confidence": answer.get("confidence"),
            "threshold": threshold,
        }


class GemmaClient(Provider):
    def __init__(self, config, **kwargs):
        if config.provider != "gemma":
            raise ValueError("GemmaClient requires the gemma provider")
        super().__init__(config, **kwargs)

    def body(self, definition, state, *, partial=False, repair=None):
        del repair  # Retained for callers of the previous evidence interface.
        instructions = (
            EXPLANATION_PROMPT
            + "\n"
            + definition.question(partial=partial)["instructions"]
        )
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": self.config.output_tokens,
        }


class KnowLineGraphClient:
    """Persist native requests by content hash, deduplicating their large states."""

    def __init__(
        self, provider, output, public_context, *, batch_size=32, progress=None
    ):
        if not 1 <= batch_size <= 64:
            raise ValueError("Native KnowLine graph batches require 1..64 questions")
        self.provider = provider
        self.output = Path(output)
        self.public_context = public_context
        self.batch_size = batch_size
        self.progress = progress
        self.completed_questions = 0

    @property
    def provenance(self):
        return {
            "api": "native-systemone",
            "implementation": NATIVE_GRAPH_VERSION,
            "confidence_readout": "maximum-option-probability",
            "provider": asdict(self.provider.config),
            "endpoint": self.provider.config.endpoint,
            "batch_size": self.batch_size,
            "timeout_s": self.provider.config.timeout,
            "max_retries": self.provider.config.retries,
        }

    async def evaluate(self, model, state, questions):
        if model != self.provider.config.model:
            raise ValueError("Graph model differs from the pinned native provider")
        state = {**state, "trace_context": self.public_context}
        state_key = content_hash(state)
        state_path = self.output / "states" / f"{state_key}.json"
        if state_path.exists():
            if content_hash(json.loads(state_path.read_text())) != state_key:
                raise ValueError("Cached graph state changed")
        else:
            atomic_save(state_path, state)
        answers = {}

        async def batch(selected):
            body = {"model": model, "state": state, "questions": selected}
            key = content_hash(
                {
                    "version": NATIVE_GRAPH_VERSION,
                    "provider": asdict(self.provider.config),
                    "wire_body": json.dumps(body, ensure_ascii=False),
                }
            )
            path = self.output / "exchanges" / f"{key}.json"
            saved = read_record(path, key)
            if not saved or saved["state"] != "complete":
                if not await self.provider.fits(body):
                    if len(selected) > 1:
                        items = list(selected.items())
                        midpoint = len(items) // 2
                        await batch(dict(items[:midpoint]))
                        await batch(dict(items[midpoint:]))
                        return
                    raise RequestBudgetExceeded(
                        {
                            "backend": "knowline",
                            "reason": "native_context_limit",
                            "budget_tokens": self.provider.config.context_tokens,
                        }
                    )
                result = await self.provider.infer(body)
                validate_answers(result["response"], selected)
                # State is saved once per window; the exact request can be rebuilt.
                result.pop("request", None)
                saved = {
                    "fingerprint": key,
                    "state": "complete",
                    **result,
                    "request": {
                        "model": model,
                        "state_ref": state_key,
                        "questions": selected,
                    },
                }
                save_record(path, saved)
            checked = validate_answers(saved["response"], selected)
            for name, answer in checked.items():
                if answer["type"] == "choice" and answer["probabilities"][
                    answer["choice"]
                ] < max(answer["probabilities"].values()):
                    raise JevError("Native choice differs from maximum probability")
                answers[name] = {
                    **answer,
                    "model": model,
                    # Native confidence is entropy-derived. Graph acceptance uses
                    # the selected option's probability, matching the checks run.
                    **(
                        {
                            "native_confidence": answer["confidence"],
                            "confidence": answer["probabilities"][answer["choice"]],
                        }
                        if answer["type"] == "choice"
                        else {}
                    ),
                }
            record_model(model)
            context = current_context()
            if context is not None:
                context.requests.append(
                    {
                        "exchange": key,
                        "artifact_root": str(self.output),
                        "state_ref": state_key,
                        "questions": list(selected),
                    }
                )
            record_request(
                {
                    "backend": "knowline",
                    "native_input_tokens": saved["native_input_tokens"],
                    "is_estimate": False,
                    "question_count": len(selected),
                    "outcome": "complete",
                }
            )
            self.completed_questions += len(selected)
            if self.progress:
                self.progress(
                    {
                        "completed_questions": self.completed_questions,
                        "last_exchange": key,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        items = list(questions.items())
        size = self.batch_size
        # Preparation for later batches overlaps inference. The shared provider
        # still bounds tokenization and inference separately across all traces.
        await rolling_map(
            batch,
            (
                dict(items[offset : offset + size])
                for offset in range(0, len(items), size)
            ),
            concurrency=self.provider.config.concurrency
            + self.provider.preparation_concurrency,
        )
        # Completion order must not change downstream node/edge construction.
        return {name: answers[name] for name in questions}
