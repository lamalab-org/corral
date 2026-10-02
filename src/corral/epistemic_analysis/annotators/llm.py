"""Annotators llm for epistemic trace analysis."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import aclosing
from functools import partial
from typing import Any

from loguru import logger

from corral.epistemic_analysis._utils import (
    format_window,
    iter_windows,
)
from corral.epistemic_analysis.budget import (
    RequestBudgetExceeded,
    estimate_request,
    is_context_error,
    numeric_usage,
    record_request,
)
from corral.epistemic_analysis.context import current_context, view_instructions
from corral.epistemic_analysis.graph import (
    coerce_edge,
    coerce_node,
)
from corral.epistemic_analysis.ontology import (
    PASS_A_INSTRUCTIONS,
    PASS_A_SYSTEM,
    PASS_B_INSTRUCTIONS,
    PASS_B_SYSTEM,
)

from .base import AnnotationBackend, record_model

NODE_WINDOW_POLICY = "first-window-per-message-v1"


class LLMError(RuntimeError):
    pass


def _load_litellm():
    import litellm

    return litellm


def _extract_json_text(raw: str) -> str:
    """Strip markdown fences and find the outermost JSON object in *raw*."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


async def llm_json_call_async(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_retries: int = 5,
    timeout_s: int = 120,
) -> dict[str, Any]:
    litellm = _load_litellm()
    last_err: Exception | None = None
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    context = current_context()
    budget = context.config.max_request_tokens if context else 0
    options = {}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    estimated, estimator = estimate_request(body)
    try:
        # LiteLLM selects the model tokenizer where available (otherwise estimates).
        estimated = litellm.token_counter(model=model, messages=messages) + 256
        estimator = f"litellm.token_counter:{model}+256"
    except Exception:
        pass
    if budget:
        reserve = context.config.llm_output_reserve_tokens
        options["max_tokens"] = reserve
        try:
            info = litellm.get_model_info(model)
        except Exception:
            info = {}
        context_limit = info.get("max_input_tokens") or info.get("max_tokens")
        if context_limit:
            budget = min(budget, max(0, context_limit - reserve))
    diagnostic = {
        "backend": "llm",
        "estimated_tokens": estimated,
        "is_estimate": True,
        "estimator": estimator,
        "budget_tokens": budget,
        "output_reserve_tokens": options.get("max_tokens", 0),
        "request_chars": len(json.dumps(body, ensure_ascii=False)),
    }
    if context and context.config.max_request_tokens and estimated > budget:
        record_request({**diagnostic, "outcome": "oversized"})
        raise RequestBudgetExceeded(diagnostic)

    async def read_response():
        response_stream = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout_s,
            stream=True,
            **options,
        )
        async with aclosing(response_stream):
            chunks = [chunk async for chunk in response_stream]
        return litellm.stream_chunk_builder(chunks, messages=messages)

    for attempt in range(1, max_retries + 1):
        try:
            # A socket read timeout alone does not bound an active stream.
            resp = await asyncio.wait_for(read_response(), timeout=timeout_s)
            if resp is None:
                raise ValueError("LLM stream returned no response chunks")
            record_model(resp.get("model"))
            usage = resp.get("usage")
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump()
            record_request(
                {
                    **diagnostic,
                    "outcome": "complete",
                    "attempt": attempt,
                    "usage": numeric_usage(usage),
                }
            )
            content = resp["choices"][0]["message"]["content"] or ""
            content = _extract_json_text(content)
            if not content:
                raise ValueError("LLM returned empty content")
            return json.loads(content)
        except Exception as e:
            if isinstance(e, RequestBudgetExceeded):
                raise
            if (
                is_context_error(str(e))
                or type(e).__name__ == "ContextWindowExceededError"
            ):
                record_request(
                    {
                        **diagnostic,
                        "outcome": "oversized",
                        "reason": "max_tokens_exceeded",
                    }
                )
                raise RequestBudgetExceeded(
                    {**diagnostic, "reason": "max_tokens_exceeded"}
                ) from e
            if isinstance(e, asyncio.TimeoutError):
                record_request({**diagnostic, "outcome": "timeout", "attempt": attempt})
            last_err = e
            logger.warning(
                f"LLM call attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}"
            )
            await asyncio.sleep(min(2**attempt, 30))
    raise LLMError(
        f"LLM call failed after {max_retries} retries: {type(last_err).__name__}: {last_err}"
    ) from last_err


async def _extract_nodes(
    messages: list[dict[str, Any]],
    model: str,
    window: int,
    overlap: int,
    max_nodes_per_window: int,
    *,
    call,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    all_nodes: list[dict[str, Any]] = []
    node_counter = 0
    covered_end = 0

    for start, end in iter_windows(len(messages), window, overlap):
        # Preamble and overlapping messages remain useful context, but each
        # message contributes nodes only from its first extraction window.
        # Otherwise differently worded labels accumulate for the same action.
        first_new = max(start, covered_end)
        covered_end = end
        window_text = format_window(messages, start, end)
        user = (
            f"{PASS_A_INSTRUCTIONS}\n\n"
            f"Message window:\n{window_text}\n\n"
            f"Return at most {max_nodes_per_window} nodes."
        )
        if view_instructions():
            user += "\n\n" + view_instructions()

        out = await call(model=model, system=PASS_A_SYSTEM, user=user)
        nodes = out.get("nodes", [])
        if not isinstance(nodes, list):
            warnings.append(f"Pass A returned non-list nodes for window {start}-{end}.")
            continue

        for n in nodes[:max_nodes_per_window]:
            if not isinstance(n, dict):
                continue
            node_counter += 1
            node_dict = dict(n)
            node_dict["node_id"] = f"N{node_counter}"
            coerced, w = coerce_node(node_dict, messages)
            warnings.extend([f"PassA[{start}-{end}] {x}" for x in w])
            if coerced is None:
                continue
            support = coerced.get("support", [])
            if not isinstance(support, list) or not support:
                warnings.append(f"PassA[{start}-{end}] Removed node without support.")
                continue
            indices = [s.get("msg_idx") for s in support if isinstance(s, dict)]
            if len(indices) != len(support) or any(
                not isinstance(i, int) or isinstance(i, bool) or not start <= i < end
                for i in indices
            ):
                continue
            earliest = min(indices)
            if not first_new <= earliest < end:
                continue
            # The node schema defines time as the earliest supporting message.
            coerced["time"] = earliest
            all_nodes.append(coerced)

    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for n in all_nodes:
        earliest = n.get("time", 0)
        key = (n.get("type"), n.get("text"), earliest)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(n)

    return deduped, warnings


async def _extract_edges(
    messages: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    model: str,
    window: int,
    overlap: int,
    *,
    call,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    all_edges: list[dict[str, Any]] = []
    node_ids = {n["node_id"] for n in nodes if "node_id" in n}

    for start, end in iter_windows(len(messages), window, overlap):
        window_text = format_window(messages, start, end)
        window_nodes = [
            n
            for n in nodes
            if isinstance(n.get("time"), int) and start <= n["time"] < end
        ]
        if len(window_nodes) < 2:
            continue

        node_list = [
            {
                "node_id": n["node_id"],
                "type": n["type"],
                "text": n["text"],
                "time": n["time"],
            }
            for n in window_nodes
        ]
        user = (
            f"{PASS_B_INSTRUCTIONS}\n\n"
            f"Message window:\n{window_text}\n\n"
            f"Nodes:\n{json.dumps(node_list, ensure_ascii=False, indent=2)}"
        )
        if view_instructions():
            user += "\n\n" + view_instructions()

        out = await call(model=model, system=PASS_B_SYSTEM, user=user)
        edges = out.get("edges", [])
        if not isinstance(edges, list):
            warnings.append(f"Pass B returned non-list edges for window {start}-{end}.")
            continue

        for e in edges:
            if not isinstance(e, dict):
                continue
            coerced, w = coerce_edge(e, messages, node_ids)
            warnings.extend([f"PassB[{start}-{end}] {x}" for x in w])
            if coerced is not None:
                all_edges.append(coerced)

    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for e in all_edges:
        key = (e.get("src"), e.get("dst"), e.get("relation"), e.get("time"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return deduped, warnings


class LLMAnnotator:
    """Generative node and edge extraction."""

    def __init__(
        self, *, temperature: float = 0.7, timeout_s: float = 120, max_retries: int = 5
    ):
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def backend(self) -> AnnotationBackend:
        async def call(**kwargs):
            kwargs["temperature"] = self.temperature
            return await llm_json_call_async(
                **kwargs, timeout_s=self.timeout_s, max_retries=self.max_retries
            )

        return AnnotationBackend(
            extract_nodes=partial(_extract_nodes, call=call),
            extract_edges=partial(_extract_edges, call=call),
            provenance={
                "annotator": "llm",
                "temperature": self.temperature,
                "timeout_s": self.timeout_s,
                "max_retries": self.max_retries,
                "node_window_policy": NODE_WINDOW_POLICY,
            },
        )
