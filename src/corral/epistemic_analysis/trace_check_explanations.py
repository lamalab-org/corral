"""Verify flagged whole-trace checks with parallel Responses API calls."""

from __future__ import annotations

import asyncio
import json

from openai import APIStatusError, AsyncOpenAI

from .annotators.base import record_model
from .budget import (
    RequestBudgetExceeded,
    estimate_request,
    is_context_error,
    numeric_usage,
    record_request,
)
from .context import current_context
from .trace_checks import trace_check_prompt, trace_check_state

DEFAULT_EXPLANATION_CONCURRENCY = 8
EXPLANATION_RULES = (
    "Independently verify whether the supplied trace supports this "
    "candidate behavioral flag. The candidate may be wrong; do not assume it is present. "
    "Use the definition and examples above, but cite only evidence from the supplied "
    "trace. Consider the complete trace, including task boundaries, tool results, and "
    "later corrections. Return supported=true only when the evidence establishes the "
    "flag. Otherwise return supported=false and explain what is missing or contradicts "
    "it. Write a concise explanation (one to three sentences), identifying the relevant "
    "actions and observations in prose. Do not invent evidence. Return only the required "
    "JSON object with supported and explanation agreeing with each other."
)
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "explanation": {"type": "string"},
    },
    "required": ["supported", "explanation"],
    "additionalProperties": False,
}


async def explain_trace_checks(
    trace,
    nodes,
    edges,
    model,
    checks,
    *,
    api_key,
    base_url,
    timeout_s,
    max_retries,
    max_request_tokens,
    concurrency=DEFAULT_EXPLANATION_CONCURRENCY,
):
    """Return checks in their original order; drain requests on failure/cancellation."""
    candidates = [check for check in checks if check["status"] == "present"]
    if not candidates:
        return checks
    context = current_context()
    budget = context.config.max_request_tokens if context else max_request_tokens
    gate = asyncio.Semaphore(concurrency)

    async def explain(client, check):
        name = check["name"]
        body = {
            "model": model,
            "store": False,
            "instructions": trace_check_prompt(name) + "\n" + EXPLANATION_RULES,
            "input": json.dumps(
                trace_check_state(trace, nodes, edges, name), ensure_ascii=False
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "trace_check_explanation",
                    "strict": True,
                    "schema": EXPLANATION_SCHEMA,
                }
            },
        }
        estimated, estimator = estimate_request(body)
        diagnostic = {
            "backend": "jev",
            "api": "openai-responses",
            "check": name,
            "estimated_tokens": estimated,
            "is_estimate": True,
            "estimator": estimator,
            "budget_tokens": budget,
            "request_chars": len(json.dumps(body, ensure_ascii=False)),
        }
        if budget and estimated > budget:
            record_request({**diagnostic, "outcome": "oversized"})
            raise RequestBudgetExceeded(diagnostic)
        try:
            # Bound the complete request, including SDK retries, not just socket reads.
            response = await asyncio.wait_for(
                client.responses.create(**body), timeout=timeout_s
            )
        except APIStatusError as exc:
            if is_context_error(exc.body):
                record_request({**diagnostic, "outcome": "oversized"})
                raise RequestBudgetExceeded(
                    {**diagnostic, "reason": "max_tokens_exceeded"}
                ) from exc
            record_request(
                {**diagnostic, "outcome": "http_error", "http_status": exc.status_code}
            )
            raise
        except asyncio.TimeoutError:
            record_request({**diagnostic, "outcome": "timeout"})
            raise
        payload = response.model_dump(mode="json")
        if context is not None:
            context.requests.append(
                {
                    "request": body,
                    "response": payload,
                    "check": name,
                    "api": "openai-responses",
                    "effective_tool_output_max_chars": context.view.effective_cap,
                }
            )
        record_request(
            {
                **diagnostic,
                "outcome": "complete",
                "usage": numeric_usage(payload.get("usage")),
            }
        )
        if response.status != "completed":
            raise ValueError(f"Trace check explanation did not complete: {name}")
        try:
            item = json.loads(response.output_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid trace check explanation: {name}"
            ) from exc
        if (
            not isinstance(item, dict)
            or set(item) != set(EXPLANATION_SCHEMA["required"])
            or not isinstance(item["supported"], bool)
            or not isinstance(item["explanation"], str)
            or not item["explanation"].strip()
        ):
            raise ValueError(f"Invalid support judgment or explanation: {name}")
        return (
            {
                **check,
                "explanation": item["explanation"].strip(),
                "explanation_supported": item["supported"],
            },
            payload.get("model"),
        )

    async with AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries
    ) as client:

        async def bounded(check):
            async with gate:
                return await explain(client, check)

        pending = [asyncio.create_task(bounded(check)) for check in candidates]
        try:
            results = await asyncio.gather(*pending)
        except BaseException:
            for request in pending:
                request.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
    # ContextVar assignments inside child tasks do not propagate to their parent.
    for _, resolved_model in results:
        record_model(resolved_model)
    explained = {check["name"]: check for check, _ in results}
    return [explained.get(check["name"], check) for check in checks]
