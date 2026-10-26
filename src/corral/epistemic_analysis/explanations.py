"""Verify Decisions candidates against the supplied scientific trajectory."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from openai import APIConnectionError, APIStatusError

from corral.epistemic_analysis.epistemic_prompts import EXPLANATION_RULES
from corral.epistemic_analysis.motifs import RELATION_DESCRIPTIONS
from corral.epistemic_analysis.ontology import NODE_CRITERIA

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from corral.epistemic_analysis.motifs import Motif


class ExplanationStreamInterrupted(RuntimeError):
    """The transport ended before publishing a terminal response."""


async def _stream_response(
    client: AsyncOpenAI, body: dict[str, Any], *, timeout_s: float, max_retries: int
) -> Any:
    # SDK retries cover opening the request, not failures while consuming SSE.
    # Own the complete attempt here to avoid multiplying the retry budgets.
    request_client = client.with_options(max_retries=0, timeout=timeout_s)

    async def read_response():
        stream = await request_client.responses.create(**body, stream=True)
        async with stream:
            async for event in stream:
                if event.type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }:
                    # Terminal events contain the full output and usage. Never
                    # use a partial JSON delta as an epistemic assessment.
                    return event.response
        raise ExplanationStreamInterrupted("No terminal explanation response")

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(read_response(), timeout=timeout_s)
        except (
            APIConnectionError,
            APIStatusError,
            httpx.TransportError,
            TimeoutError,
            ExplanationStreamInterrupted,
        ) as exc:
            if isinstance(exc, APIStatusError) and not (
                exc.status_code in {408, 409, 429} or exc.status_code >= 500
            ):
                raise
            if attempt == max_retries:
                raise
            logger.warning(
                "Epistemic explanation stream retry {}/{} after {}",
                attempt + 1,
                max_retries,
                type(exc).__name__,
            )
            await asyncio.sleep(min(0.5 * 2**attempt, 8))
    raise AssertionError("unreachable")


async def explain_patterns(
    client: AsyncOpenAI,
    *,
    model: str,
    motifs: list[Motif],
    trajectory: list[dict[str, Any]],
    task: str | list[dict[str, Any]],
    stream: bool = False,
    timeout_s: float = 120,
    max_retries: int = 2,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify candidates with a support judgment and a concrete prose explanation."""
    item_schema = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "explanation": {"type": "string"},
        },
        "required": ["supported", "explanation"],
        "additionalProperties": False,
    }
    names = [m.name for m in motifs]
    body = {
        "model": model,
        "store": False,
        "instructions": EXPLANATION_RULES,
        "input": json.dumps(
            {
                "task": task,
                "messages": trajectory,
                "candidate_patterns": [asdict(m) for m in motifs],
                "node_glossary": NODE_CRITERIA,
                "relationship_glossary": RELATION_DESCRIPTIONS,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "epistemic_pattern_explanations",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": dict.fromkeys(names, item_schema),
                    "required": names,
                    "additionalProperties": False,
                },
            }
        },
    }
    response = (
        await _stream_response(
            client, body, timeout_s=timeout_s, max_retries=max_retries
        )
        if stream
        else await client.responses.create(**body)
    )
    if response.status != "completed":
        raise ValueError("Epistemic explanation response did not complete")
    try:
        explanations = json.loads(response.output_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid epistemic explanations") from exc
    if not isinstance(explanations, dict) or set(explanations) != set(names):
        raise ValueError("Expected exactly one explanation per candidate pattern")
    for name, item in explanations.items():
        if not isinstance(item, dict) or set(item) != set(item_schema["required"]):
            raise ValueError(f"Invalid explanation: {name}")
        reason = item["explanation"]
        if (
            not isinstance(item["supported"], bool)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(f"Invalid support judgment or explanation: {name}")
        item["explanation"] = reason.strip()
    return explanations, response.model_dump(mode="json")
