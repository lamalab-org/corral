"""Cautious late trimming, maximum native windows and positive-any-window labels."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from tests.epistemic_analysis.test_check_pipeline import (
    Services,
    chat_answer,
    native_answer,
)
from tests.epistemic_analysis.test_overflow import catalog, inputs

from corral.epistemic_analysis.overflow import (
    WINDOW_SCOPE,
    long_message_view,
    prepare_overflow,
    tool_excerpt_view,
    view_fragments,
)
from corral.epistemic_analysis.overflow_windows import (
    maximal_windows,
    verify_retained_coverage,
)
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import read_record
from corral.epistemic_analysis.trace_checks import CheckPipeline, check_definitions


def test_repetition_samples_counts_and_original_offsets_survive_late_trim():
    text = "A real response ends here." + "\n" * 62413 + "END OF RESPONSE"
    source = inputs(assistant=text)
    tools = tool_excerpt_view(source.trace, cap=256, catalog_keep=2)
    assert tools.messages[1]["content"] == text
    view = long_message_view(source.trace, tools)
    mapping = view.mappings[1]
    assert mapping["repetition_signals"][0]["repetitions"] == 62413
    assert mapping["repetition_signals"][0]["retained_sample_chars"] == 1024
    assert "\n" * 512 in view.messages[1]["content"]
    assert "A real response ends here." in view.messages[1]["content"]
    assert "END OF RESPONSE" in view.messages[1]["content"]
    assert mapping["message_omitted_chars"] == 62413 - 1024
    assert view.coverage["omitted_message_chars"] == 62413 - 1024
    assert view.coverage["omitted_tool_chars"] == tools.coverage["omitted_tool_chars"]
    assert view.mappings[2] == tools.mappings[2]  # Complete retained reactions survive.
    for part in view_fragments(source.trace, view):
        assert (
            part["content"]
            == source.trace.messages[part["msg_idx"]]["content"][
                part["start"] : part["end"]
            ]
        )


def test_message_trimming_is_after_full_and_tool_fit_checks(tmp_path):
    async def exercise():
        source = inputs(assistant="actual reasoning " + "\n" * 62413)
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=5000)
        observed = []
        mode = "full"

        async def count(body):
            omissions = body["state"].get("tool_output_omissions", [])
            has_long = any(m.get("message_omitted_chars") for m in omissions)
            observed.append((bool(omissions), has_long))
            return (
                100
                if mode == "full" or (mode == "tools" and omissions) or has_long
                else 6000
            )

        decision.count = count
        try:
            definitions = list(check_definitions(["generation_breakdown"]).values())
            for mode in ("full", "tools", "long"):
                observed.clear()
                result = await prepare_overflow(
                    source,
                    definitions,
                    decision,
                    lambda parts: {"messages": parts},
                    cap=256,
                )
                assert (
                    result["stage"]
                    == {
                        "full": "full",
                        "tools": "tool_outputs",
                        "long": "long_messages",
                    }[mode]
                )
                assert result["fits"]
                if mode == "full":
                    assert result["view"] is None
                elif mode == "tools":
                    assert (
                        result["view"]["messages"][1]["content"]
                        == source.trace.messages[1]["content"]
                    )
                else:
                    assert observed.index((True, True)) > observed.index((True, False))
                    assert (
                        result["long_message_cap"] is None
                    )  # Preserve all non-repetitive prose.
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_generic_long_message_keeps_the_largest_fitting_samples(tmp_path):
    async def exercise():
        text = "".join(
            f"Unique record {i:06d}: measured value {i * 17}.\n" for i in range(4000)
        )
        source = CheckInput.from_document(
            {
                "messages": [
                    {"role": "user", "content": "Inspect records."},
                    {"role": "assistant", "content": text},
                ]
            }
        )
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=40000)

        async def count(body):
            return 100 + sum(len(p["content"]) for p in body["state"]["messages"])

        decision.count = count
        try:
            prepared = await prepare_overflow(
                source,
                list(check_definitions(["generation_breakdown"]).values()),
                decision,
                lambda parts: {"messages": parts},
            )
            assert prepared["stage"] == "long_messages"
            assert prepared["long_message_cap"] == 32000
            mapping = prepared["view"]["mappings"][1]
            assert len(mapping["retained_ranges"]) == 3
            assert (
                sum(
                    r["original_end"] - r["original_start"]
                    for r in mapping["retained_ranges"]
                )
                == 32000
            )
            assert mapping["retained_ranges"][0]["original_start"] == 0
            assert mapping["retained_ranges"][-1]["original_end"] == len(text)
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_maximal_windows_fill_across_messages_preserve_unicode_empty_and_omissions():
    async def exercise():
        source = inputs(repr(catalog()), assistant="abcdef雪🙂" * 40)
        source.trace.messages.insert(2, {"role": "assistant", "content": ""})
        source.trace.message_sources.insert(2, {"index": 99})
        view = tool_excerpt_view(source.trace, cap=100, catalog_keep=2)
        retained = view_fragments(source.trace, view)

        async def fits(parts):
            return sum(len(p["content"]) for p in parts) + 5 * len(parts) <= 400

        windows = await maximal_windows(source.trace, retained, fits, overlap_chars=20)
        verify_retained_coverage(source.trace, retained, windows)
        assert len(windows) > 2
        assert any(p["msg_idx"] == 2 for w in windows for p in w)
        for window in windows[:-1]:
            # Can leave at most a new-fragment envelope, rather than wasting the
            # space remaining before the next large message.
            assert (
                395 <= sum(len(p["content"]) for p in window) + 5 * len(window) <= 400
            )
        assert any(p["context_only"] for w in windows for p in w)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("positive", "failure", "expected"),
    [
        (True, False, "present"),
        (True, True, "present"),
        (False, False, "absent"),
        (False, True, "uncertain"),
    ],
)
def test_window_or_preserves_positive_without_model_reconciliation(
    tmp_path, positive, failure, expected
):
    async def exercise():
        source = CheckInput.from_document(
            {
                "messages": [
                    {"role": "user", "content": "Inspect."},
                    *[
                        {
                            "role": "assistant",
                            "content": (f"Observation {i}: " + "abcde" * 100),
                        }
                        for i in range(5)
                    ],
                ]
            }
        )
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=1000, retries=0)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            state = body["state"]
            cost = 100 + sum(len(p["content"]) for p in state["messages"])
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": cost})
            assert state["scope"] == WINDOW_SCOPE
            assert cost + 1 <= 1000
            calls.append(body)
            if failure and len(calls) == 2:
                return httpx.Response(200, json={"bad_response": True})
            status = "present" if positive and len(calls) == 1 else "absent"
            return httpx.Response(
                200,
                json=native_answer(
                    "cheating_attempt",
                    status,
                    {
                        s: 0.8 if s == status else 0.1
                        for s in ("present", "absent", "uncertain")
                    },
                ),
            )

        await decision.client.aclose()
        decision.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            pipeline = CheckPipeline(
                tmp_path,
                decision,
                overflow_tool_output_max_chars=100,
                decision_window_tokens=200,
                reconcile_windowed=True,
            )
            run = await pipeline.run(source, checks=["cheating_attempt"])
            key = run["labels"]["cheating_attempt"]
            label = read_record(tmp_path / "labels" / f"{key}.json", key)
            assert label["status"] == expected
            assert label["decision_source"] == "code_any_window"
            assert label["aggregation_policy"] == "any_window_present"
            assert len(calls) == len(label["window_labels"]) >= 3
            assert label["window_labels"][0]["native_input_tokens"] == 999
            assert "reconciliation" not in label
            await pipeline.run(
                source,
                checks=["cheating_attempt"],
                evidence_only=True,
                label_records=run["labels"],
            )
            assert len(calls) == len(label["window_labels"])
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_window_evidence_uses_positive_windows_and_original_quotes(tmp_path):
    async def exercise():
        source = CheckInput.from_document(
            {
                "messages": [
                    {"role": "user", "content": "Inspect."},
                    {"role": "assistant", "content": "positive proof " + "abcde" * 200},
                    {"role": "assistant", "content": "later negative " + "fghij" * 200},
                ]
            }
        )
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=1000)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            gemma = request.url.path.startswith("/gemma")
            if request.url.path.endswith("tokenize"):
                cost = (
                    100
                    if gemma
                    else 100 + sum(len(p["content"]) for p in body["state"]["messages"])
                )
                return httpx.Response(200, json={"count": cost})
            calls.append(request.url.path)
            if gemma:
                assert "positive proof" in request.content.decode()
                assert "later negative" not in request.content.decode()
                return httpx.Response(
                    200,
                    json=chat_answer(
                        evidence=[{"message_id": "m1", "quote": "positive proof"}]
                    ),
                )
            status = (
                "present"
                if any(
                    "positive proof" in p["content"] for p in body["state"]["messages"]
                )
                else "absent"
            )
            return httpx.Response(
                200,
                json=native_answer(
                    "cheating_attempt",
                    status,
                    {
                        s: 0.8 if s == status else 0.1
                        for s in ("present", "absent", "uncertain")
                    },
                ),
            )

        await decision.client.aclose()
        await evidence.client.aclose()
        decision.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        evidence.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        evidence.api = evidence.api.with_options(http_client=evidence.client)
        try:
            pipeline = CheckPipeline(
                tmp_path, decision, evidence, overflow_tool_output_max_chars=100
            )
            run = await pipeline.run(source, checks=["cheating_attempt"])
            key = run["evidence"]["cheating_attempt"]
            support = read_record(tmp_path / "evidence" / f"{key}.json", key)
            assert support["state"] == "requires_review"
            assert support["evidence"] == []
            assert "positive proof" in support["explanation"]
            before = list(calls)
            await pipeline.run(
                source,
                checks=["cheating_attempt"],
                evidence_only=True,
                label_records=run["labels"],
            )
            assert calls == before
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())
