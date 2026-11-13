"""Overflow-only excerpts preserve complete catalog records and source evidence."""

import asyncio
import copy
import json
from dataclasses import replace

import httpx
from tests.epistemic_analysis.test_check_pipeline import (
    Services,
    chat_answer,
    native_answer,
    result,
)

from corral.epistemic_analysis.overflow import (
    OVERFLOW_SCOPE,
    catalog_records,
    prepare_overflow,
    tool_excerpt_view,
    view_fragments,
)
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.trace_checks import CheckPipeline, check_definitions


def catalog(count=8, padding=500):
    return [
        {
            "reaction_id": i,
            "smarts": "[C:1]>>[C:1]O",
            "nested": {"note": "braces } and [ and a quoted ' reaction 雪"},
            "details": "z" * padding,
        }
        for i in range(count)
    ]


def inputs(text=None, assistant="Inspect reaction 1."):
    return CheckInput.from_document(
        {
            "messages": [
                {"role": "user", "content": "Plan a synthesis."},
                {"role": "assistant", "content": assistant},
                result(
                    "search_template_catalog_by_criteria",
                    text or repr(catalog()),
                    "catalog",
                ),
                {"role": "assistant", "content": "My final route."},
            ]
        }
    )


def test_complete_catalog_records_and_original_quotes_survive():
    for text in (repr(catalog()), json.dumps(catalog(), ensure_ascii=False)):
        source = inputs(text)
        original = copy.deepcopy(source.snapshot())
        view = tool_excerpt_view(source.trace, cap=100, catalog_keep=3)
        mapping = view.mappings[2]
        assert mapping["catalog_reactions"] == {"available": 8, "retained": 3}
        spans = catalog_records(text)
        for start, end in spans[:3]:
            assert text[start:end] in view.messages[2]["content"]
        assert text[spans[3][0] : spans[3][1]] not in view.messages[2]["content"]
        assert "5 catalog reactions omitted" in view.messages[2]["content"]
        assert view.messages[1] == source.trace.messages[1]
        assert view.messages[3] == source.trace.messages[3]
        assert source.snapshot() == original
        for part in view_fragments(source.trace, view):
            assert (
                part["content"]
                == source.trace.messages[part["msg_idx"]]["content"][
                    part["start"] : part["end"]
                ]
            )
        quote = text[spans[1][0] : spans[1][1]]
        mapped = view.map_support(
            [{"msg_idx": 2, "quote": quote}], source.trace.messages
        )
        assert mapped[0]["start"] == spans[1][0]


def test_small_catalog_and_assistant_observations_are_never_cut():
    source = inputs(repr(catalog(2)), assistant="Observation:" + "a" * 10000)
    view = tool_excerpt_view(source.trace, cap=100)
    assert view.messages == source.trace.messages
    assert not view.coverage["reduced_context"]
    assert catalog_records("__import__('os').system('anything')") == []
    assert catalog_records("[{'reaction_id': 1}") == []


def test_full_native_fit_does_not_preprocess_large_outputs(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            source = inputs()
            prepared = await prepare_overflow(
                source,
                list(check_definitions().values()),
                decision,
                lambda parts: {"messages": parts},
                cap=100,
            )
            assert prepared["full_fits"]
            assert prepared["fits"]
            assert prepared["view"] is None
            assert set(prepared["full_counts"]) == set(check_definitions())
            assert all(path.endswith("tokenize") for path in services.calls)
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_two_reactions_are_kept_when_three_cannot_fit(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        decision.config = replace(decision.config, context_tokens=1000)

        async def count(body):
            coverage = body["state"].get("tool_output_omissions", [])
            keep = coverage[0]["catalog_reactions"]["retained"] if coverage else 8
            return 1200 if keep > 2 else 999

        decision.count = count
        try:
            prepared = await prepare_overflow(
                inputs(),
                list(check_definitions(["cheating_attempt"]).values()),
                decision,
                lambda parts: {"messages": parts},
                cap=100,
            )
            assert not prepared["full_fits"]
            assert prepared["fits"]
            assert prepared["view"]["mappings"][2]["catalog_reactions"]["retained"] == 2
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_pipeline_uses_one_excerpt_request_and_evidence_original_offsets(tmp_path):
    async def exercise():
        source = inputs()
        services = Services(tmp_path)
        decision, evidence = services.clients()
        decision.config = replace(decision.config, context_tokens=1000)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            if request.url.path.endswith("tokenize"):
                count = (
                    100
                    if request.url.path.startswith("/gemma")
                    or body["state"]["scope"] == OVERFLOW_SCOPE
                    else 2000
                )
                return httpx.Response(200, json={"count": count})
            calls.append(body)
            if request.url.path.startswith("/knowline"):
                assert body["state"]["scope"] == OVERFLOW_SCOPE
                assert (
                    body["state"]["excerpt_coverage"]["full_source_windows_assessed"]
                    is False
                )
                return httpx.Response(200, json=native_answer("cheating_attempt"))
            return httpx.Response(
                200,
                json=chat_answer(
                    evidence=[{"message_id": "m2", "quote": "[C:1]>>[C:1]O"}]
                ),
            )

        await decision.client.aclose()
        await evidence.client.aclose()
        decision.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        evidence.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            pipeline = CheckPipeline(
                tmp_path,
                decision,
                evidence,
                windowed=True,
                overflow_tool_output_max_chars=100,
            )
            run = await pipeline.run(source, checks=["cheating_attempt"])
            label = json.loads(
                (
                    tmp_path / "labels" / f"{run['labels']['cheating_attempt']}.json"
                ).read_text()
            )
            assert label["state"] == "requires_review"
            assert label["status"] == "present"
            assert len(label["exchanges"]) == 1
            assert label["window_labels"] == []
            assert len(calls) == 2
            support = json.loads(
                (
                    tmp_path
                    / "evidence"
                    / f"{run['evidence']['cheating_attempt']}.json"
                ).read_text()
            )
            assert support["state"] == "requires_review"
            quote = support["evidence"][0]
            assert (
                source.trace.messages[quote["msg_idx"]]["content"][
                    quote["start"] : quote["end"]
                ]
                == quote["quote"]
            )
            # Completed labels can be reused by the separate evidence worker.
            await pipeline.run(
                source,
                checks=["cheating_attempt"],
                evidence_only=True,
                label_records=run["labels"],
            )
            assert len(calls) == 2
            # CLI evidence-only resumes without an explicit label-record mapping.
            await pipeline.run(source, checks=["cheating_attempt"], evidence_only=True)
            assert len(calls) == 2
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_still_oversized_preserves_assistant_text_and_does_not_scan_windows(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        decision.config = replace(decision.config, context_tokens=100)
        try:
            source = inputs(assistant="A" * 10000)
            run = await CheckPipeline(
                tmp_path, decision, windowed=True, overflow_tool_output_max_chars=100
            ).run(source, checks=["cheating_attempt"])
            label = json.loads(
                (
                    tmp_path / "labels" / f"{run['labels']['cheating_attempt']}.json"
                ).read_text()
            )
            assert label["state"] == "requires_review"
            assert label["status"] == "uncertain"
            assert not label["model_called"]
            assert all(path.endswith("tokenize") for path in services.calls)
            assert not label["exchanges"]
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())
