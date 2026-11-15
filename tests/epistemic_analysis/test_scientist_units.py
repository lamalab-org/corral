"""Independent scientist labels, source partitions, and exact embedded excerpts."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from tests.epistemic_analysis.test_check_pipeline import (
    Services,
    chat_answer,
    native_answer,
    provider_config,
)
from tests.epistemic_analysis.test_overflow import catalog

from corral.epistemic_analysis.annotation_transcript import TASK_PROMPT_ORIGIN
from corral.epistemic_analysis.annotators.local import KnowLineClient
from corral.epistemic_analysis.overflow import (
    OVERFLOW_SCOPE,
    WINDOW_SCOPE,
    embedded_tool_results,
    tool_excerpt_view,
    view_fragments,
)
from corral.epistemic_analysis.schema import (
    GroupingMetadata,
    NormalizedTrace,
    SourceReference,
)
from corral.epistemic_analysis.scientist_units import scientist_units
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import export_review, read_record
from corral.epistemic_analysis.trace_checks import CheckPipeline

CHECK = "cheating_attempt"


def source(rows):
    return CheckInput(
        NormalizedTrace(
            messages=[
                {"role": "user", "content": "Plan a synthesis."},
                *[{"role": role, "content": text} for _, _, role, text in rows],
            ],
            message_sources=[
                {"origin": TASK_PROMPT_ORIGIN},
                *[{"agent_run_id": actor, "name": name} for actor, name, _, _ in rows],
            ],
            source=SourceReference("scientist", "test", "source-hash"),
            grouping=GroupingMetadata(agent_model="ai-scientist__test"),
        ),
        {},
    )


def observation(value):
    return {
        "action_index": 0,
        "purpose": "Consult catalog",
        "tool_name": "search_template_catalog_by_criteria",
        "arguments": {"query": "KEEP ARGUMENT"},
        "success": True,
        "result": value,
        "error": None,
    }


def test_scientist_units_roll_and_keep_the_same_aggregate_as_serial(
    tmp_path, monkeypatch
):
    from corral.epistemic_analysis import scientist_units as module

    async def exercise():
        last_started = asyncio.Event()
        inputs = source(
            [
                ("a", "task_formulation", "assistant", "MAIN"),
                *[
                    ("a", f"experiment_node_{i}_step_0", "assistant", f"UNIT{i}")
                    for i in range(1, 8)
                ],
            ]
        )

        async def handle(request):
            body = json.loads(request.content)
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 100})
            contents = " ".join(m["content"] for m in body["state"]["messages"])
            if "MAIN" in contents:
                await asyncio.wait_for(last_started.wait(), 2)
            if "UNIT7" in contents:
                last_started.set()
            return httpx.Response(200, json=native_answer(CHECK, "absent"))

        decision = KnowLineClient(
            replace(provider_config("knowline"), concurrency=2),
            transport=httpx.MockTransport(handle),
        )
        try:
            rolled = await CheckPipeline(tmp_path / "rolling", decision).run(
                inputs, checks=[CHECK]
            )
            rolling_map = module.rolling_map

            async def serial(function, items, *, concurrency):
                return await rolling_map(function, items, concurrency=1)

            monkeypatch.setattr(module, "rolling_map", serial)
            serial_run = await CheckPipeline(tmp_path / "serial", decision).run(
                inputs, checks=[CHECK]
            )
            assert rolled["labels"] == serial_run["labels"]
            assert [u["id"] for u in rolled["units"]] == [
                u["id"] for u in serial_run["units"]
            ]
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_partition_keeps_actor_identity_shared_prompt_and_all_observed_messages():
    inputs = source(
        [
            ("a", "task_formulation", "assistant", "Main A"),
            ("a", "experiment_node_1_step_0", "user", "Node A"),
            ("b", "experiment_node_1_step_0", "user", "Node B"),
            ("a", "evaluate_visual_node_1", "assistant", "Critic A"),
            ("b", "final_synthesis", "assistant", "Main B"),
        ]
    )
    units = scientist_units(inputs)
    assert len(units) == 4
    assert len({u["id"] for u in units}) == 4
    assert sorted(i for u in units for i in u["observed_indices"]) == list(range(1, 6))
    assert all(u["indices"][0] == 0 for u in units)
    node = next(
        u for u in units if u["agent_run_id"] == "a" and u["node_id"] == "node_1"
    )
    assert node["observed_indices"] == [2, 4]
    assert [s["parent_msg_idx"] for s in node["inputs"].trace.message_sources] == [
        0,
        2,
        4,
    ]
    main_only = scientist_units(
        source([("a", "task_formulation", "user", "Retries" * 1000)])
    )
    assert len(main_only) == 1
    assert main_only[0]["kind"] == "main"


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_embedded_results_trim_complete_catalogs_with_exact_json_source_offsets(
    ensure_ascii,
):
    results = [repr(catalog()) + "\n雪🙂", 'A quoted "value" 雪🙂\n' * 300]
    text = (
        "KEEP INSTRUCTIONS\n"
        + json.dumps([observation(r) for r in results], ensure_ascii=ensure_ascii)
        + "\nKEEP REASONING"
    )
    parsed = list(embedded_tool_results(text))
    assert len(parsed) == 2
    for original, (decoded, offsets) in zip(results, parsed, strict=True):
        assert decoded == original
        assert json.loads('"' + text[offsets[0] : offsets[-1]] + '"') == original
    inputs = source([("a", "experiment_node_1_step_0", "user", text)])
    unit = scientist_units(inputs)[0]["inputs"]
    # A worker's startup prompt is preserved even when it embeds a catalog.
    assert tool_excerpt_view(unit.trace, cap=100).messages[1]["content"] == text
    unit.trace.messages.insert(1, {"role": "assistant", "content": "Consult catalog."})
    unit.trace.message_sources.insert(1, {"index": 99})
    view = tool_excerpt_view(unit.trace, cap=100)
    mapping = view.mappings[2]
    assert mapping["embedded_catalog_reactions"] == [{"available": 8, "retained": 3}]
    assert mapping["omitted_chars"] > 0
    assert "KEEP INSTRUCTIONS" in view.messages[2]["content"]
    assert "KEEP REASONING" in view.messages[2]["content"]
    assert "KEEP ARGUMENT" in view.messages[2]["content"]
    assert "'reaction_id': 2" in view.messages[2]["content"]
    assert "'reaction_id': 3" not in view.messages[2]["content"]
    for part in view_fragments(unit.trace, view):
        assert (
            part["content"]
            == unit.trace.messages[part["msg_idx"]]["content"][
                part["start"] : part["end"]
            ]
        )
    # An assistant quoting an observation remains reasoning, not a tool result.
    unit.trace.messages[2]["role"] = "assistant"
    assert tool_excerpt_view(unit.trace, cap=100).messages[2]["content"] == text


@pytest.mark.parametrize(
    ("main_status", "node_status", "expected"),
    [
        ("absent", "present", "present"),
        ("present", "absent", "present"),
        ("absent", "absent", "absent"),
        ("absent", "uncertain", "uncertain"),
        ("present", "uncertain", "present"),
    ],
)
def test_independent_labels_code_or_and_evidence_parent_offsets(
    tmp_path, main_status, node_status, expected
):
    async def exercise():
        inputs = source(
            [
                ("a", "task_formulation", "assistant", "MAIN text"),
                ("a", "experiment_node_1_step_0", "assistant", "NODE text"),
            ]
        )
        services = Services(tmp_path)
        decision, evidence = services.clients()
        calls = []

        def handle(request):
            body = json.loads(request.content)
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 100})
            calls.append(request.url.path)
            if request.url.path.startswith("/knowline"):
                contents = " ".join(m["content"] for m in body["state"]["messages"])
                assert ("MAIN text" in contents) != ("NODE text" in contents)
                status = main_status if "MAIN text" in contents else node_status
                return httpx.Response(
                    200,
                    json=native_answer(
                        CHECK,
                        status,
                        {
                            s: 0.8 if s == status else 0.1
                            for s in ("present", "absent", "uncertain")
                        },
                    ),
                )
            # Every unit label and the aggregate must be sealed before evidence.
            assert (
                len(list((tmp_path / "scientist-units").glob("*/*/labels/*.json"))) == 2
            )
            assert len(list((tmp_path / "labels").glob("*.json"))) == 1
            text = (
                "MAIN text" if "MAIN text" in request.content.decode() else "NODE text"
            )
            return httpx.Response(
                200, json=chat_answer(evidence=[{"message_id": "m1", "quote": text}])
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
            run = await pipeline.run(inputs, checks=[CHECK])
            key = run["labels"][CHECK]
            label = read_record(tmp_path / "labels" / f"{key}.json", key)
            assert label["status"] == expected
            assert label["decision_source"] == "code_any_unit"
            assert not label["model_called"]
            assert calls.count("/knowline/infer") == 2
            if expected == "present":
                key = run["evidence"][CHECK]
                support = read_record(tmp_path / "evidence" / f"{key}.json", key)
                assert support["state"] == "complete"
                assert support["evidence"] == []
                for unit in support["unit_evidence"]:
                    record = next(
                        (tmp_path / "scientist-units").glob(
                            f"*/{unit['unit_id']}/evidence/{unit['record']}.json"
                        )
                    )
                    explanation = json.loads(record.read_text())["explanation"]
                    assert "MAIN text" in explanation or "NODE text" in explanation
            before = list(calls)
            await pipeline.run(
                inputs, checks=[CHECK], evidence_only=True, label_records=run["labels"]
            )
            assert calls == before
            await pipeline.run(inputs, checks=[CHECK], evidence_only=True)
            assert calls == before
            export_review(tmp_path, run, tmp_path / "review")
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_only_oversized_unit_is_trimmed_before_independent_label(tmp_path):
    async def exercise():
        inputs = source(
            [
                ("a", "task_formulation", "assistant", "MAIN text"),
                (
                    "a",
                    "experiment_node_1_step_0",
                    "user",
                    json.dumps(observation(repr(catalog()))),
                ),
            ]
        )
        services = Services(tmp_path)
        decision, evidence = services.clients()
        decision.config = replace(decision.config, context_tokens=1000)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            state = body["state"]
            main = any("MAIN text" in m["content"] for m in state["messages"])
            trimmed = state["scope"] == OVERFLOW_SCOPE
            if request.url.path.endswith("tokenize"):
                return httpx.Response(
                    200, json={"count": 100 if main or trimmed else 2000}
                )
            calls.append(state)
            assert main != trimmed
            if trimmed:
                assert "'reaction_id': 3" not in json.dumps(state)
                assert len(state["initial_instructions"]) > 1
            return httpx.Response(200, json=native_answer(CHECK))

        await decision.client.aclose()
        decision.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            run = await CheckPipeline(
                tmp_path, decision, windowed=True, overflow_tool_output_max_chars=100
            ).run(inputs, checks=[CHECK])
            assert len(calls) == 2
            assert sum(s["scope"] == OVERFLOW_SCOPE for s in calls) == 1
            key = run["labels"][CHECK]
            label = read_record(tmp_path / "labels" / f"{key}.json", key)
            assert label["status"] == "present"
            assert label["state"] == "requires_review"
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_omitted_followup_tool_payload_is_not_sent_to_explainer(tmp_path):
    async def exercise():
        reactions = catalog()
        omitted = "OMITTED_ONLY_EVIDENCE"
        reactions[4]["details"] = omitted
        inputs = source(
            [
                ("a", "experiment_node_1_step_0", "assistant", "Consult the catalog."),
                (
                    "a",
                    "experiment_node_1_step_0",
                    "user",
                    json.dumps(observation(repr(reactions))),
                ),
                ("a", "experiment_node_1_step_0", "assistant", "Finished the node."),
            ]
        )
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=1000)
        evidence_requests = []

        def handle(request):
            body = json.loads(request.content)
            gemma = request.url.path.startswith("/gemma")
            if request.url.path.endswith("tokenize"):
                count = (
                    100 if gemma or body["state"]["scope"] == OVERFLOW_SCOPE else 2000
                )
                return httpx.Response(200, json={"count": count})
            assert omitted not in request.content.decode()
            if not gemma:
                return httpx.Response(200, json=native_answer(CHECK))
            evidence_requests.append(body)
            return httpx.Response(
                200, json=chat_answer(evidence=[{"message_id": "m2", "quote": omitted}])
            )

        await decision.client.aclose()
        await evidence.client.aclose()
        decision.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        evidence.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        evidence.api = evidence.api.with_options(http_client=evidence.client)
        try:
            run = await CheckPipeline(
                tmp_path,
                decision,
                evidence,
                repair_attempts=0,
                overflow_tool_output_max_chars=100,
            ).run(inputs, checks=[CHECK])
            assert len(evidence_requests) == 1
            key = run["evidence"][CHECK]
            support = read_record(tmp_path / "evidence" / f"{key}.json", key)
            assert support["state"] == "requires_review"
            assert support["evidence"] == []
            assert support["unit_evidence"][0]["state"] == "requires_review"
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_positive_window_in_one_node_makes_scientist_aggregate_positive(tmp_path):
    async def exercise():
        inputs = source(
            [
                ("a", "task_formulation", "assistant", "MAIN absent"),
                *[
                    (
                        "a",
                        "experiment_node_1_step_0",
                        "assistant",
                        f"NODE part {i} " + "abcde" * 100,
                    )
                    for i in range(4)
                ],
            ]
        )
        decision, evidence = Services(tmp_path).clients()
        decision.config = replace(decision.config, context_tokens=1000)
        calls = []

        def handle(request):
            body = json.loads(request.content)
            state = body["state"]
            if request.url.path.endswith("tokenize"):
                return httpx.Response(
                    200,
                    json={
                        "count": 100 + sum(len(p["content"]) for p in state["messages"])
                    },
                )
            kinds = {p["source"]["scientist_unit"]["kind"] for p in state["messages"]}
            assert len(kinds) == 1
            calls.append(state)
            present = state["scope"] == WINDOW_SCOPE and any(
                "NODE part 2" in p["content"] for p in state["messages"]
            )
            status = "present" if present else "absent"
            return httpx.Response(
                200,
                json=native_answer(
                    CHECK,
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
            run = await CheckPipeline(
                tmp_path, decision, overflow_tool_output_max_chars=100
            ).run(inputs, checks=[CHECK])
            key = run["labels"][CHECK]
            aggregate = read_record(tmp_path / "labels" / f"{key}.json", key)
            assert aggregate["status"] == "present"
            assert aggregate["decision_source"] == "code_any_unit"
            assert len(aggregate["unit_assessments"]) == 2
            assert {a["status"] for a in aggregate["unit_assessments"]} == {
                "present",
                "absent",
            }
            assert sum(s["scope"] == WINDOW_SCOPE for s in calls) >= 3
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())
