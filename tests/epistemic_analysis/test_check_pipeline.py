"""Provider, persistence, and policy boundaries for the checks-only pipeline."""

import asyncio
import copy
import json
from dataclasses import asdict, replace

import httpx
import pytest

from corral.epistemic_analysis.annotators.local import (
    KnowLineClient,
    OpenAIClient,
    ProviderConfig,
)
from corral.epistemic_analysis.decision_applicability import (
    ValidationPolicy,
    extract_facts,
)
from corral.epistemic_analysis.decision_units import (
    native_windows,
    verify_coverage,
)
from corral.epistemic_analysis.explanations import read_explanation
from corral.epistemic_analysis.schema import NormalizedTrace, SourceReference
from corral.epistemic_analysis.sources.inputs import CheckInput
from corral.epistemic_analysis.storage import export_review, read_record
from corral.epistemic_analysis.trace_checks import CheckPipeline, check_definitions

CHECK = "insufficient_validation"


def call(name, args, identity, *, actor="agent", branch="main"):
    return {
        "role": "assistant",
        "content": "",
        "agent_run_id": actor,
        "branch_id": branch,
        "tool_calls": [
            {"id": identity, "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def result(name, output, identity, *, actor="agent", branch="main"):
    return {
        "role": "tool",
        "content": output,
        "name": name,
        "tool_call_id": identity,
        "agent_run_id": actor,
        "branch_id": branch,
    }


def saved(messages=None):
    return CheckInput.from_document(
        {
            "completion_status": "complete",
            "reconstruction_status": "complete",
            "score": "SECRET_SCORE",
            "output": {"target": "SECRET_ANSWER"},
            "messages": messages
            or [
                {
                    "role": "user",
                    "content": "Check the submitted structure against measurements.",
                },
                call("validate_smiles", {"smiles": "CCO"}, "validate"),
                result("validate_smiles", "true", "validate"),
                call("submit_answer", {"answer": "CCO"}, "submit"),
                result("submit_answer", "answer accepted", "submit"),
            ],
        }
    )


def provider_config(provider):
    return ProviderConfig(
        provider=provider,
        model=provider,
        revision="pinned-revision",
        endpoint=f"http://test/{provider}/infer",
        token_endpoint=f"http://test/{provider}/tokenize",
        metadata_endpoint=f"http://test/{provider}/metadata",
        output_tokens=1 if provider == "knowline" else 512,
    )


def test_prompt_facts_do_not_repeat_large_normalization_mappings(tmp_path):
    inputs = saved()
    inputs.trace.message_sources[1]["merged_dispatch_indices"] = list(range(1000))
    facts = extract_facts(inputs, ValidationPolicy())
    original = copy.deepcopy(facts)
    source = facts["events"][0]["call_source"]
    assert len(source["source"]["merged_dispatch_indices"]) == 1000
    state = CheckPipeline(tmp_path, None).state(inputs, facts, [])
    projected = state["validation_facts"]["events"][0]["call_source"]
    assert facts == original
    assert projected == {
        k: v for k, v in source.items() if k not in {"quote", "source"}
    }
    assert projected["msg_idx"] == 1
    assert (
        inputs.trace.messages[1]["content"][projected["start"] : projected["end"]]
        == source["quote"]
    )
    assert "merged_dispatch_indices" not in json.dumps(state["validation_facts"])


def test_native_preparation_buffer_requires_verified_server_limit():
    async def exercise(limit):
        config = provider_config("knowline")
        entered = 0
        both_preparing = asyncio.Event()
        release = asyncio.Event()
        backend = {
            "backend": "vllm-native-choice-1",
            "adapter": {"max_concurrency": limit},
        }

        async def handle(request):
            nonlocal entered
            if request.url.path.endswith("metadata"):
                return httpx.Response(
                    200,
                    json={
                        "revision": config.revision,
                        "scoring": config.scoring,
                        "context_tokens": config.context_tokens,
                        "inference_backend": backend,
                    },
                )
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 1})
            entered += 1
            if entered == 2:
                both_preparing.set()
            await release.wait()
            return httpx.Response(200, json={})

        client = KnowLineClient(
            config, preparation_buffer=1, transport=httpx.MockTransport(handle)
        )
        tasks = []
        try:
            with pytest.raises(ValueError, match="requires server preflight"):
                await client.infer({})
            if limit != 1:
                with pytest.raises(ValueError, match="verified native vLLM"):
                    await client.preflight()
                assert entered == 0
                return
            await client.preflight()
            tasks = [asyncio.create_task(client.infer({})) for _ in range(3)]
            await asyncio.wait_for(both_preparing.wait(), 2)
            assert entered == 2  # One inference slot plus one preparation slot.
            release.set()
            exchanges = await asyncio.gather(*tasks)
            assert all(
                x["client_scheduling"]
                == {
                    "preparation_buffer": 1,
                    "request_limit": 2,
                    "server_inference_limit": 1,
                }
                for x in exchanges
            )
            assert client.config == config  # Operational buffering preserves caches.
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.close()

    for limit in [None, 0, 2, 1]:
        asyncio.run(exercise(limit))


def native_answer(name=CHECK, choice="present", probabilities=None):
    return {
        "answers": {
            name: {
                "type": "choice",
                "choice": choice,
                "confidence": 0.8,
                "probabilities": probabilities
                or {"present": 0.8, "absent": 0.1, "uncertain": 0.1},
            }
        }
    }


def chat_answer(assessment="present", evidence=None, finish="stop"):
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": json.dumps(
                        {
                            "assessment": assessment,
                            "explanation": "Independent assessment.",
                            "evidence": evidence
                            if evidence is not None
                            else [
                                {
                                    "message_id": "m0",
                                    "quote": "Check",
                                    "start": 0,
                                    "end": 5,
                                }
                            ],
                        }
                    )
                },
            }
        ]
    }


class Services:
    def __init__(self, directory, *, decision=None, evidence=None):
        self.directory, self.decision, self.evidence = directory, decision, evidence
        self.calls = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request):
        self.calls.append(request.url.path)
        if request.url.path.endswith("tokenize"):
            return httpx.Response(200, json={"count": 100})
        if request.url.path == "/knowline/infer":
            body = json.loads(request.content)
            assert isinstance(body["questions"], dict)  # Native, not remote Decisions.
            assert next(iter(body["questions"].values()))["criteria"].keys() == {
                "present",
                "absent",
                "uncertain",
            }
            assert "SECRET_" not in request.content.decode()
            return httpx.Response(
                200, json=self.decision or native_answer(next(iter(body["questions"])))
            )
        if request.url.path == "/gemma/infer":
            body = json.loads(request.content)
            assert "response_format" not in body
            assert "plain text" in body["messages"][0]["content"]
            labels = list((self.directory / "labels").glob("*.json"))
            assert labels
            assert all(json.loads(p.read_text())["state"] == "complete" for p in labels)
            return httpx.Response(200, json=self.evidence or text_answer())
        raise AssertionError(request.url)

    def clients(self):
        return (
            KnowLineClient(provider_config("knowline"), transport=self.transport),
            OpenAIClient(provider_config("gemma"), transport=self.transport),
        )


def text_answer(
    content="The trace supports the label because a relevant check was omitted.",
    finish="stop",
):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


def test_plain_text_explanation_keeps_text_and_does_not_parse_quotes_or_json():
    text = ' A paraphrase, "not an exact quote". {not valid JSON}\n'
    value = read_explanation(text_answer(text, finish="length"))
    assert value["explanation"] == text
    assert value["state"] == "complete"
    assert value["truncated"]
    assert value["evidence"] == []
    assert value["quote_verification"] == "not_requested"
    assert read_explanation(text_answer(""))["state"] == "failed"


def stage(directory, run, name):
    key = run[name][CHECK]
    return read_record(directory / name / f"{key}.json", key)


def test_native_counts_reused_only_for_identical_ordered_requests(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        body = decision.body(check_definitions([CHECK])[CHECK], {"a": 1, "b": 2})
        try:
            assert await decision.fits(body)
            response = await decision.infer(body)
            assert response["native_input_tokens"] == 100
            assert services.calls.count("/knowline/tokenize") == 1
            body["state"] = {"b": 2, "a": 1}
            await decision.count(body)
            assert services.calls.count("/knowline/tokenize") == 2
            body["state"]["a"] = 3
            await decision.count(body)
            assert services.calls.count("/knowline/tokenize") == 3
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_tokenization_overlaps_inference_without_overlapping_gpu_requests():
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        active, maximum = 0, 0

        async def handle(request):
            nonlocal active, maximum
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 100})
            active += 1
            maximum = max(maximum, active)
            started.set()
            await release.wait()
            active -= 1
            return httpx.Response(200, json=native_answer())

        decision = KnowLineClient(
            provider_config("knowline"), transport=httpx.MockTransport(handle)
        )
        body = decision.body(check_definitions([CHECK])[CHECK], {"trace": "first"})
        first = asyncio.create_task(decision.infer(body))
        try:
            await asyncio.wait_for(started.wait(), 1)
            other = decision.body(
                check_definitions([CHECK])[CHECK], {"trace": "second"}
            )
            assert await asyncio.wait_for(decision.count(other), 1) == 100
            second = asyncio.create_task(decision.infer(other))
            await asyncio.sleep(0)
            assert maximum == 1
            release.set()
            await asyncio.gather(first, second)
            assert maximum == 1
        finally:
            release.set()
            await first
            await decision.close()

    asyncio.run(exercise())


def test_model_owns_insufficient_validation_even_after_matching_last_validator(
    tmp_path,
):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(tmp_path, decision, evidence).run(
                saved(), checks=[CHECK]
            )
            label, support = (
                stage(tmp_path, run, "labels"),
                stage(tmp_path, run, "evidence"),
            )
            assert label["status"] == "present"
            assert label["decision_source"] == "model"
            assert support["state"] == "complete"
            assert run["code_checks"] == []
            assert set(label["probabilities"]) == {"present", "absent", "uncertain"}
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_adding_evaluation_shortcut_reuses_existing_checks_and_saves_evidence(tmp_path):
    async def exercise():
        name = "evaluation_shortcut"
        existing = "cheating_attempt"
        prompt = (
            "Recall the official answer if memorized; otherwise solve very briefly."
        )
        inputs = saved(
            [
                {"role": "user", "content": "Improve a frozen student's inference."},
                call("query_student", {"prompt": prompt}, "probe"),
                result("query_student", "29/64", "probe"),
            ]
        )
        services = Services(
            tmp_path,
            evidence=text_answer(
                "The probe explicitly requests a memorized official answer."
            ),
        )
        decision, evidence = services.clients()
        try:
            pipeline = CheckPipeline(tmp_path, decision, evidence)
            first = await pipeline.run(inputs, checks=[existing])
            expanded = await pipeline.run(inputs, checks=[existing, name])
            assert expanded["state"] == "complete"
            assert expanded["labels"][existing] == first["labels"][existing]
            assert expanded["evidence"][existing] == first["evidence"][existing]
            label_key, evidence_key = (
                expanded["labels"][name],
                expanded["evidence"][name],
            )
            label = read_record(tmp_path / "labels" / f"{label_key}.json", label_key)
            support = read_record(
                tmp_path / "evidence" / f"{evidence_key}.json", evidence_key
            )
            assert label["name"] == name
            assert label["decision_source"] == "model"
            assert support["quote_verification"] == "not_requested"
            assert support["evidence"] == []
            assert (
                support["explanation"]
                == "The probe explicitly requests a memorized official answer."
            )
            assert services.calls.count("/knowline/infer") == 2
            assert services.calls.count("/gemma/infer") == 2

            # Default selection also runs the new check and reuses both saved results.
            all_checks = await pipeline.run(inputs)
            assert all_checks["state"] == "complete"
            assert all_checks["labels"][name] == label_key
            assert all_checks["evidence"][name] == evidence_key
            assert set(all_checks["labels"]) == set(check_definitions())
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_plain_text_has_no_quote_validation_or_repair_retry(tmp_path):
    async def exercise():
        services = Services(
            tmp_path,
            evidence=text_answer(
                "The agent submits without checking the relevant measurements."
            ),
        )
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(
                tmp_path, decision, evidence, repair_attempts=1
            ).run(saved(), checks=[CHECK])
            assert stage(tmp_path, run, "labels")["status"] == "present"
            support = stage(tmp_path, run, "evidence")
            assert support["state"] == "complete"
            assert support["evidence"] == []
            assert services.calls.count("/gemma/infer") == 1
            assert run["state"] == "complete"
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_explanation_does_not_overwrite_label_and_changes_reuse_decisions(
    tmp_path,
):
    async def exercise():
        services = Services(
            tmp_path, evidence=text_answer("The label is not supported by this trace.")
        )
        decision, evidence = services.clients()
        try:
            pipeline = CheckPipeline(tmp_path, decision, evidence)
            first = await pipeline.run(saved(), checks=[CHECK])
            assert stage(tmp_path, first, "labels")["status"] == "present"
            assert stage(tmp_path, first, "evidence")["support_assessment"] is None
            assert (
                stage(tmp_path, first, "evidence")["explanation"]
                == "The label is not supported by this trace."
            )
            evidence.config = replace(evidence.config, revision="new-gemma-revision")
            second = await pipeline.run(saved(), checks=[CHECK], evidence_only=True)
            assert first["labels"] == second["labels"]
            assert first["evidence"] != second["evidence"]
            assert services.calls.count("/knowline/infer") == 1
            assert services.calls.count("/gemma/infer") == 2
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_unknown_and_reviewed_na_are_code_owned_not_empty_tool_inventory(tmp_path):
    inputs = saved([{"role": "assistant", "content": "Unfinished plan to submit."}])
    unknown = ValidationPolicy()
    assert (
        unknown.decision(CHECK, extract_facts(inputs, unknown))["status"] == "uncertain"
    )
    with pytest.raises(ValueError, match="reviewer"):
        ValidationPolicy(applicability="not_applicable")
    policy = ValidationPolicy(
        applicability="not_applicable",
        reviewed_by="reviewer",
        rationale="No submission obligation in this scope.",
        source_evidence=[{"contract": "reviewed-policy"}],
    )

    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(tmp_path, decision).run(
                saved(), policy=policy, checks=[CHECK]
            )
            label = stage(tmp_path, run, "labels")
            assert label["status"] == "not_applicable"
            assert label["decision_source"] == "code"
            assert not services.calls
            inputs = saved()
            inputs.context["configured_tools"] = {
                "available_tools": [],
                "validation_tools": [],
                "complete": True,
            }
            run = await CheckPipeline(tmp_path, decision).run(inputs, checks=[CHECK])
            assert stage(tmp_path, run, "labels")["decision_source"] == "model"
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "response",
    [
        native_answer(choice="not_applicable"),
        native_answer(probabilities={"present": 1, "absent": 1, "uncertain": 1}),
        native_answer(choice="absent"),
    ],
)
def test_invalid_native_labels_fail_closed(tmp_path, response):
    async def exercise():
        services = Services(tmp_path, decision=response)
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(tmp_path, decision, evidence).run(
                saved(), checks=[CHECK]
            )
            label = stage(tmp_path, run, "labels")
            assert label["state"] == "failed"
            assert label["status"] == "uncertain"
            assert run["evidence"] == {}
            assert list((tmp_path / "exchanges/labels").glob("*.json"))
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_fact_timing_candidate_equivalence_and_actor_boundaries():
    messages = [
        {"role": "user", "content": "Check final candidate."},
        call("validate_smiles", {"smiles": "CCO"}, "a"),
        result("validate_smiles", "false", "a"),
        call("validate_smiles", {"smiles": "CO"}, "b"),
        result("validate_smiles", "true", "b"),
        call("submit_answer", {"answer": "CCO"}, "s"),
        result("submit_answer", "answer accepted", "s"),
    ]
    facts = extract_facts(saved(messages), ValidationPolicy())
    assert facts["events"][0]["execution"] == "completed"
    assert facts["events"][0]["verdict"] == "fail"
    assert (
        facts["candidate_relationships"][0]["candidate_relation"]
        == "identical_representation"
    )
    assert facts["candidate_relationships"][0]["available_before_submission"] is True
    assert "status" not in facts
    messages[5] = call("submit_answer", {"answer": "OCC"}, "s")
    facts = extract_facts(saved(messages), ValidationPolicy())
    assert (
        facts["candidate_relationships"][0]["candidate_relation"]
        == "different_representation_equivalence_unknown"
    )
    messages[2]["agent_run_id"] = "other-agent"
    facts = extract_facts(saved(messages), ValidationPolicy())
    assert facts["events"][0]["execution"] == "no_paired_result"
    assert not facts["candidate_relationships"][0]["available_before_submission"]
    assert any(
        item["reason"] == "unpaired_or_ambiguous_result"
        for item in facts["capture_limitations"]
    )


def test_partial_spectra_and_crashes_are_separate_from_behavioral_labels():
    inputs = saved(
        [
            call("simulate_spectra", {"smiles": "C"}, "v"),
            result(
                "simulate_spectra",
                json.dumps({"h_nmr": "1.2 ppm", "ir": "failed"}),
                "v",
            ),
            call("validate_model_syntax", {"syntax": "f =~ x"}, "c"),
            result("validate_model_syntax", '{"error":"outage"}', "c"),
        ]
    )
    facts = extract_facts(inputs, ValidationPolicy())
    assert facts["events"][0]["properties"] == ["h_nmr_prediction"]
    assert facts["events"][0]["verdict"] == "not_a_verdict"
    assert facts["events"][1]["execution"] == "failed"
    assert not facts["final_accepted_submission"]


def test_native_windows_keep_every_unicode_character_and_overlap():
    trace = NormalizedTrace(
        [{"role": "tool", "content": "αβ🌍\n" * 100}], SourceReference("t", "t", "hash")
    )

    async def fits(parts):
        return sum(len(p["content"]) for p in parts) <= 53

    windows = asyncio.run(native_windows(trace, fits, overlap_chars=5))
    assert len(windows) > 2
    verify_coverage(trace, windows)
    assert any(p["context_only"] for w in windows for p in w)
    broken = copy.deepcopy(windows)
    broken[0][0]["content"] = "invented"
    with pytest.raises(ValueError):
        verify_coverage(trace, broken)


def test_stale_cache_and_changed_context_cannot_reuse_labels(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            pipeline = CheckPipeline(tmp_path, decision)
            first = await pipeline.run(saved(), checks=[CHECK])
            changed = saved()
            changed.context["capture_limitations"] = ["missing branch"]
            with pytest.raises(ValueError, match="No reusable label"):
                await pipeline.run(changed, checks=[CHECK], evidence_only=True)
            path = tmp_path / "labels" / f"{first['labels'][CHECK]}.json"
            record = json.loads(path.read_text())
            record["status"] = "absent"
            path.write_text(json.dumps(record))
            with pytest.raises(ValueError, match="content changed"):
                await pipeline.run(saved(), checks=[CHECK])
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_cancellation_persists_label_and_resumes_evidence_without_rescoring(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        started = asyncio.Event()
        original = evidence.infer

        async def blocked(body):
            started.set()
            await asyncio.Event().wait()

        evidence.infer = blocked
        pipeline = CheckPipeline(tmp_path, decision, evidence)
        task = asyncio.create_task(pipeline.run(saved(), checks=[CHECK]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert all(
            json.loads(p.read_text())["state"] == "complete"
            for p in (tmp_path / "labels").glob("*.json")
        )
        assert any(
            json.loads(p.read_text())["state"] == "cancelled"
            for p in (tmp_path / "evidence").glob("*.json")
        )
        evidence.infer = original
        run = await pipeline.run(saved(), checks=[CHECK], evidence_only=True)
        assert run["state"] == "complete"
        assert services.calls.count("/knowline/infer") == 1
        await decision.close()
        await evidence.close()

    asyncio.run(exercise())


def test_review_export_separates_facts_labels_evidence_and_preserves_history(tmp_path):
    pytest.importorskip(
        "corral.epistemic_analysis.viewer",
        reason="Local annotation viewer is not installed",
    )
    from corral.epistemic_analysis.viewer.benchmark import ReviewStore

    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(tmp_path, decision, evidence).run(
                saved(), checks=[CHECK]
            )
            directory = export_review(tmp_path, run, tmp_path / "review")
            store = ReviewStore(directory)
            identifier = next(iter(store.rows))
            assessment = store.assessment("knowline", identifier)["checks"][0]
            assert assessment["probabilities"]["present"] == 0.8
            assert assessment["evidence_record"]["support_assessment"] is None
            assert assessment["evidence_record"]["explanation"]
            assert assessment["validation_facts"]["events"]
            assert store.records() == []
            with pytest.raises(ValueError, match="never replaced"):
                export_review(tmp_path, run, directory)
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_public_inputs_reject_task_configuration_and_preserve_recovered_origin():
    with pytest.raises(ValueError, match="public context"):
        CheckInput(saved().trace, {"scoring_params": {"target": "SECRET"}})
    with pytest.raises(ValueError, match="public text"):
        CheckInput(
            saved().trace, {"recovered_task_context": {"input": {}, "output": "SECRET"}}
        )
    inputs = CheckInput(
        saved().trace,
        {
            "recovered_task_context": {
                "text": "Check measurements.",
                "source_evidence": [{"sha256": "source"}],
            }
        },
    )
    assert "recovered_task_context" in inputs.public_state()["context"]
    assert "SECRET_" not in json.dumps(inputs.public_state())


def test_one_active_definition_and_no_tool_count_rules():
    from corral.epistemic_analysis.epistemic_prompts import TRACE_CHECK_DEFINITIONS

    definitions = check_definitions([CHECK])
    assert set(definitions) == {CHECK}
    assert TRACE_CHECK_DEFINITIONS[CHECK] in definitions[CHECK].instructions
    assert "tool availability alone is insufficient" in definitions[CHECK].instructions
    assert "not_applicable" not in definitions[CHECK].question()["criteria"]
    assert "version" not in asdict(definitions[CHECK])


def test_window_local_positives_cannot_become_whole_trace_labels(tmp_path):
    async def exercise():
        def handler(request):
            body = json.loads(request.content)
            if request.url.path.endswith("tokenize"):
                return httpx.Response(
                    200,
                    json={
                        "count": 10
                        + sum(len(p["content"]) for p in body["state"]["messages"])
                    },
                )
            return httpx.Response(200, json=native_answer())

        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=60),
            transport=httpx.MockTransport(handler),
        )
        inputs = saved([{"role": "tool", "content": "αβ🌍\n" * 100}])
        policy = ValidationPolicy(
            applicability="applicable",
            reviewed_by="reviewer",
            rationale="Reviewed task scope.",
            source_evidence=[{"contract": "task"}],
        )
        try:
            full = await CheckPipeline(tmp_path, decision).run(
                inputs, checks=[CHECK], policy=policy
            )
            assert stage(tmp_path, full, "labels")["state"] == "failed"
            run = await CheckPipeline(tmp_path, decision, windowed=True).run(
                inputs, checks=[CHECK], policy=policy
            )
            label = stage(tmp_path, run, "labels")
            assert label["status"] == "uncertain"
            assert label["state"] == "requires_review"
            assert all(w["status"] == "present" for w in label["window_labels"])
            assert label["reconciliation"]["status"] == "present"
            assert label["coverage"]["all_source_characters"]
        finally:
            await decision.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("window_tokens", "effective_budget"), [(None, 80), (40, 40), (10, 23)]
)
def test_checks_share_windows_that_fit_the_largest_rubric(
    tmp_path, window_tokens, effective_budget
):
    async def exercise():
        names = ["contradiction_ignored", "submission_with_concerns"]
        seen = {name: [] for name in names}

        def handler(request):
            body = json.loads(request.content)
            name = next(iter(body["questions"]))
            parts = body["state"]["messages"]
            count = sum(len(p["content"]) for p in parts)
            count += 20 if name == names[0] else 5
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": count})
            assert count < 80
            if body["state"]["scope"] == "partial_window":
                assert count + 1 <= effective_budget
                seen[name].append(copy.deepcopy(parts))
            return httpx.Response(200, json=native_answer(name=name))

        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=80, concurrency=2),
            transport=httpx.MockTransport(handler),
        )
        inputs = saved([{"role": "tool", "content": "αβ🌍\n" * 60}])
        try:
            pipeline = CheckPipeline(
                tmp_path, decision, windowed=True, decision_window_tokens=window_tokens
            )
            run = await pipeline.run(inputs, checks=names)
            assert len(pipeline._window_tasks) == 1
            assert len(seen[names[0]]) > 1
            assert seen[names[0]] == seen[names[1]]
            verify_coverage(inputs.trace, seen[names[0]])
            for name in names:
                label = read_record(
                    tmp_path / "labels" / f"{run['labels'][name]}.json",
                    run["labels"][name],
                )
                assert label["coverage"]["all_source_characters"]
                assert label["state"] == "requires_review"
            previous = copy.deepcopy(seen)
            await pipeline.run(inputs, checks=names, evidence_only=True)
            assert seen == previous  # Saved labels remain reusable by explanations.
            original_windows = [
                parts
                async for parts in pipeline._window_stream(
                    inputs,
                    extract_facts(inputs, ValidationPolicy()),
                    list(check_definitions(names).values()),
                )
            ]

            # A fresh process can recover the sealed window plan without another
            # round of large native tokenization probes.
            async def no_tokenization(body):
                raise AssertionError("Saved plan was tokenized again")

            decision.count = no_tokenization
            fresh = CheckPipeline(
                tmp_path, decision, windowed=True, decision_window_tokens=window_tokens
            )
            windows = [
                parts
                async for parts in fresh._window_stream(
                    inputs,
                    extract_facts(inputs, ValidationPolicy()),
                    list(check_definitions(names).values()),
                )
            ]
            assert windows == original_windows
            # A corrupt plan must report failure, not strand stream consumers
            # waiting forever for a window-ready event.
            path = next((tmp_path / "window-plans").glob("*.json"))
            corrupt = json.loads(path.read_text())
            corrupt["budget_tokens"] += 1
            path.write_text(json.dumps(corrupt))
            broken = CheckPipeline(
                tmp_path, decision, windowed=True, decision_window_tokens=window_tokens
            )

            async def consume():
                return [
                    parts
                    async for parts in broken._window_stream(
                        inputs,
                        extract_facts(inputs, ValidationPolicy()),
                        list(check_definitions(names).values()),
                    )
                ]

            with pytest.raises(ValueError, match="content changed"):
                await asyncio.wait_for(consume(), 2)
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_preflight_and_token_budget_verify_native_configuration():
    async def exercise():
        calls = []

        async def handler(request):
            calls.append(request.url.path)
            if request.url.path.endswith("metadata"):
                return httpx.Response(
                    200,
                    json={
                        "revision": "changed",
                        "scoring": {},
                        "context_tokens": 131072,
                    },
                )
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 131072})
            raise AssertionError("Overflow must not send inference")

        decision = KnowLineClient(
            provider_config("knowline"), transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(ValueError, match="revision/scoring"):
                await decision.preflight()
            with pytest.raises(ValueError, match="budget exceeded"):
                await decision.infer(
                    decision.body(check_definitions([CHECK])[CHECK], {})
                )
            assert "/knowline/infer" not in calls
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_inference_read_timeouts_are_not_replayed_but_tokenization_can_retry(
    monkeypatch,
):
    async def exercise(tokenization):
        attempts = 0

        async def no_delay(seconds):
            pass

        monkeypatch.setattr(asyncio, "sleep", no_delay)

        async def handler(request):
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("fixture timeout", request=request)

        decision = KnowLineClient(
            provider_config("knowline"),
            request_timeout=7200,
            transport=httpx.MockTransport(handler),
        )
        try:
            assert decision.client.timeout.read == 7200
            assert decision.config.timeout == 900  # Preserve existing cache identity.
            with pytest.raises(httpx.ReadTimeout):
                await decision.request(
                    decision.config.token_endpoint
                    if tokenization
                    else decision.config.endpoint,
                    {},
                )
            assert attempts == (decision.config.retries + 1 if tokenization else 1)
        finally:
            await decision.close()

    asyncio.run(exercise(False))
    asyncio.run(exercise(True))


def test_exhausted_http_error_preserves_server_diagnosis(monkeypatch):
    async def exercise():
        attempts = 0

        async def no_delay(seconds):
            pass

        monkeypatch.setattr(asyncio, "sleep", no_delay)

        async def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(500, json={"error": "CUDA out of memory"})

        decision = KnowLineClient(
            provider_config("knowline"), transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(
                httpx.HTTPStatusError, match="CUDA out of memory"
            ) as caught:
                await decision.request(decision.config.endpoint, {})
            assert attempts == decision.config.retries + 1
            assert caught.value.response.status_code == 500
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_all_selected_labels_are_durable_before_evidence(tmp_path):
    async def exercise():
        services = Services(tmp_path)
        decision, evidence = services.clients()
        try:
            run = await CheckPipeline(tmp_path, decision, evidence).run(
                saved(), checks=[CHECK, "false_reporting"]
            )
            assert len(run["labels"]) == 2
            first_evidence = services.calls.index("/gemma/infer")
            assert services.calls[:first_evidence].count("/knowline/infer") == 2
            assert "/knowline/infer" not in services.calls[first_evidence:]
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_positive_handoff_does_not_wait_for_sibling_checks(tmp_path):
    async def exercise():
        handed_off = asyncio.Event()
        seen = []

        async def handle(request):
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 100})
            name = next(iter(json.loads(request.content)["questions"]))
            if name == "false_reporting":
                await asyncio.wait_for(handed_off.wait(), 2)
            return httpx.Response(200, json=native_answer(name))

        async def handoff(label):
            # Publication happens only after the final label is sealed on disk.
            key = label["fingerprint"]
            assert read_record(tmp_path / "labels" / f"{key}.json", key) == label
            assert label["state"] == "complete"
            seen.append(label["name"])
            handed_off.set()

        decision = KnowLineClient(
            replace(provider_config("knowline"), concurrency=2),
            transport=httpx.MockTransport(handle),
        )
        try:
            await CheckPipeline(tmp_path, decision).run(
                saved(), checks=[CHECK, "false_reporting"], on_label=handoff
            )
            assert seen == [CHECK, "false_reporting"]
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_source_spans_and_result_availability_use_original_timing():
    inputs = saved(
        [
            call("validate_smiles", {"smiles": "CCO"}, "v"),
            call("submit_answer", {"answer": "CCO"}, "s"),
            result("validate_smiles", "true", "v"),
            result("submit_answer", "answer accepted", "s"),
        ]
    )
    facts = extract_facts(inputs, ValidationPolicy())
    assert not facts["candidate_relationships"][0]["available_before_submission"]
    for event in facts["events"]:
        for key in ("call_source", "result_source"):
            source = event[key]
            assert (
                inputs.trace.messages[source["msg_idx"]]["content"][
                    source["start"] : source["end"]
                ]
                == source["quote"]
            )


def test_native_token_endpoint_uses_server_renderer_without_model_execution(
    monkeypatch, tmp_path
):
    import importlib
    import io
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    server = importlib.import_module("serve_knowline_benchmark")

    class Handler:
        def do_GET(self):
            raise AssertionError("Unexpected GET fallback")

        def do_POST(self):
            raise AssertionError("Unexpected POST fallback")

        def _send(self, code, value):
            return code, value

    vendor = SimpleNamespace(Handler=Handler, REQUEST_MAX_BYTES=100000)
    server.install_annotation_endpoints(vendor, tmp_path, 1000, "pinned")
    handler = Handler()
    handler.path = "/tokenize"
    body = json.dumps(
        {"state": {"messages": []}, "questions": {"q": {"type": "choice"}}}
    ).encode()
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.server = SimpleNamespace(
        engine=SimpleNamespace(
            _render=lambda *args: (
                "prefix",
                {"q": ("native rendered prompt", ["present", "absent"])},
                [],
            ),
            processor=SimpleNamespace(
                encode=lambda text, add_special_tokens: list(text)
            ),
        )
    )
    status, response = handler.do_POST()
    assert status == 200
    assert response["count"] == len("native rendered prompt")


def test_empty_explanation_is_not_retried_and_explicit_force_preserves_history(
    tmp_path,
):
    async def exercise():
        services = Services(
            tmp_path,
            evidence=text_answer(""),
        )
        decision, evidence = services.clients()
        pipeline = CheckPipeline(tmp_path, decision, evidence, repair_attempts=0)
        try:
            first = await pipeline.run(saved(), checks=[CHECK])
            assert stage(tmp_path, first, "evidence")["state"] == "failed"
            await pipeline.run(saved(), checks=[CHECK], evidence_only=True)
            assert services.calls.count("/gemma/infer") == 1
            services.evidence = text_answer()
            second = await pipeline.run(
                saved(), checks=[CHECK], evidence_only=True, force_evidence=True
            )
            assert stage(tmp_path, second, "evidence")["state"] == "complete"
            assert services.calls.count("/gemma/infer") == 2
            assert list((tmp_path / "evidence/history").glob("*.json"))
            assert list((tmp_path / "exchanges/evidence/history").glob("*.json"))
            original = stage(tmp_path, second, "labels")
            services.decision = native_answer(
                choice="absent",
                probabilities={"present": 0.1, "absent": 0.8, "uncertain": 0.1},
            )
            third = await pipeline.run(saved(), checks=[CHECK], force_labels=True)
            assert stage(tmp_path, third, "labels")["status"] == "absent"
            assert third["evidence"] == {}
            history = [
                json.loads(p.read_text())
                for p in (tmp_path / "labels/history").glob("*.json")
            ]
            assert original in history
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_model_na_is_rejected_while_frozen_code_na_remains_readable():
    from corral.epistemic_analysis.trace_checks import validate_trace_checks

    record = [{"name": CHECK, "status": "not_applicable", "support": []}]
    assert validate_trace_checks(record, [], expected_names=[CHECK]) == record
    with pytest.raises(ValueError, match="status"):
        validate_trace_checks(record, [], expected_names=[CHECK], allow_code_na=False)


@pytest.mark.parametrize("choice", ["present", "absent", "uncertain"])
def test_whole_conversation_pass_reuses_windows_and_keeps_reduced_coverage(
    tmp_path, choice
):
    name = "contradiction_ignored"
    seen = []

    async def exercise():
        inputs = saved()
        inputs.trace.messages.insert(
            1, {"role": "tool", "content": "head " + "x" * 40000 + " tail"}
        )
        inputs.trace.message_sources.insert(1, {"index": 99})

        def handle(request):
            body = json.loads(request.content)
            if request.url.path == "/knowline/tokenize":
                return httpx.Response(
                    200,
                    json={
                        "count": 10
                        + sum(len(p["content"]) for p in body["state"]["messages"])
                    },
                )
            if request.url.path.endswith("tokenize"):
                return httpx.Response(200, json={"count": 100})
            if request.url.path == "/knowline/infer":
                state = body["state"]
                seen.append(state["scope"])
                if state["scope"] == "whole_conversation_with_tool_excerpts":
                    assert state["excerpt_coverage"]["reduced_context"]
                    assert state["window_assessments"]
                    assert {p["msg_idx"] for p in state["messages"]} == set(
                        range(len(inputs.trace.messages))
                    )
                    for part in state["messages"]:
                        assert (
                            part["content"]
                            == inputs.trace.messages[part["msg_idx"]]["content"][
                                part["start"] : part["end"]
                            ]
                        )
                    assert (
                        "without windows or tool-output clipping"
                        not in body["questions"][name]["instructions"]
                    )
                    return httpx.Response(
                        200,
                        json=native_answer(
                            name,
                            choice,
                            {
                                k: 0.8 if k == choice else 0.1
                                for k in ["present", "absent", "uncertain"]
                            },
                        ),
                    )
                return httpx.Response(200, json=native_answer(name))
            state = json.loads(body["messages"][1]["content"])
            assert state["scope"] == "whole_conversation_with_tool_excerpts"
            return httpx.Response(200, json=chat_answer())

        transport = httpx.MockTransport(handle)
        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=12000),
            transport=transport,
        )
        evidence = OpenAIClient(provider_config("gemma"), transport=transport)
        try:
            original = await CheckPipeline(tmp_path, decision, windowed=True).run(
                inputs, checks=[name]
            )
            original_path = tmp_path / "labels" / f"{original['labels'][name]}.json"
            original_bytes = original_path.read_bytes()
            assert json.loads(original_bytes)["state"] == "requires_review"
            prior_calls = len(seen)
            pipeline = CheckPipeline(
                tmp_path, decision, evidence, windowed=True, reconcile_windowed=True
            )
            run = await pipeline.run(
                inputs, checks=[name], label_records=original["labels"]
            )
            label = read_record(
                tmp_path / "labels" / f"{run['labels'][name]}.json", run["labels"][name]
            )
            assert len(seen) == prior_calls + 1
            assert label["status"] == choice
            assert label["state"] == (
                "requires_review" if choice == "uncertain" else "complete"
            )
            assert label["coverage"]["reduced_context"]
            assert label["coverage"]["full_source_windows_assessed"]
            assert not label["coverage"]["all_source_characters"]
            assert original_path.read_bytes() == original_bytes
            assert bool(run["evidence"]) == (choice == "present")
            if choice == "present":
                result = read_record(
                    tmp_path / "evidence" / f"{run['evidence'][name]}.json",
                    run["evidence"][name],
                )
                assert result["state"] == "complete"
                assert result["coverage"]["reduced_context"]
                assert not result["coverage"]["all_source_characters"]
            await pipeline.run(
                inputs, checks=[name], evidence_only=True, label_records=run["labels"]
            )
            assert len(seen) == prior_calls + 1
            changed = saved()
            with pytest.raises(ValueError, match="Prior label"):
                await pipeline.run(changed, checks=[name], label_records=run["labels"])
        finally:
            await decision.close()
            await evidence.close()

    asyncio.run(exercise())


def test_window_inference_starts_before_all_windows_are_prepared(tmp_path, monkeypatch):
    from corral.epistemic_analysis import trace_checks
    from corral.epistemic_analysis.decision_units import fragment

    async def exercise():
        inferred = asyncio.Event()
        inputs = saved([{"role": "tool", "content": "x" * 100}])

        async def prepare(trace, fits, *, on_window):
            windows = [[fragment(trace, 0, 0, 50)], [fragment(trace, 0, 50, 100)]]
            await on_window(windows[0])
            await asyncio.wait_for(inferred.wait(), 2)
            await on_window(windows[1])
            return windows

        def handle(request):
            body = json.loads(request.content)
            if request.url.path.endswith("tokenize"):
                return httpx.Response(
                    200,
                    json={
                        "count": sum(
                            len(p["content"]) for p in body["state"]["messages"]
                        )
                    },
                )
            inferred.set()
            return httpx.Response(200, json=native_answer("contradiction_ignored"))

        monkeypatch.setattr(trace_checks, "native_windows", prepare)
        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=60),
            transport=httpx.MockTransport(handle),
        )
        try:
            result = await CheckPipeline(tmp_path, decision, windowed=True).run(
                inputs, checks=["contradiction_ignored"]
            )
            assert result["state"] == "requires_review"
            label = read_record(
                tmp_path
                / "labels"
                / f"{result['labels']['contradiction_ignored']}.json",
                result["labels"]["contradiction_ignored"],
            )
            assert len(label["window_labels"]) == 2
        finally:
            await decision.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("share_prefill", [False, True])
def test_failed_later_window_reuses_completed_earlier_window(
    tmp_path, monkeypatch, share_prefill
):
    from corral.epistemic_analysis import trace_checks
    from corral.epistemic_analysis.decision_units import fragment

    async def exercise():
        inputs = saved([{"role": "tool", "content": "x" * 100}])
        calls = []
        fail = True

        async def prepare(trace, fits, *, on_window):
            windows = [[fragment(trace, 0, 0, 50)], [fragment(trace, 0, 50, 100)]]
            for window in windows:
                await on_window(window)
            return windows

        def handle(request):
            body = json.loads(request.content)
            parts = body["state"]["messages"]
            if request.url.path.endswith("tokenize"):
                return httpx.Response(
                    200, json={"count": sum(len(p["content"]) for p in parts)}
                )
            start = parts[0]["start"] if parts else "reconciliation"
            calls.append(start)
            if fail and start == 50:
                raise httpx.ReadTimeout("late window timed out", request=request)
            return httpx.Response(200, json=native_answer("contradiction_ignored"))

        monkeypatch.setattr(trace_checks, "native_windows", prepare)
        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=60),
            transport=httpx.MockTransport(handle),
        )
        try:
            first = await CheckPipeline(
                tmp_path, decision, windowed=True, share_window_prefill=share_prefill
            ).run(inputs, checks=["contradiction_ignored"])
            key = first["labels"]["contradiction_ignored"]
            assert (
                read_record(tmp_path / "labels" / f"{key}.json", key)["state"]
                == "failed"
            )
            assert calls == [0, 50]
            fail = False
            second = await CheckPipeline(
                tmp_path, decision, windowed=True, share_window_prefill=share_prefill
            ).run(inputs, checks=["contradiction_ignored"])
            key = second["labels"]["contradiction_ignored"]
            assert (
                read_record(tmp_path / "labels" / f"{key}.json", key)["state"]
                == "requires_review"
            )
            assert calls == [0, 50, 50, "reconciliation"]
            assert json.loads((tmp_path / "status/labels.json").read_text())[
                "last_success_at"
            ]
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_shared_prefill_warms_windows_without_a_completion_barrier(
    tmp_path, monkeypatch
):
    from corral.epistemic_analysis import trace_checks
    from corral.epistemic_analysis.decision_units import fragment

    async def exercise():
        names = [
            "contradiction_ignored",
            "repeated_failure",
            "cheating_attempt",
            "false_reporting",
        ]
        inputs = saved([{"role": "tool", "content": "x" * 100}])
        warmed, finished, calls = set(), {0: set(), 50: set()}, []
        next_window_started = asyncio.Event()

        async def prepare(trace, fits, *, on_window):
            windows = [[fragment(trace, 0, 0, 50)], [fragment(trace, 0, 50, 100)]]
            for window in windows:
                await on_window(window)
            return windows

        async def handle(request):
            body = json.loads(request.content)
            name = next(iter(body["questions"]))
            parts = body["state"]["messages"]
            if request.url.path.endswith("tokenize"):
                count = (
                    1
                    if name == "false_reporting"
                    else sum(len(p["content"]) for p in parts)
                )
                return httpx.Response(200, json={"count": count})
            calls.append(body)
            if body["state"]["scope"] == "partial_window":
                start = parts[0]["start"]
                if start == 50:
                    next_window_started.set()
                if start == 0 and name == names[1]:
                    await asyncio.wait_for(next_window_started.wait(), 2)
                if name == names[0]:
                    assert start not in warmed
                    await asyncio.sleep(0.01)
                    warmed.add(start)
                else:
                    assert start in warmed, "Repeated cold prefill for the same window"
                finished[start].add(name)
            return httpx.Response(200, json=native_answer(name))

        monkeypatch.setattr(trace_checks, "native_windows", prepare)
        decision = KnowLineClient(
            replace(provider_config("knowline"), context_tokens=60, concurrency=4),
            transport=httpx.MockTransport(handle),
        )
        try:
            run = await asyncio.wait_for(
                CheckPipeline(
                    tmp_path, decision, windowed=True, share_window_prefill=True
                ).run(inputs, checks=names),
                3,
            )
            assert finished[50] == set(names[:3])
            assert run["state"] == "requires_review"
            previous_calls = len(calls)
            resumed = await CheckPipeline(tmp_path, decision, windowed=True).run(
                inputs, checks=names
            )
            assert resumed["fingerprint"] == run["fingerprint"]
            assert resumed["labels"] == run["labels"]
            assert len(calls) == previous_calls
        finally:
            await decision.close()

    asyncio.run(exercise())


def test_failed_prefill_leader_cannot_strand_other_checks():
    from corral.epistemic_analysis.trace_checks import WindowScheduler

    async def exercise():
        scheduler = WindowScheduler(["a", "b"])
        await scheduler.before("a", 0)
        waiting = asyncio.create_task(scheduler.before("b", 0))
        await asyncio.sleep(0)
        assert not waiting.done()
        scheduler.finish("a")
        await asyncio.wait_for(waiting, 1)
        await asyncio.wait_for(scheduler.after("b", 0), 1)
        scheduler.finish("b")

    asyncio.run(exercise())
