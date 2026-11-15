"""Native graph transport, saved-call provenance, coverage and queue barriers."""

import asyncio
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from corral.epistemic_analysis.annotators.graph import GraphExtractor, candidate_spans
from corral.epistemic_analysis.annotators.local import (
    GraphQuestionClient,
    KnowLineClient,
    ProviderConfig,
)
from corral.epistemic_analysis.budget import RequestBudgetExceeded
from corral.epistemic_analysis.context import annotation_context, build_annotation_view
from corral.epistemic_analysis.decision_applicability import SAVED_FIELDS
from corral.epistemic_analysis.pipeline import annotate_trace, can_reuse
from corral.epistemic_analysis.schema import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
    content_hash,
)
from corral.epistemic_analysis.storage import atomic_save


def config():
    return ProviderConfig(
        provider="knowline",
        model="local",
        revision="revision",
        endpoint="http://local/infer",
        token_endpoint="http://local/tokenize",
        metadata_endpoint="http://local/metadata",
    )


def test_jev_graph_uses_the_same_decisions_with_native_transport(tmp_path):
    class Client:
        def __init__(self):
            self.requests = []

        async def evaluate(self, model, state, questions):
            self.requests.append((state, questions))
            answers = {}
            for name, question in questions.items():
                criteria = question["criteria"]
                if "present" in criteria:
                    candidate = state["candidates"][name.rsplit("_", 1)[0]]
                    role = state["messages"][f"m{candidate['msg_idx']}"]["role"]
                    choice = (
                        "present"
                        if name.endswith("_H") and role == "assistant"
                        else "absent"
                    )
                elif "unrelated" in criteria:
                    choice = next(value for value in criteria if value != "unrelated")
                else:
                    choice = "m18"
                answers[name] = {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 1.0,
                    "probabilities": {
                        value: float(value == choice) for value in criteria
                    },
                }
            return answers

    async def exercise():
        messages = [{"role": "user", "content": f"Observation {i}"} for i in range(40)]
        for i in (1, 18, 34):
            messages[i] = {"role": "assistant", "content": f"Hypothesis at step {i}."}
        baseline_client, native_answers = Client(), Client()

        async def handle(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 100})
            assert request.url.path == "/infer"
            body = json.loads(request.content)
            state = body["state"]
            assert state.pop("trace_context") == {"task": "shared context"}
            answers = await native_answers.evaluate(
                body["model"], state, body["questions"]
            )
            return httpx.Response(200, json={"answers": answers})

        provider = KnowLineClient(config(), transport=httpx.MockTransport(handle))
        native_client = GraphQuestionClient(
            provider, tmp_path, {"task": "shared context"}, batch_size=64
        )
        outputs = []
        try:
            for client in (baseline_client, native_client):
                annotator = GraphExtractor(client)
                nodes, warnings = await annotator.extract_nodes(
                    messages, "local", 32, 16, 10000
                )
                assert not warnings
                edges, warnings = await annotator.extract_edges(
                    messages, nodes, "local", 32, 16
                )
                # Raw model metadata differs by transport; the graph must not.
                outputs.append(
                    tuple(
                        [
                            {k: v for k, v in item.items() if k != "jev"}
                            for item in items
                        ]
                        for items in (nodes, edges)
                    )
                )
                assert warnings == [
                    "1 eligible node pairs cross graph windows and were not examined."
                ]
            assert outputs[0] == outputs[1]
            assert len(outputs[0][0]) == 3
            assert outputs[0][1]
            assert [
                (state, questions)
                for state, questions in baseline_client.requests
                if questions
            ] == native_answers.requests
            assert {
                tuple(state["messages"]) for state, _ in native_answers.requests
            } == {
                tuple(f"m{i}" for i in range(32)),
                tuple(f"m{i}" for i in range(16, 40)),
            }
            backend = GraphExtractor(native_client).backend(stop_at_submission=False)
            assert backend.provenance["annotator"] == "graph"
            assert backend.provenance["api"] == "native-systemone"
            assert backend.provenance["provider"]["revision"] == "revision"
            assert backend.provenance["endpoint"] == config().endpoint
            assert "trace_check_explanations" not in backend.provenance
        finally:
            await provider.close()

    asyncio.run(exercise())


def native(messages, function="submit_answer", arguments=None):
    return [
        *messages,
        {
            "role": "assistant",
            "content": "I have a candidate."
            + SAVED_FIELDS
            + json.dumps(
                {
                    "agent_run_id": "agent",
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {
                                "name": function,
                                "arguments": json.dumps(arguments or {"answer": "αβ"}),
                            },
                        }
                    ],
                }
            ),
        },
    ]


def test_native_calls_keep_exact_offsets_and_submission_identity():
    messages = native([{"role": "user", "content": "Task"}])
    spans = candidate_spans(messages, 1200)
    actions = [s for s in spans if s.tool_name == "submit_answer"]
    assert len(actions) == 1
    assert actions[0].tool_name == "submit_answer"
    assert GraphExtractor.submission_span(actions[0])
    assert actions[0].kind == "final"
    assert actions[0].quote == r"\\u03b1\\u03b2"
    assert (
        messages[1]["content"][
            actions[0].offset : actions[0].offset + len(actions[0].quote)
        ]
        == actions[0].quote
    )
    assert not any("agent_run_id" in s.quote for s in spans)
    assert any(s.quote == "I have a candidate." for s in spans)


def test_react_dispatch_metadata_does_not_duplicate_real_actions():
    message = native([], "read_file", {"path": "/workspace/a"})[0]
    message["content"] = message["content"].replace(
        "I have a candidate.",
        '<action>read_file</action><action_input>{"path":"/workspace/a"}</action_input>',
    )
    spans = candidate_spans([message], 1200)
    assert len(spans) == 1
    assert spans[0].quote.startswith("<action>")


def test_escaped_native_arguments_cannot_be_mistaken_for_json_members():
    messages = native(
        [],
        "write_file",
        {"text": 'fake "tool_calls": [{"function": {"name": "submit_answer"}}]'},
    )
    actions = [s for s in candidate_spans(messages, 1200) if s.kind == "action"]
    assert len(actions) == 1
    assert actions[0].tool_name == "write_file"
    assert not GraphExtractor.submission_span(actions[0])


def test_native_transport_batches_caches_and_preserves_wire_protocol(tmp_path):
    async def exercise():
        calls = []

        def handle(request):
            body = json.loads(request.content)
            calls.append((request.url.path, body))
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 50})
            assert "input" not in body
            assert isinstance(body["questions"], dict)
            answers = {
                name: {
                    "type": "choice",
                    "choice": "present",
                    "confidence": 0.2,
                    "probabilities": {"present": 0.9, "absent": 0.1},
                }
                for name in body["questions"]
            }
            return httpx.Response(200, json={"answers": answers, "model": "local"})

        provider = KnowLineClient(config(), transport=httpx.MockTransport(handle))
        client = GraphQuestionClient(
            provider, tmp_path, {"source_hashes": {"messages": "x"}}, batch_size=2
        )
        questions = {
            f"q{i}": {
                "type": "choice",
                "instructions": "Present?",
                "criteria": {"present": "yes", "absent": "no"},
            }
            for i in range(5)
        }
        try:
            answers = await client.evaluate("local", {"messages": {}}, questions)
            assert set(answers) == set(questions)
            assert answers["q0"]["confidence"] == 0.9
            assert answers["q0"]["native_confidence"] == 0.2
            assert [
                len(body["questions"]) for path, body in calls if path == "/infer"
            ] == [2, 2, 1]
            count = len(calls)
            assert (
                await client.evaluate("local", {"messages": {}}, questions) == answers
            )
            assert len(calls) == count
            assert len(list((tmp_path / "states").glob("*.json"))) == 1
            artifact = next((tmp_path / "exchanges").glob("*.json"))
            record = json.loads(artifact.read_text())
            assert "state_ref" in record["request"]
            record["response"]["model"] = "changed"
            artifact.write_text(json.dumps(record))
            with pytest.raises(ValueError, match="content changed"):
                await client.evaluate("local", {"messages": {}}, questions)
        finally:
            await provider.close()

    asyncio.run(exercise())


def test_native_batches_prepare_and_refill_while_an_earlier_request_is_slow(tmp_path):
    async def exercise():
        first_inference = asyncio.Event()
        last_inference = asyncio.Event()
        active = peak = 0

        async def handle(request):
            nonlocal active, peak
            body = json.loads(request.content)
            name = next(iter(body["questions"]))
            if request.url.path == "/tokenize":
                if name == "q2":
                    await asyncio.wait_for(first_inference.wait(), 2)
                return httpx.Response(200, json={"count": 50})
            active += 1
            peak = max(peak, active)
            try:
                if name == "q0":
                    first_inference.set()
                    await asyncio.wait_for(last_inference.wait(), 2)
                if name == "q9":
                    last_inference.set()
                return httpx.Response(
                    200,
                    json={
                        "answers": {
                            name: {
                                "type": "choice",
                                "choice": "present",
                                "confidence": 0.9,
                                "probabilities": {"present": 0.9, "absent": 0.1},
                            }
                        }
                    },
                )
            finally:
                active -= 1

        provider = KnowLineClient(
            replace(config(), concurrency=2), transport=httpx.MockTransport(handle)
        )
        client = GraphQuestionClient(provider, tmp_path, {}, batch_size=1)
        questions = {
            f"q{i}": {
                "type": "choice",
                "instructions": "Present?",
                "criteria": {"present": "yes", "absent": "no"},
            }
            for i in range(10)
        }
        try:
            answers = await client.evaluate("local", {"messages": {}}, questions)
            assert list(answers) == list(questions)
            assert peak == 2
            assert client.completed_questions == 10
            assert len(list((tmp_path / "exchanges").glob("*.json"))) == 10
        finally:
            await provider.close()

    asyncio.run(exercise())


def test_native_context_failure_does_not_truncate_source(tmp_path):
    async def exercise():
        def handle(request):
            assert request.url.path == "/tokenize"
            return httpx.Response(200, json={"count": 999999})

        provider = KnowLineClient(config(), transport=httpx.MockTransport(handle))
        client = GraphQuestionClient(provider, tmp_path, {})
        try:
            with pytest.raises(RequestBudgetExceeded):
                await client.evaluate(
                    "local",
                    {"messages": {"m0": "all source"}},
                    {
                        "q": {
                            "type": "choice",
                            "criteria": {"present": "yes", "absent": "no"},
                        }
                    },
                )
            state = json.loads(next((tmp_path / "states").glob("*.json")).read_text())
            assert state["messages"]["m0"] == "all source"
        finally:
            await provider.close()

    asyncio.run(exercise())


def test_native_backend_retains_messages_after_submit_and_skips_behavioral_pass(
    tmp_path,
):
    class AbsentClient:
        provider = SimpleNamespace(config=config())
        batch_size = 32

        async def evaluate(self, model, state, questions):
            return {
                name: {
                    "type": "choice",
                    "choice": "absent",
                    "confidence": 0.9,
                    "probabilities": {"present": 0.1, "absent": 0.9},
                }
                for name in questions
            }

    async def exercise():
        trace = NormalizedTrace(
            [
                {"role": "user", "content": "Task"},
                {
                    "role": "assistant",
                    "content": '<action>submit_answer</action><action_input>{"answer":"A"}</action_input>',
                },
                {"role": "assistant", "content": "I revise my earlier answer."},
            ],
            SourceReference("t", "t", "hash"),
        )
        annotator = GraphExtractor(AbsentClient()).backend(stop_at_submission=False)
        settings = AnnotationConfig(
            model="local", window=3, overlap=1, tool_output_max_chars=0
        )
        graph = await annotate_trace(trace, annotator, settings)
        assert len(graph.annotation_view["messages"]) == 3
        assert graph.trace_checks == []
        # Missing F is explicitly reviewable, rather than dropping later messages.
        assert graph.annotation_status == "incomplete"
        graph.annotation_status = "complete"
        assert can_reuse(graph, trace, annotator, settings)

    asyncio.run(exercise())


def test_cross_window_edges_remain_explicitly_unexamined(tmp_path):
    class EmptyClient:
        async def evaluate(self, *_args):
            return {}

    async def exercise():
        messages = [{"role": "assistant", "content": f"message {i}"} for i in range(4)]
        nodes = [
            {"node_id": "a", "type": "H", "time": 0, "text": "claim", "support": []},
            {
                "node_id": "b",
                "type": "J",
                "time": 3,
                "text": "interpretation",
                "support": [],
            },
        ]
        trace = NormalizedTrace(messages, SourceReference("t", "t", "hash"))
        settings = AnnotationConfig(model="local")
        diagnostics, issues = [], []
        with annotation_context(
            build_annotation_view(trace, settings), settings, diagnostics, [], issues
        ):
            edges, _ = await GraphExtractor(EmptyClient()).extract_edges(
                messages, nodes, "local", 2, 1
            )
        assert edges == []
        assert diagnostics[-1]["unexamined_node_pairs"] > 0
        assert issues

    asyncio.run(exercise())


def test_graph_queue_refills_before_slow_trace_finishes(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/run_graph_annotation_batch.py"
    spec = importlib.util.spec_from_file_location("rolling_graph_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = {"replicas": {"0": {}}}
    atomic_save(tmp_path / "profile.json", profile)
    atomic_save(
        tmp_path / "queue-request.json",
        {
            "execution_enabled": True,
            "method": "knowline",
            "predecessor": None,
            "profile_fingerprint": content_hash(profile),
            "selection": [{"id": str(i), "gpu": 0} for i in range(8)],
        },
    )
    closed = []

    class Client:
        def __init__(self, config):
            pass

        async def preflight(self):
            pass

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "verify_code", lambda root, manifest=None: None)
    monkeypatch.setattr(module, "ProviderConfig", lambda **kw: kw)
    monkeypatch.setattr(module, "create_provider", Client)

    async def exercise():
        last_started = asyncio.Event()
        seen = []

        async def process(
            root, job, gpu, provider, profile, configuration, request, progress
        ):
            seen.append(job["id"])
            progress({"completed_questions": 1})
            if job["id"] == "0":
                await asyncio.wait_for(last_started.wait(), 2)
            if job["id"] == "7":
                last_started.set()
            atomic_save(root / "completed" / f"{job['id']}.json", {"state": "complete"})

        monkeypatch.setattr(module, "process_job", process)
        await module.run(tmp_path, 0)
        assert seen == [str(i) for i in range(8)]
        assert closed == [True]
        assert (
            json.loads((tmp_path / "status/gpu-0.json").read_text())["state"]
            == "finished"
        )

    asyncio.run(exercise())


def test_queue_waits_for_both_stages_and_matching_inputs(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/run_graph_annotation_batch.py"
    spec = importlib.util.spec_from_file_location("graph_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "manifest.json").write_text("{}")
    request = {
        "predecessor": {
            "directory": str(tmp_path),
            "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
            "required_stages": ["labels", "evidence"],
            "accepted_terminal_states": ["complete", "requires_review"],
        },
        "selection": [{"id": "t", "input_fingerprint": "input"}],
    }
    assert not module.predecessor_status(request)["ready"]
    atomic_save(
        tmp_path / "completed/labels/t.json",
        {"state": "complete", "input_fingerprint": "input"},
    )
    assert not module.predecessor_status(request)["ready"]
    atomic_save(
        tmp_path / "completed/evidence/t.json",
        {"state": "requires_review", "input_fingerprint": "input"},
    )
    assert module.predecessor_status(request)["ready"]
    atomic_save(
        tmp_path / "completed/evidence/t.json",
        {"state": "failed", "input_fingerprint": "input"},
    )
    assert not module.predecessor_status(request)["ready"]
    atomic_save(
        tmp_path / "completed/evidence/t.json",
        {"state": "complete", "input_fingerprint": "changed"},
    )
    with pytest.raises(ValueError, match="fingerprint changed"):
        module.predecessor_status(request)
