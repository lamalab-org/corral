"""One graph workflow with native scoring or OpenAI-compatible JSON choices."""

import asyncio
import json
from dataclasses import asdict

import httpx
import pytest

from corral.epistemic_analysis.annotators.graph import AnnotationError, GraphExtractor
from corral.epistemic_analysis.annotators.local import (
    DEFAULT_ANNOTATION_MODEL,
    DEFAULT_CHAT_MODEL,
    GraphAnnotator,
    GraphQuestionClient,
    KnowLineClient,
    OpenAIClient,
    ProviderConfig,
    provider_config,
)
from corral.epistemic_analysis.explanations import explain_patterns
from corral.epistemic_analysis.motifs import MOTIFS
from corral.epistemic_analysis.pipeline import annotate_trace
from corral.epistemic_analysis.schema import (
    AnnotationConfig,
    NormalizedTrace,
    SourceReference,
)


def chat_response(content, *, finish="stop", usage=None):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 1,
        "model": "custom",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": usage
        or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def chat_config(**kwargs):
    return ProviderConfig(
        provider="openai",
        model="custom",
        base_url="http://example.test/v1",
        output_tokens=1000,
        retries=0,
        **kwargs,
    )


def test_defaults_are_local_without_a_deployment_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORRAL_ANNOTATION_PROFILE", raising=False)
    annotation = provider_config()
    assert annotation.model == DEFAULT_ANNOTATION_MODEL
    assert annotation.endpoint.startswith("http://127.0.0.1:")
    for role in ("explanation", "actor"):
        config = provider_config(role)
        assert config.model == DEFAULT_CHAT_MODEL
        assert config.provider == "openai"
        assert config.endpoint == "http://127.0.0.1:8790/v1/chat/completions"


def test_custom_endpoint_does_not_inherit_local_metadata_or_revision():
    config = provider_config(
        "annotation",
        {
            "provider": "openai",
            "model": "replacement",
            "base_url": "https://example.test/v1",
            "api_key_env": "ANNOTATION_TEST_KEY",
        },
    )
    assert config.endpoint == "https://example.test/v1/chat/completions"
    assert not config.token_endpoint
    assert not config.metadata_endpoint
    assert not config.revision
    assert config.extra_body == {}
    assert config.output_tokens > 1


def test_shared_profile_and_explicit_overrides(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                role: asdict(chat_config())
                for role in ("annotation", "explanation", "actor")
            }
        )
    )
    annotator = GraphAnnotator(profile=profile, model="override", max_retries=0)
    assert annotator.model == "override"
    assert annotator.explanation_config.model == "custom"
    assert annotator.provider_config.retries == 0
    assert annotator.config == AnnotationConfig("override")


def test_same_questions_produce_same_graph_for_both_protocols(tmp_path):
    requests = {"knowline": [], "openai": []}

    def answer(state, questions):
        answers = {}
        for key in questions:
            if key.startswith("same_"):
                choice = "distinct"
            elif key.startswith("claim_"):
                choice = "single"
            elif key.startswith("edge_"):
                choice = "unrelated"
            else:
                span, node_type = key.rsplit("_", 1)
                candidate = state["candidates"][span]
                expected = "H" if candidate["kind"] == "prose" else None
                choice = "present" if node_type == expected else "absent"
            answers[key] = {"choice": choice, "confidence": 1.0}
        return answers

    async def run(kind):
        def handle(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 30})
            body = json.loads(request.content)
            if kind == "knowline":
                state, questions = body["state"], body["questions"]
            else:
                assert request.url.path == "/v1/chat/completions"
                parsed = json.loads(body["messages"][1]["content"])
                state, questions = parsed["state"], parsed["questions"]
                assert "chat_template_kwargs" not in body
            assert not any(key in {m.name for m in MOTIFS} for key in questions)
            requests[kind].append((state, questions))
            answers = answer(state, questions)
            if kind == "knowline":
                payload = {
                    "answers": {
                        key: {
                            "type": "choice",
                            **value,
                            "probabilities": {
                                c: float(c == value["choice"])
                                for c in questions[key]["criteria"]
                            },
                        }
                        for key, value in answers.items()
                    }
                }
            else:
                payload = chat_response(json.dumps(answers))
            return httpx.Response(200, json=payload)

        config = (
            ProviderConfig(
                provider="knowline",
                model="custom",
                endpoint="http://example.test/infer",
                token_endpoint="http://example.test/tokenize",
                retries=0,
            )
            if kind == "knowline"
            else chat_config()
        )
        client = (KnowLineClient if kind == "knowline" else OpenAIClient)(
            config, transport=httpx.MockTransport(handle)
        )
        try:
            questions = GraphQuestionClient(client, tmp_path / kind, {})
            graph = await annotate_trace(
                NormalizedTrace(
                    [{"role": "assistant", "content": "The resistor may be 100 ohms."}],
                    SourceReference("x", "x", "x"),
                ),
                GraphExtractor(questions).backend(),
                AnnotationConfig("custom"),
            )
            assert graph.annotation_status == "complete"
            assert questions.provenance["confidence_readout"] == (
                "maximum-option-probability"
                if kind == "knowline"
                else "model-self-assessment"
            )
            return [
                {k: v for k, v in node.items() if k != "jev"} for node in graph.nodes
            ], graph.edges
        finally:
            await client.close()

    native = asyncio.run(run("knowline"))
    generic = asyncio.run(run("openai"))
    assert native == generic
    assert requests["knowline"] == requests["openai"]


def test_sdk_authentication_and_no_secret_in_saved_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNOTATION_TEST_KEY", "secret-test-value")
    config = chat_config(api_key_env="ANNOTATION_TEST_KEY")

    def handle(request):
        assert request.headers["authorization"] == "Bearer secret-test-value"
        return httpx.Response(
            200, json=chat_response('{"q":{"choice":"yes","confidence":0.8}}')
        )

    async def run():
        client = OpenAIClient(config, transport=httpx.MockTransport(handle))
        try:
            questions = GraphQuestionClient(client, tmp_path, {})
            result = await questions.evaluate(
                "custom",
                {},
                {"q": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}},
            )
            assert result["q"]["confidence"] == 0.8
            assert "probabilities" not in result["q"]
        finally:
            await client.close()

    asyncio.run(run())
    saved = json.loads(next((tmp_path / "exchanges").glob("*.json")).read_text())
    assert saved["estimated_input_tokens"] > 0
    assert "native_input_tokens" not in saved
    assert "secret-test-value" not in "".join(
        path.read_text() for path in tmp_path.rglob("*.json")
    )


@pytest.mark.parametrize(
    ("content", "finish"),
    [
        ('{"q":{"choice":"bad","confidence":1}}', "stop"),
        ("{}", "stop"),
        ("not JSON", "stop"),
        ("{}", "length"),
    ],
)
def test_invalid_or_incomplete_chat_annotations_fail(content, finish):
    async def run():
        client = OpenAIClient(
            chat_config(),
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json=chat_response(content, finish=finish)
                )
            ),
        )
        try:
            with pytest.raises(AnnotationError):
                client.answers(
                    await client.request(
                        client.config.endpoint, client.chat_body("test", {})
                    ),
                    {"q": {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}},
                )
        finally:
            await client.close()

    asyncio.run(run())


def test_explanations_only_describe_supplied_graph_matches():
    motif = next(m for m in MOTIFS if m.name == "untested_claim")

    def handle(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][1]["content"])
        assert user["graph"]["pattern_counts"] == {motif.name: 1}
        assert [m["name"] for m in user["matched_patterns"]] == [motif.name]
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"] == {motif.name: {"type": "string"}}
        assert "supported" not in schema["properties"]
        return httpx.Response(
            200,
            json=chat_response(
                json.dumps(
                    {motif.name: "H1 has no linked test in the annotated graph."}
                )
            ),
        )

    async def run():
        client = OpenAIClient(chat_config(), transport=httpx.MockTransport(handle))
        try:
            result, exchange = await explain_patterns(
                client,
                motifs=[motif],
                graph={"pattern_counts": {motif.name: 1}},
                trajectory=[],
                task="test",
            )
            assert result[motif.name].startswith("H1")
            assert exchange["response"]["usage"]["prompt_tokens"] == 10
        finally:
            await client.close()

    asyncio.run(run())


def test_model_override_keeps_endpoint_authentication_and_timeout(tmp_path):
    profile = tmp_path / "models.json"
    original = asdict(
        chat_config(
            api_key_env="MODEL_KEY",
            timeout=45,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )
    profile.write_text(json.dumps({"annotation": original}))
    changed = provider_config("annotation", {"model": "other-model"}, profile=profile)
    assert changed.model == "other-model"
    assert changed.endpoint == original["endpoint"]
    assert changed.api_key_env == "MODEL_KEY"
    assert changed.timeout == 45
    assert changed.extra_body == {}


def test_switching_protocol_selects_a_compatible_local_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORRAL_ANNOTATION_PROFILE", raising=False)
    config = provider_config("annotation", {"provider": "openai"})
    assert config.model == DEFAULT_CHAT_MODEL
    assert config.endpoint == "http://127.0.0.1:8790/v1/chat/completions"


def test_behavioral_checks_can_use_the_same_chat_client():
    from corral.epistemic_analysis.annotators.local import create_provider
    from corral.epistemic_analysis.trace_checks import check_definitions

    definition = check_definitions(["fabricated_evidence"])["fabricated_evidence"]

    def handle(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        assert set(data["questions"]) == {definition.name}
        return httpx.Response(
            200,
            json=chat_response(
                json.dumps({definition.name: {"choice": "absent", "confidence": 0.8}})
            ),
        )

    async def run():
        client = create_provider(chat_config(), transport=httpx.MockTransport(handle))
        try:
            exchange = await client.infer(client.body(definition, {"messages": []}))
            result = client.parse(exchange["response"], definition, 0.6)
            assert result["status"] == "absent"
            assert result["maximum_option_probability"] is None
            assert result["confidence_source"] == "model-self-assessment"
        finally:
            await client.close()

    asyncio.run(run())
