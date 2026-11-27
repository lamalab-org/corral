"""Preserve KnowLine option scoring through the vLLM transport."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def adapter():
    if find_spec("serve_knowline_vllm") is None:
        pytest.skip(
            "Set PYTHONPATH to a frozen KnowLine vLLM adapter to run these tests"
        )
    import serve_knowline_vllm

    return serve_knowline_vllm


def fixture_scorer(adapter, probabilities=(0.2, 0.3, 0.5), prompt_tokens=4):
    captured = []
    response = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        "choices": [
            {
                "text": "ignored",
                "logprobs": {
                    "top_logprobs": [
                        {
                            f"token_id:{i}": math.log(p)
                            for i, p in enumerate(probabilities)
                        }
                    ]
                },
            }
        ],
    }

    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        return [11, 12, 13, 14]

    def post(url, *, json, timeout):
        captured.append(json)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

    scorer = adapter.VLLMScorer(
        "http://local", "pinned", SimpleNamespace(encode=encode)
    )
    scorer.http = SimpleNamespace(post=post)
    return scorer, captured


def test_native_options_and_calibration_are_preserved(adapter):
    scorer, calls = fixture_scorer(adapter)
    values, tokens = scorer.score("native template", [], [2, 0, 1])
    assert tokens == 4
    assert [math.exp(v) for v in values] == pytest.approx([0.5, 0.2, 0.3])
    assert calls[0]["prompt"] == [11, 12, 13, 14]
    assert calls[0]["allowed_token_ids"] == [2, 0, 1]
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["logprobs"] == 3
    # Applying the official temperature to normalized label logprobs gives the
    # same result as applying it to the raw selected logits.
    temperature = 0.6944
    calibrated = [math.exp(v / temperature) for v in values]
    expected = [p ** (1 / temperature) for p in [0.5, 0.2, 0.3]]
    assert [v / sum(calibrated) for v in calibrated] == pytest.approx(
        [v / sum(expected) for v in expected]
    )
    assert scorer.measurements[0]["reused_prompt_tokens"] == 3


def test_raw_top_k_or_missing_options_are_rejected(adapter):
    scorer, _ = fixture_scorer(adapter, probabilities=(0.01, 0.02, 0.03))
    with pytest.raises(ValueError, match="processed logprobs"):
        scorer.score("native template", [], [0, 1, 2])
    scorer, _ = fixture_scorer(adapter)
    with pytest.raises(ValueError, match="omitted"):
        scorer.score("native template", [], [0, 1, 3])


def test_prompt_changes_and_overflow_are_rejected(adapter):
    scorer, _ = fixture_scorer(adapter, prompt_tokens=3)
    with pytest.raises(ValueError, match="changed native prompt"):
        scorer.score("native template", [], [0, 1, 2])
    scorer, calls = fixture_scorer(adapter)
    scorer.max_tokens = 4
    with pytest.raises(ValueError, match="no truncation"):
        scorer.score("native template", [], [0, 1, 2])
    assert calls == []


def test_engine_mode_and_checkpoint_are_verified(adapter, monkeypatch, tmp_path):
    config = {
        "model_config": {
            "model": str(tmp_path),
            "dtype": "bfloat16",
            "logprobs_mode": "processed_logprobs",
            "max_model_len": 131072,
            "max_logprobs": 255,
        },
        "cache_config": {"enable_prefix_caching": True},
        "scheduler_config": {},
    }
    monkeypatch.setattr(
        adapter.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"vllm_config": config}
        ),
    )
    assert adapter.verify_engine("http://local", tmp_path, 131072)["backend"]
    config["model_config"]["logprobs_mode"] = "raw_logprobs"
    with pytest.raises(ValueError, match="processed_logprobs"):
        adapter.verify_engine("http://local", tmp_path, 131072)


def test_concurrent_requests_overlap_and_keep_their_own_metrics(adapter, monkeypatch):
    barrier = threading.Barrier(4)
    sessions = []

    def session():
        identity = object()
        sessions.append(identity)

        def post(url, *, json, timeout):
            # All four questions must reach the engine concurrently. A global
            # serialization lock would deadlock here and fail the test.
            barrier.wait(timeout=5)
            response = {
                "usage": {"prompt_tokens": len(json["prompt"]), "completion_tokens": 1},
                "choices": [
                    {
                        "logprobs": {
                            "top_logprobs": [
                                {
                                    "token_id:0": math.log(0.25),
                                    "token_id:1": math.log(0.75),
                                }
                            ]
                        }
                    }
                ],
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

        return SimpleNamespace(post=post)

    monkeypatch.setattr(adapter.requests, "Session", session)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(range(int(text))))
    scorer = adapter.VLLMScorer("http://local", "model", tokenizer, max_concurrency=4)
    with adapter.ContextExecutor(max_workers=4) as questions:

        def request(lengths):
            with scorer.request_measurements() as metrics:
                list(questions.map(lambda n: scorer.score(str(n), [], [0, 1]), lengths))
                return sorted(m["native_prompt_tokens"] for m in metrics)

        with ThreadPoolExecutor(max_workers=2) as callers:
            results = list(callers.map(request, [[4, 5], [8, 9]]))
    assert results == [[4, 5], [8, 9]]
    assert len(sessions) == 4  # Each scoring thread owns its HTTP session.


def test_measurement_context_resets_after_failure(adapter):
    scorer, _ = fixture_scorer(adapter)
    with pytest.raises(RuntimeError), scorer.request_measurements() as failed:
        scorer.score("native", [], [0, 1, 2])
        raise RuntimeError("request failed")
    with scorer.request_measurements() as next_request:
        scorer.score("native", [], [0, 1, 2])
    assert len(failed) == len(next_request) == 1
    assert scorer.measurements == []


def test_context_admission_is_fifo_and_releases_tokens_after_failure(adapter):
    capacity = adapter.ContextCapacity(4, 10)
    entered = []
    release = threading.Event()

    def run(name, tokens):
        with capacity.reserve(tokens):
            entered.append(name)
            if name == "large":
                assert release.wait(3)
                raise RuntimeError("inference failed")

    def queued(count):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with capacity.condition:
                if len(capacity.waiting) == count:
                    return
            threading.Event().wait(0.01)
        raise AssertionError("request never queued")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with capacity.reserve(10):
            large = pool.submit(run, "large", 10)
            queued(1)
            small = pool.submit(run, "small", 1)
            queued(2)
            assert entered == []
        queued(1)
        assert entered == ["large"]
        assert capacity.tokens == 10
        release.set()
        with pytest.raises(RuntimeError, match="inference failed"):
            large.result(timeout=3)
        small.result(timeout=3)
    assert entered == ["large", "small"]
    assert capacity.active == capacity.tokens == 0
    assert not capacity.waiting
    with pytest.raises(ValueError, match="context budget"), capacity.reserve(11):
        pass


def test_only_warm_identical_prefixes_share_capacity(adapter):
    cap = adapter.ContextCapacity(4, 12)
    prefix = (b"same", 8)
    cap.remember(prefix)
    with cap.reserve(10, prefix):
        with cap.reserve(10, prefix):
            assert cap.tokens == 12
            assert cap.active == 2
        assert cap.tokens == 10
    assert cap.tokens == 0
    assert not cap.references


def test_preflight_and_score_reuse_exact_token_ids(adapter):
    scorer, captured = fixture_scorer(adapter)
    scorer.cache_limit = 16
    calls = []
    encode = scorer.tokenizer.encode

    def counted(text, **kwargs):
        calls.append(text)
        return encode(text, **kwargs)

    scorer.tokenizer.encode = counted
    first = scorer.encode("native").tolist()
    scorer.score("native", [], [0, 1, 2])
    assert calls == ["native"]
    assert len(captured) == 1
    assert captured[0]["prompt"] == first
    assert scorer.cache_bytes == 16


def test_render_count_and_score_share_long_trace_tokens(adapter):
    calls = []

    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        calls.append(text)
        return list(text.encode())

    native = SimpleNamespace(encode=encode, apply_chat_template=object())
    scorer = adapter.VLLMScorer(
        "http://local", "model", native, token_cache_bytes=2**20, split_turns=True
    )
    template = adapter.CachedTemplate(native, scorer.encode)
    assert template.apply_chat_template is native.apply_chat_template
    prefix = "猫" * adapter.LONG_CONTEXT_CHARS + "<|turn>"
    for question in ("is this a node?", "is this an edge?", "is this present?"):
        text = prefix + "user\n" + question
        expected = list(text.encode())
        assert (
            list(template.tokenizer.encode(text, add_special_tokens=False)) == expected
        )
        assert list(scorer.encode(text)) == expected
    assert calls.count(prefix) == 1
    assert len(calls) == 4
    for text in ("short<|turn>user\n", "long without a native boundary" * 2048):
        assert list(scorer.encode(text)) == list(text.encode())
        assert calls[-1] == text


def test_context_groups_rotate_after_bounded_native_batches(adapter):
    groups = adapter.ContextGroups(64)
    entered = []

    def batch(key, n):
        with groups.reserve(key, n):
            entered.append(key)
            if key == "B":
                raise ValueError("failed batch")

    def queued(key):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with groups.condition:
                if key in groups.pending:
                    return
            time.sleep(0.001)
        pytest.fail("Batch never queued")

    with ThreadPoolExecutor(3) as pool:
        with groups.reserve("A", 32):
            other = pool.submit(batch, "B", 32)
            queued("B")
            same = pool.submit(batch, "A", 32)
            same.result(timeout=3)
            next_turn = pool.submit(batch, "A", 1)
            queued("A")
            assert entered == ["A"]
        with pytest.raises(ValueError, match="failed batch"):
            other.result(timeout=3)
        next_turn.result(timeout=3)
    assert entered == ["A", "B", "A"]
    assert groups.active == 0
    assert not groups.pending

    with groups.reserve("A", 64), groups.for_request({"m1": "short"}, {"q": {}}):
        assert groups.active == 1
