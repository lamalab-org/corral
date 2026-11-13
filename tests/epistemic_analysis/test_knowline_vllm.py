"""Preserve KnowLine option scoring through the vLLM transport."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
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
