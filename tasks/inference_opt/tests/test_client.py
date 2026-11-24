import pytest

from inference_opt.client import StudentEndpoint


class EmptyResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": []}


def test_api_key_is_allowed_for_local_http(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post", lambda *args, **kwargs: EmptyResponse()
    )

    endpoint = StudentEndpoint("http://127.0.0.1:8000", api_key="secret")
    assert [c.text for c in endpoint.complete("hello")] == [""]


def test_api_key_is_rejected_for_remote_http(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post",
        lambda *args, **kwargs: pytest.fail("request was sent"),
    )

    with pytest.raises(ValueError, match="non-local HTTP"):
        StudentEndpoint("http://student.example", api_key="secret").complete("hello")


def test_remote_http_without_api_key_is_allowed(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post", lambda *args, **kwargs: EmptyResponse()
    )

    endpoint = StudentEndpoint("http://student.example")
    assert [c.text for c in endpoint.complete("hello")] == [""]


class JsonResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self):
        return self._payload


def test_truncated_reasoning_is_empty_text_not_none(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": None, "reasoning": "Let me think..."},
                "finish_reason": "length",
            }
        ]
    }
    monkeypatch.setattr(
        "inference_opt.client.requests.post", lambda *a, **k: JsonResponse(payload)
    )
    endpoint = StudentEndpoint("http://127.0.0.1:8000")
    [completion] = endpoint.complete("hi", max_tokens=16)
    assert completion.text == ""
    assert completion.finish_reason == "length"
    assert completion.reasoning == "Let me think..."
    assert [c.text for c in endpoint.complete("hi")] == [""]


def test_several_samples_are_separate_single_requests(monkeypatch):
    bodies = []

    def post(url, json, timeout, headers):
        bodies.append(json)
        return JsonResponse(
            {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr("inference_opt.client.requests.post", post)
    completions = StudentEndpoint("http://127.0.0.1:8000").complete("2+2?", n=3)
    assert [c.text for c in completions] == ["4", "4", "4"]
    assert len(bodies) == 3
    assert all("n" not in body for body in bodies)


def test_server_error_message_is_surfaced(monkeypatch):
    error = {"error": {"message": "Only n=1 is supported.", "code": "bad"}}
    monkeypatch.setattr(
        "inference_opt.client.requests.post",
        lambda *a, **k: JsonResponse(error, status=400),
    )
    with pytest.raises(RuntimeError, match=r"\(400\): Only n=1 is supported"):
        StudentEndpoint("http://127.0.0.1:8000").complete("hi")


def test_timeout_grows_with_max_tokens(monkeypatch):
    seen = []

    def post(url, json, timeout, headers):
        seen.append(timeout)
        return EmptyResponse()

    monkeypatch.setattr("inference_opt.client.requests.post", post)
    endpoint = StudentEndpoint("http://127.0.0.1:8000")
    endpoint.complete("hi", max_tokens=256)
    endpoint.complete("hi", max_tokens=8192)
    assert seen[1] > seen[0] >= 120
    # A 8192-token answer at 5 tokens/s must still fit.
    assert seen[1] >= 8192 / 5
    # Never longer than the per-question limit.
    endpoint.complete("hi", max_tokens=100_000)
    assert seen[2] == 1800


def test_reasoning_off_is_sent_in_each_backends_own_form(monkeypatch):
    from inference_opt.client import student_endpoint

    bodies = []

    def post(url, json, timeout, headers):
        bodies.append(json)
        return JsonResponse(
            {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr("inference_opt.client.requests.post", post)
    off = {"reasoning": False}
    student_endpoint("openrouter/qwen/qwen3.5-9b@together", options=off).chat([])
    student_endpoint("vllm/Qwen/Qwen3.5-9B", "http://127.0.0.1:8000/v1", options=off).chat([])
    student_endpoint("vllm/Qwen/Qwen3.5-9B", "http://127.0.0.1:8000/v1").chat([])
    assert bodies[0]["reasoning"] == {"enabled": False}
    assert bodies[0]["provider"] == {"order": ["together"], "allow_fallbacks": False}
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in bodies[2]
    assert "chat_template_kwargs" not in bodies[2]


def test_a_reply_the_provider_broke_off_is_retried(monkeypatch):
    replies = [
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]},
        {"choices": [{"message": {"content": "ANSWER: 4"}, "finish_reason": "stop"}]},
    ]
    monkeypatch.setattr("inference_opt.client.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "inference_opt.client.requests.post",
        lambda *a, **k: JsonResponse(replies.pop(0)),
    )
    completion = StudentEndpoint("http://127.0.0.1:8000").chat([])
    assert completion.text == "ANSWER: 4"
    assert completion.finish_reason == "stop"
