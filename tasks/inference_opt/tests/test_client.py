import pytest

from inference_opt.client import StudentEndpoint


class EmptyResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": []}


def test_api_key_is_allowed_for_local_http(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post", lambda *args, **kwargs: EmptyResponse()
    )

    endpoint = StudentEndpoint("http://127.0.0.1:8000", api_key="secret")
    assert endpoint.generate("hello") == [""]


def test_api_key_is_rejected_for_remote_http(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post",
        lambda *args, **kwargs: pytest.fail("request was sent"),
    )

    with pytest.raises(ValueError, match="non-local HTTP"):
        StudentEndpoint("http://student.example", api_key="secret").generate("hello")


def test_remote_http_without_api_key_is_allowed(monkeypatch):
    monkeypatch.setattr(
        "inference_opt.client.requests.post", lambda *args, **kwargs: EmptyResponse()
    )

    endpoint = StudentEndpoint("http://student.example")
    assert endpoint.generate("hello") == [""]
