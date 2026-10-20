from corral.core.errors import concise_error_message


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, body: dict | None = None) -> None:
        super().__init__(message)
        self.body = body


def test_concise_error_message_uses_leaf_exception_only():
    try:
        raise ValueError("response schema is invalid")
    except ValueError as cause:
        error = RuntimeError("model request failed")
        error.__cause__ = cause

    assert concise_error_message(error) == "response schema is invalid"


def test_concise_error_message_prefers_structured_provider_detail():
    error = ProviderError(
        "litellm.BadRequestError: request failed through provider adapter",
        body={"error": {"message": "Required field 'answer' is missing."}},
    )
    error.__cause__ = RuntimeError("HTTP 400")

    assert concise_error_message(error) == "Required field 'answer' is missing."


def test_concise_error_message_collapses_multiline_text():
    error = RuntimeError("first line\n  final detail")

    assert concise_error_message(error) == "first line final detail"


def test_concise_error_message_reads_http_provider_response():
    import httpx

    message = "You have reached your specified API usage limits."
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://provider.test/v1/messages"),
        json={"error": {"type": "invalid_request_error", "message": message}},
    )
    error = httpx.HTTPStatusError(
        "HTTP 400", request=response.request, response=response
    )
    assert concise_error_message(error) == message


def test_concise_error_message_keeps_non_json_http_error():
    import httpx

    response = httpx.Response(
        502,
        request=httpx.Request("POST", "https://provider.test/v1/messages"),
        text="gateway failure",
    )
    error = httpx.HTTPStatusError(
        "HTTP 502", request=response.request, response=response
    )
    assert concise_error_message(error) == "HTTP 502"
