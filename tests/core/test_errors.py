from corral.core.errors import concise_error_message


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, body: dict | None = None) -> None:
        super().__init__(message)
        self.body = body


class HTTPStatusError(RuntimeError):
    __module__ = "httpx"


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


def test_concise_error_message_uses_outer_message_for_httpx_status_error():
    error = RuntimeError("maximum context length is 65536 tokens")
    error.__cause__ = RuntimeError("request failed")
    error.__cause__.__cause__ = HTTPStatusError(
        "Client error '400 Bad Request' for url 'https://example.test'"
    )

    assert concise_error_message(error) == "maximum context length is 65536 tokens"


def test_concise_error_message_collapses_multiline_text():
    error = RuntimeError("first line\n  final detail")

    assert concise_error_message(error) == "first line final detail"
