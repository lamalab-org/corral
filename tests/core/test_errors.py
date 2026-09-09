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
