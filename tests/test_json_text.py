"""Reading the JSON value out of text a model returns."""

from __future__ import annotations

import pytest

from corral.utils.json_text import parse_json_text

FENCE = "`" * 3
ANSWER = {"classifications": {"case_1": ["population_a"]}}
BODY = '{"classifications": {"case_1": ["population_a"]}}'


@pytest.mark.parametrize(
    "text",
    [
        BODY,
        f"  {BODY}\n",
        f"{FENCE}json\n{BODY}\n{FENCE}",
        f"{FENCE}JSON\n{BODY}\n{FENCE}",
        f"{FENCE}\n{BODY}\n{FENCE}",
        f"{FENCE}json {BODY}{FENCE}",
        f"Here is my answer:\n\n{FENCE}json\n{BODY}\n{FENCE}\n\nThanks.",
        f"Draft:\n{FENCE}json\n{BODY}\n{FENCE}\nFinal:\n{FENCE}json\n{BODY}\n{FENCE}",
        f"{FENCE}python\nprint(1)\n{FENCE}\n{FENCE}json\n{BODY}\n{FENCE}",
    ],
)
def test_reads_bare_and_fenced_json(text):
    assert parse_json_text(text) == ANSWER


def test_a_value_already_decoded_passes_through():
    assert parse_json_text(ANSWER) is ANSWER


def test_a_value_starting_with_json_is_not_truncated():
    # A removeprefix("json") cleanup would strip the start of such a string.
    assert parse_json_text('"json-like"') == "json-like"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "no valid JSON"),
        ("the answer is population_a", "no valid JSON"),
        (f"{FENCE}json\n{{not json}}\n{FENCE}", "no valid JSON"),
        (
            f'{FENCE}json\n{BODY}\n{FENCE}\n{FENCE}json\n{{"classifications": {{}}}}\n{FENCE}',
            "several different",
        ),
    ],
)
def test_rejects_text_without_one_clear_value(text, reason):
    with pytest.raises(ValueError, match=reason):
        parse_json_text(text)
