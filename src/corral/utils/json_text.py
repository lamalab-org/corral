"""Read the JSON value out of text a model returns, such as a task submission."""

from __future__ import annotations

import json
import re
from typing import Any

# A Markdown code fence with an optional language tag (```json, ```JSON, ```).
_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n?(.*?)```", re.DOTALL)


def parse_json_text(value: Any) -> Any:
    """Return the JSON value that model-written text holds.

    A non-string value is returned unchanged. A string is read as bare
    JSON first; failing that, as the JSON inside a Markdown code fence, with
    any prose around the fence ignored. Several fences are accepted only when
    every one that parses holds the same value, so the caller never has to
    guess which of two different answers was meant.

    Raises ValueError with a short reason fit to show in a report.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    blocks = []
    for block in _FENCE.findall(text):
        try:
            blocks.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if not blocks:
        raise ValueError("no valid JSON found")
    if any(block != blocks[0] for block in blocks[1:]):
        raise ValueError("several different JSON code blocks found")
    return blocks[0]
