"""Fixed projection of structured tool docstrings into runtime metadata.

Corral no longer routes or varies tool verbosity. Runtime tool schemas always
use the historical default: the ``[BRIEF]`` section when present, otherwise the
ordinary untagged description. Broader tags may remain in source docstrings as
authoring material, but they never become a benchmark parameter.
"""

from __future__ import annotations

import re
from typing import Final

_KEYWORDS: Final = (
    "BRIEF",
    "DETAILED",
    "PROCEDURAL",
    "WORKFLOW_INTEGRATION",
    "CONTEXTUAL",
    "SYNTACTICAL",
    "RAISES",
    "LIMITATIONS",
    "EXAMPLES",
    "ARGS_BRIEF",
    "ARGS_DETAILED",
    "ARGS_SYNTACTICAL",
    "ARGS_EXAMPLES",
    "RETURNS_BRIEF",
    "RETURNS_DETAILED",
    "RETURNS_EXAMPLES",
)
_KEYWORD_PATTERN = "|".join(re.escape(keyword) for keyword in _KEYWORDS)
_TAG = re.compile(
    rf"\[({_KEYWORD_PATTERN})\](.*?)\[/\1\]",
    re.DOTALL | re.IGNORECASE,
)
_ARGS = re.compile(r"Args:(.*?)(?=Returns:|$)", re.DOTALL | re.IGNORECASE)
_RETURNS = re.compile(r"Returns:(.*?)$", re.DOTALL | re.IGNORECASE)


def _clean_nested_tags(content: str) -> str:
    nested = re.compile(r"\[([A-Z_]+)\](.*?)\[/\1\]", re.DOTALL)
    while nested.search(content):
        content = nested.sub(r"\2", content)
    return content.strip()


def _sections(description: str) -> dict[str, str]:
    sections = {
        name.upper(): _clean_nested_tags(content)
        for name, content in _TAG.findall(description)
        if content.strip()
    }
    untagged = _TAG.sub("", description)
    untagged = _ARGS.sub("", untagged)
    untagged = _RETURNS.sub("", untagged).strip()
    if untagged:
        sections["BASIC"] = untagged
    return sections


def default_tool_description(description: str) -> str:
    """Return the one fixed runtime description for a tool."""
    if not description:
        return "No description available"
    sections = _sections(description)
    return sections.get("BRIEF") or sections.get("BASIC") or description.splitlines()[0]


def default_argument_description(description: str) -> str:
    """Return the fixed brief projection for one argument description."""
    if not description:
        return description
    sections = {
        name.removeprefix("ARGS_"): value
        for name, value in _sections(description).items()
    }
    projected = sections.get("BRIEF") or sections.get("BASIC") or description
    if "(choices:" in description and "(choices:" not in projected:
        start = description.find("(choices:")
        end = description.find(")", start)
        if end != -1:
            projected = f"{projected}\n{description[start : end + 1]}"
    return projected


__all__ = ["default_argument_description", "default_tool_description"]
