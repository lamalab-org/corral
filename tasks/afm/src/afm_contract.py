"""Minimal tool contract and query-scoped, offline AFM API documentation."""

import re
from pathlib import Path

CONTRACT_VERSION = "managed-afm-v2"

INSTRUCTIONS = """AFM execution contract: managed-afm-v2
These tools control a real AFM. Use only the provided instrument-control interface.
InstrumentUnavailable indicates an infrastructure failure requiring operator recovery.
"""

_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "use",
        "what",
        "with",
    ]
)


def _terms(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP_WORDS
    aliases = {
        "connect": "connection",
        "connecting": "connection",
        "acquire": "acquisition",
        "scans": "scan",
        "scanning": "scan",
        "parameters": "settings",
        "units": "unit",
    }
    return {aliases.get(word, word) for word in words}


def reference(query: str) -> str:
    """Return at most two relevant API sections, never an unsolicited workflow."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Provide a nonempty AFM API question")
    document = (
        Path(__file__).with_name("afm_api_reference.md").read_text(encoding="utf-8")
    )
    sections = re.split(r"(?m)^## ", document)[1:]
    if not sections:
        raise ValueError("AFM API reference has no sections")
    terms = _terms(query)
    ranked = []
    for section in sections:
        heading = section.splitlines()[0]
        score = len(terms & _terms(section)) + 2 * len(terms & _terms(heading))
        if score:
            ranked.append((score, section))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        topics = ", ".join(section.splitlines()[0] for section in sections)
        return f"No matching API section. Available topics: {topics}."
    relevant = [section for score, section in ranked[:2] if score >= ranked[0][0] / 2]
    return f"AFM API reference ({CONTRACT_VERSION})\n\n" + "\n\n".join(
        "## " + section.strip() for section in relevant
    )
