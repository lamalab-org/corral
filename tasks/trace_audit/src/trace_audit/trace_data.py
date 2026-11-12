"""Read and mask frozen pilot cases."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
REDACTIONS = re.compile(
    r"(?<![A-Za-z0-9])(?:claude[ _-]?code|claudecode|ai[ _-]?scientist(?:agent)?|"
    r"aiscientist\d*|openhands|codex|react|claude[ -]?opus[ -]?5[.-]?5|"
    r"opus55|gpt[ -]?5[.-]?6[ -]?sol|gpt56sol|kimi[ -]?k3|kimik3)"
    r"(?![A-Za-z0-9])|anthropic/|openai/|openrouter/moonshotai/",
    re.IGNORECASE,
)


def mask(value):
    """Remove source labels from a case while retaining neutral trial IDs."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"source_agent", "source_model"}:
                continue
            if key == "source_path" and "trial_id" in value:
                result[key] = (
                    f"traces/{value.get('environment', 'traces')}/"
                    f"run-{value['trial_id']}.json"
                )
            else:
                result[key] = mask(item)
        return result
    if isinstance(value, list):
        return [mask(item) for item in value]
    if isinstance(value, str):
        return REDACTIONS.sub("[identity]", value)
    return value


def load_case(case_id: str, level: int, data_dir: Path = DATA) -> dict:
    """Load a frozen case, masking source labels for Level 2."""
    flat_path = data_dir / "cases" / f"{case_id}.json.gz"
    paths = [flat_path] if flat_path.exists() else list(
        (data_dir / "cases").glob(f"*/{case_id}.json.gz")
    )
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one stored file for {case_id}; found {len(paths)}")
    with gzip.open(paths[0], "rt", encoding="utf-8") as stream:
        case = json.load(stream)
    return case if level == 1 else mask(case)
