"""Versioned interaction settings, independent of scientific scoring."""

import hashlib
import json
from pathlib import Path

BLIND_PROTOCOL = "stargazer-blind-v1"
ASSISTED_PROTOCOL = "stargazer-assisted-v1"


def protocol_version(analysis_assistance: bool) -> str:
    if not isinstance(analysis_assistance, bool):
        raise ValueError("analysis_assistance must be a boolean")
    return ASSISTED_PROTOCOL if analysis_assistance else BLIND_PROTOCOL


def public_resources() -> dict[str, str]:
    """Read only these two audited files; never export a repository directory."""
    root = Path(__file__).parent
    return {
        name: (root / name).read_text(encoding="utf-8")
        for name in ("public_rv.py", "analysis-guide.md")
    }


def execution_fingerprint(*, protocol, bank, criteria, development_mode, source_hash):
    settings = json.dumps(
        {
            "protocol": protocol,
            "bank": bank,
            "criteria": criteria,
            "development_mode": development_mode,
            "source_hash": source_hash,
        },
        sort_keys=True,
    )
    return protocol + ":" + hashlib.sha256(settings.encode()).hexdigest()
