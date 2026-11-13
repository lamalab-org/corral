"""Current definition snapshots; no API calls."""

import hashlib
import json
import re
from pathlib import Path

from corral.epistemic_analysis import ontology

FIXTURES = Path(__file__).parent / "fixtures"


def test_current_prompt_snapshots_and_shared_jev_criteria_match():
    fixture = json.loads((FIXTURES / "prompt_hashes.json").read_text())
    for name, expected in fixture.items():
        assert hashlib.sha256(getattr(ontology, name).encode()).hexdigest() == expected
    assert (
        dict(
            re.findall(
                r"^([HTEJCNF]) = (.+)$", ontology.PASS_A_INSTRUCTIONS, re.MULTILINE
            )
        )
        == ontology.NODE_CRITERIA
    )
