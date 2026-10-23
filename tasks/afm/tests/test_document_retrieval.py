"""The AFM reference is local and independent of launch directory or credentials."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "afm_contract", ROOT / "src/afm_contract.py"
)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


@pytest.mark.parametrize("launch_dir", [ROOT, ROOT / "src"])
def test_reference_is_complete_without_provider_credentials(launch_dir, monkeypatch):
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    content = contract.reference("RMS friction")
    assert content == contract.reference("connect and acquire")
    assert contract.CONTRACT_VERSION in content
    assert "calculate_rms_friction=True" in content
    assert "PhaseContrast" in content
    assert "LateralForce" in content
    assert "nanosurf.SPM()" in content
    assert "C3000()" not in content
    assert "CoreAFM()" not in content


@pytest.mark.parametrize("query", ["", "  ", None, 12])
def test_empty_or_invalid_query_is_rejected(query):
    with pytest.raises(ValueError, match="nonempty"):
        contract.reference(query)
