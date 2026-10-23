"""The query-scoped AFM reference needs neither credentials nor a specific cwd."""

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
def test_reference_returns_requested_topic_without_provider_credentials(
    launch_dir, monkeypatch
):
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    content = contract.reference("RMS friction")
    assert contract.CONTRACT_VERSION in content
    assert "calculate_rms_friction" in content
    assert "nanosurf.SPM()" not in content
    assert "StartFrameUp" not in content
    assert "PhaseContrast" not in content


def test_connection_reference_does_not_suggest_application_probing_or_workflow():
    content = contract.reference("How do I connect?")
    assert "nanosurf.SPM()" in content
    for hint in (
        "C3000()",
        "CoreAFM()",
        "StartFrameUp",
        "calculate_rms_friction",
        "SetPoint =",
    ):
        assert hint not in content


def test_multi_topic_query_returns_relevant_sections_only():
    content = contract.reference("connection and acquisition")
    assert "nanosurf.SPM()" in content
    assert "StartFrameUp" in content
    assert "ImageWidth" not in content
    assert "calculate_rms_friction" not in content


def test_property_lookup_finds_units():
    content = contract.reference("ImageWidth units")
    assert "Scan geometry" in content
    assert "metres" in content
    assert "nanosurf.SPM()" not in content


def test_unknown_question_lists_topics_without_dumping_manual():
    content = contract.reference("unrelated zebras")
    assert "No matching API section" in content
    assert "Frame acquisition" in content
    assert "StartFrameUp" not in content


@pytest.mark.parametrize("query", ["", "  ", None, 12])
def test_empty_or_invalid_query_is_rejected(query):
    with pytest.raises(ValueError, match="nonempty"):
        contract.reference(query)
