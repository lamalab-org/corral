"""ReAct annotations need the public startup contract as well as actions."""

import json
from pathlib import Path

import pytest

from corral.epistemic_analysis.react_prompt_context import PinnedReActContext

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contexts():
    return {
        name: PinnedReActContext(ROOT, "test-revision", name)
        for name in (
            "spectra_elucidation",
            "wetlab",
            "resistor_network",
            "retrosynthesis",
        )
    }


def recover(context, **task):
    return context.recover(
        {"id": "test-task", "output": "HIDDEN_ANSWER_CANARY", **task},
        "Identify the supplied sample.",
        [{"path": "task.json", "sha256": "test", "git_commit": "test-revision"}],
        {},
    )


def test_spectra_startup_includes_contract_but_no_hidden_sample(contexts):
    messages, sources, tools = recover(
        contexts["spectra_elucidation"],
        tools=["mass_spectrometry_spectra", "proton_nmr_spectra", "search_by_smiles"],
    )
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert "measurements" in messages[0]["content"]
    assert "Identify the supplied sample." in messages[1]["content"]
    assert "<action_input>" in messages[1]["content"]
    catalog = json.loads(messages[2]["content"].split("\n", 1)[1])
    assert catalog == tools
    by_name = {t["function"]["name"]: t["function"] for t in catalog}
    assert set(by_name) == {
        "mass_spectrometry_spectra",
        "proton_nmr_spectra",
        "search_by_smiles",
        "submit_answer",
    }
    for name in ("mass_spectrometry_spectra", "proton_nmr_spectra"):
        assert by_name[name]["parameters"]["properties"] == {}
        assert by_name[name]["parameters"].get("required", []) == []
    assert (
        "nmr_database_archive"
        not in by_name["search_by_smiles"]["parameters"]["properties"]
    )
    assert by_name["submit_answer"]["parameters"]["required"] == ["answer"]
    assert "HIDDEN_ANSWER_CANARY" not in json.dumps(messages)
    assert all(
        s["origin"] == "reconstructed_react_initial_instructions"
        and s["source_evidence"]
        for s in sources
    )
    with pytest.raises(RuntimeError, match="cannot execute"):
        contexts["spectra_elucidation"].pool["mass_spectrometry_spectra"]._func("C")


def test_environment_selection_and_workspace_contracts(contexts):
    _, _, wetlab = recover(contexts["wetlab"], excluded_tools=["measure_pH"])
    names = {t["function"]["name"] for t in wetlab}
    assert "measure_pH" not in names
    assert "perform_flame_test" in names
    assert all(
        "wetlab" not in t["function"]["parameters"]["properties"] for t in wetlab
    )
    _, _, resistor = recover(
        contexts["resistor_network"], tools=["calculate_series_resistance"]
    )
    by_name = {t["function"]["name"]: t["function"] for t in resistor}
    assert set(by_name) == {
        "calculate_series_resistance",
        "list_files",
        "read_file",
        "write_file",
        "file_info",
        "cat_files",
        "copy_file",
        "submit_answer",
    }
    assert (
        by_name["list_files"]["parameters"]["properties"]["path"]["default"]
        == "/workspace"
    )
    _, _, retro = recover(contexts["retrosynthesis"], tools=[])
    assert {t["function"]["name"] for t in retro} == set(
        contexts["retrosynthesis"].pool
    ) | {"submit_answer"}


def test_unknown_tools_and_prompt_overrides_fail_closed(contexts):
    with pytest.raises(ValueError, match="missing tools"):
        recover(contexts["spectra_elucidation"], tools=["not_a_real_tool"])
    with pytest.raises(ValueError, match="overrides"):
        contexts["spectra_elucidation"].recover(
            {"id": "task"}, "Task", [], {"user_prompt": "custom"}
        )
