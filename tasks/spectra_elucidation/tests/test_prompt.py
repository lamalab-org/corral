from types import SimpleNamespace

import pytest
from spectra_elucidation.env import _spectra_prompt


def _env(initial_input):
    task = SimpleNamespace(
        description="Identify the sample.",
        submission_format="SMILES string",
        input_map={},
        initial_input=initial_input,
    )
    return SimpleNamespace(
        current_task=task,
        resolve_inputs=lambda _state: {},
        group_tasks={},
        task_id="t",
    )


@pytest.mark.parametrize("initial_input", [{}, {"work_dir": "/tmp/w"}])
def test_prompt_omits_empty_inputs_section(initial_input):
    prompt = _spectra_prompt(_env(initial_input), state=None)

    assert "Here are the inputs I have" not in prompt
    assert prompt.endswith("SMILES string\n\n")


def test_prompt_lists_real_inputs_but_not_work_dir():
    prompt = _spectra_prompt(
        _env({"work_dir": "/tmp/w", "formula": "C2H6O"}), state=None
    )

    assert "Here are the inputs I have:\n- formula: C2H6O\n" in prompt
    assert "work_dir" not in prompt
