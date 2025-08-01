from pathlib import Path

import pytest
from corral_md.score import check_numerical, check_potential_file, check_structure

BASE_DIR = Path(__file__).parent.parent.resolve()

GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"


def test_check_potential_file():
    target = "/potentials/EAM/Al99.eam.alloy"
    score_fn = check_potential_file(target)

    assert score_fn("/some/path/valid_file.txt") == 0.0
    assert score_fn("/potentials/EAM/Al99.eam.alloy") == 1.0
    assert score_fn("randomstring") == 0.0
    assert score_fn(None) == 0.0
    assert score_fn("") == 0.0
    assert score_fn("/potentials/EAM/Cu_Zhou04.eam.alloy") == 0.0


@pytest.mark.parametrize(
    ("target", "json_string", "expected"),  # ← Wrap param names in a tuple
    [
        # Valid BULK ENERGY string within tolerance
        (
            2.2173842,
            '{"BULK ENERGY": 2.0, "Relaxed BULK Structure_path": "/structure/structure.dat"}',
            0.0,
        ),
        # Valid BULK ENERGY string outside tolerance
        (
            2.2173842,
            '{"BULK ENERGY": 2.216, "Relaxed BULK Structure_path": "/structure/structure.dat"}',
            1.0,
        ),
        (
            2.2173842,
            '{"BULK ENERGY": "2.216", "Relaxed BULK Structure_path": /structure/structure.dat}',
            1.0,
        ),
        # Valid SLAB ENERGY string within tolerance
        (
            1671.89,
            '{"SLAB ENERGY": 1671.88, "Relaxed SLAB Structure_path": "/structure/structure.dat"}',
            1.0,
        ),
        # SLAB ENERGY outside tolerance
        (
            1671.89,
            '{"SLAB ENERGY": 1600.0, "Relaxed SLAB Structure_path": "/structure/structure.dat"}',
            0.0,
        ),
        # Valid density key
        (
            0.0589124159899187,
            '{"density": 0.05871, "Relaxed BULK Structure_path": "/structure/structure.dat"}',
            1.0,
        ),
        # Missing all keys
        (0.0589124159899187, '{"volume": 100.0, "structure": "something"}', 0.0),
        (
            2.2173842,
            '{"BULK ENERGY": "2.217", "Relaxed BULK Structure_path": "/structure/structure.dat"}',
            1.0,
        ),
        # Invalid JSON format
        (2.2173842, '{"BULK ENERGY": "not_a_number"}', 0.0),
        (2.2173842, "{bad json}", 0.0),
        # Null or empty input
        (2.2173842, None, 0.0),
        (2.2173842, "", 0.0),
        (2.2173842, "2.2173842", 1.0),
    ],
)
def test_check_numerical_submission_format(target, json_string, expected):
    score_fn = check_numerical(target=target, tolerance=2e-2)
    assert score_fn(json_string) == expected


@pytest.mark.parametrize(
    ("target_path", "result_path", "atom_style", "use_modal", "expected_score"),
    [
        (f"{GROUND_TRUTH_DIR}/structures/Al.data", None, "atomic", True, 0.0),
        (f"{GROUND_TRUTH_DIR}/structures/Al.data", "", "atomic", True, 0.0),
        (
            f"{GROUND_TRUTH_DIR}/structures/Al.data",
            "/results/23_July_2025/test/gpt_4o/subtask/aluminum_structure_retrieval_subtask_em_trial_0/Aluminum_structure.data",
            "atomic",
            True,
            1.0,
        ),
        (
            f"{GROUND_TRUTH_DIR}/energy_minimisation/Aluminum/Al_minimised_structure.dat",
            "/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.data",
            "atomic",
            True,
            1.0,
        ),
        (
            f"{GROUND_TRUTH_DIR}/energy_minimisation/Aluminum/Al_minimised_structure.dat",
            "/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.in",
            "atomic",
            True,
            0.0,
        ),
        (
            f"{GROUND_TRUTH_DIR}/structures/Si.data",
            "/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.in",
            "atomic",
            True,
            0.0,
        ),
        (
            f"{GROUND_TRUTH_DIR}/structures/Si.data",
            "/results/23_July_2025/test/gpt_4o/subtask/aluminum_energy_minimisation_subtask_npt_trial_0/relaxed_structure.in",
            "full",
            True,
            0.0,
        ),
        (
            f"{GROUND_TRUTH_DIR}/structures/Si.data",
            "/results/new_benchmark_data_new/react/gpt_4o/surface_energy/task_10/task_10_4_05142114/silicon.data",
            "full",
            True,
            1.0,
        ),
    ],
)
def test_check_structure_varied_styles(
    target_path, result_path, atom_style, use_modal, expected_score
):
    score_fn = check_structure(target_path, atom_style=atom_style, use_modal=use_modal)
    assert score_fn(result_path) == expected_score


if __name__ == "__main__":
    pytest.main()
