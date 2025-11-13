from pathlib import Path

import pytest
from corral_md.score import check_numerical, check_potential_file

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
    assert score_fn("Al99.eam.alloy") == 0.0


@pytest.mark.parametrize(
    ("target", "json_string", "expected"),  # ← Wrap param names in a tuple
    [
        (
            2.2173842,
            '{"BULK ENERGY": "2.21", "path to relaxed structure": "/structure/structure.dat"}',
            0.0,
        ),
        (
            2.2173842,
            '{"BULK ENERGY": "2.21", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            1.0,
        ),
        (
            2.2173842,
            '{"BULK ENERGY": "2.0", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            0.0,
        ),
        (
            2.2173842,
            '{"SLAB ENERGY": "2.2", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            1.0,
        ),
        (
            2.2173842,
            '{"density": "2.2", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            0.0,
        ),
        (
            2.2173842,
            '{"DENSITY": "2.2", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            0.0,
        ),
        (
            2.2173842,
            '{"BULK_ENERGY": "2.2", "path to relaxed structure": "/test_files/test_minimise/Al_minimised_structure.dat"}',
            0.0,
        ),
        (
            2.2180,
            '{"density": "2.2", "trajectory_file": "/test_files/Al_new/melt_Al_0.001_1500_npt_eam.lammpstrj"}',
            1.0,
        ),
        (
            2.2180,
            '{"density": "3.0", "trajectory_file": "/test_files/Al_new/melt_Al_0.001_1500_npt_eam.lammpstrj"}',
            0.0,
        ),
        (
            2.2180,
            '{"density": "2.2"}',
            0.0,
        ),
        (
            2.2180,
            '{"density": "2.2", "trajectory_file": "/invalid/path/file.lammpstrj"}',
            0.0,
        ),
        (2.2173842, '{"BULK ENERGY": "not_a_number"}', 0.0),
        (2.2173842, "{bad json}", 0.0),
        (2.2173842, None, 0.0),
        (2.2173842, "", 0.0),
        (2.2173842, "2.2173842", 1.0),
    ],
)
def test_check_numerical_submission_format(target, json_string, expected):
    score_fn = check_numerical(target=target, tolerance=2e-2)
    assert score_fn(json_string) == expected


if __name__ == "__main__":
    pytest.main()
