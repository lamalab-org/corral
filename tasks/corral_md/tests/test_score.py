from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corral_md.score import check_numerical, check_potential_file, check_structure

BASE_DIR = Path(__file__).parent.parent.resolve()

GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"

BASE_DIR = Path(__file__).parent.parent.resolve()
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"


@pytest.fixture()
def mock_modal_lookup():
    """
    Unified fixture for:
    - check_potential_file
    - check_numerical
    - check_structure
    """
    with patch("corral_md.score.modal.Function.lookup") as mock_lookup:

        def fake_lookup(_name, func):
            mock_func = MagicMock()

            # --- file_info branch ---
            if func == "file_info":

                def fake_file_info(path: str):
                    if path == "/potentials/EAM/Al99.eam.alloy":
                        return {"size": 1234}
                    if path and "relaxed_structure" in path:
                        return {"size": 5678}
                    raise RuntimeError("file not found")

                mock_func.remote.side_effect = fake_file_info

            # --- read_file branch ---
            elif func == "read_file":

                def fake_read_file(path: str):
                    if not path:  # empty string or None
                        raise RuntimeError("file not found")
                    # Otherwise, return dummy content
                    return "fake LAMMPS content"

                mock_func.remote.side_effect = fake_read_file

            return mock_func

        mock_lookup.side_effect = fake_lookup
        yield mock_lookup


def test_check_potential_file(mock_modal_lookup):
    _ = mock_modal_lookup
    target = "/potentials/EAM/Al99.eam.alloy"
    score_fn = check_potential_file(target)

    assert score_fn("/some/path/valid_file.txt") == 0.0
    assert score_fn("/potentials/EAM/Al99.eam.alloy") == 1.0
    assert score_fn("randomstring") == 0.0
    assert score_fn(None) == 0.0
    assert score_fn("") == 0.0
    assert score_fn("Al99.eam.alloy") == 0.0


@pytest.mark.parametrize(
    ("target", "json_string", "expected"),
    [
        (
            2.2173842,
            '{"BULK ENERGY": "2.21", "path to relaxed structure": "/structure/structure.dat"}',
            0.0,
        ),
        (
            2.2173842,
            '{"BULK ENERGY": "2.21", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            1.0,
        ),
        (
            2.2173842,
            '{"BULK ENERGY": "2.0", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            0.0,
        ),
        (
            2.2173842,
            '{"SLAB ENERGY": "2.2", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            1.0,
        ),
        (
            2.2173842,
            '{"density": "2.2", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            1.0,
        ),
        (
            2.2173842,
            '{"DENSITY": "2.2", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            0.0,
        ),
        (
            2.2173842,
            '{"BULK_ENERGY": "2.2", "path to relaxed structure": "/results/.../relaxed_structure.data"}',
            0.0,
        ),
        (2.2173842, '{"BULK ENERGY": "not_a_number"}', 0.0),
        (2.2173842, "{bad json}", 0.0),
        (2.2173842, None, 0.0),
        (2.2173842, "", 0.0),
        (2.2173842, "2.2173842", 1.0),
    ],
)
def test_check_numerical_submission_format(
    mock_modal_lookup, target, json_string, expected
):
    _ = mock_modal_lookup
    score_fn = check_numerical(target=target, tolerance=2e-2)
    assert score_fn(json_string) == expected


@pytest.mark.parametrize(
    ("target_path", "result_path", "atom_style", "expected_score"),
    [
        (
            "/ground_truth/structures/Al.data",
            "/results/Aluminum_structure.data",
            "atomic",
            1.0,
        ),
        ("/ground_truth/structures/Al.data", None, "atomic", 0.0),
        ("/ground_truth/structures/Al.data", "", "atomic", 0.0),
        ("/ground_truth/structures/Si.data", "/results/silicon.data", "full", 1.0),
    ],
)
def test_check_structure_varied_styles_mocked(
    mock_modal_lookup, target_path, result_path, atom_style, expected_score
):
    _ = mock_modal_lookup
    with (
        patch("pymatgen.io.lammps.data.LammpsData.from_file") as mock_ld,
        patch("pymatgen.analysis.structure_matcher.StructureMatcher") as mock_matcher,
    ):
        # LammpsData.from_file returns dummy objects
        class DummyLammpsData:
            def __init__(self, path, **_kwargs):
                self.structure = path

        mock_ld.side_effect = lambda path, **_kwargs: DummyLammpsData(path)

        # StructureMatcher returns dummy matcher
        class DummyMatcher:
            def fit(self, _s1, _s2):
                # Always return True for testing purposes
                return True

        mock_matcher.return_value = DummyMatcher()

        score_fn = check_structure(target_path, atom_style=atom_style)
        score = score_fn(result_path)

        assert score == expected_score


if __name__ == "__main__":
    pytest.main()
