from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corral_md.tools import (
    convert_structure_to_lammps_data,
    execute_python_script,
    get_potential_metadata,
    get_structure_from_mp_text,
    run_lammps,
)


@pytest.fixture()
def mock_modal_function():
    """Fixture to mock modal.Function.from_name and return a configurable mock."""
    with patch("modal.Function.from_name") as mock_from_name:
        mock_function = MagicMock()
        mock_from_name.return_value = mock_function
        yield mock_from_name, mock_function


@pytest.mark.parametrize(
    ("file_path", "expected_metadata"),
    [
        (
            "path/to/Si.sw",
            "{potential type : Stillinger Weber (SW), elements supported : Si (Silicon), pair_style : sw}",
        ),
        (
            "/potentials/TERSOFF/2007_SiO.tersoff",
            "{potential type : tersoff, elements supported : Si (Silicon), Oxygen (O), pair_style : tersoff}",
        ),
        (
            "./potentials/Al99.eam.alloy",
            "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}",
        ),
        (
            "/data/potentials/Cu_Zhou04.eam.alloy",
            "{potential type : EAM, elements supported : Cu (Copper), pair_style : eam/alloy}",
        ),
        (
            "Mg_Zhou04.eam.alloy",
            "{potential type : EAM, elements supported : Mg (Magnesium), pair_style : eam/alloy}",
        ),
        (
            "some/dir/Fe-C_Hepburn_Ackland.eam.fs",
            "{potential type : EAM, elements supported : Fe (Iron), C (Carbon), pair_style : eam/fs}",
        ),
    ],
)
def test_get_potential_metadata_valid(file_path, expected_metadata):
    assert get_potential_metadata.execute(file_path=file_path) == expected_metadata


def test_get_potential_metadata_invalid():
    with pytest.raises(ValueError, match=r"Unrecognized potential file: unknown.eam"):
        get_potential_metadata.execute(file_path="potentials/unknown.eam")


@pytest.mark.parametrize("invalid_path", [None, ""])
def test_get_potential_metadata_none_or_empty(invalid_path):
    with pytest.raises(ValueError, match=r"File path must not be None or empty."):
        get_potential_metadata.execute(file_path=invalid_path)


def test_run_lammps_success(mock_modal_function):
    input_file = "input_file.data"
    expected_log_file = f"{Path(input_file).stem}.log"
    mock_from_name, mock_function = mock_modal_function

    result = run_lammps.execute(input_file=input_file)

    mock_function.remote.assert_called_once_with(input_file, expected_log_file)
    assert (
        result
        == f"Simulation ran successfully using input: {input_file}, log saved at: {expected_log_file}"
    )


@pytest.mark.parametrize(
    ("exception_type", "exception_message", "expected_match"),
    [
        (
            ValueError,
            "Some value error",
            "The LAMMPS simulation failed with a ValueError: Some value error",
        ),
        (
            RuntimeError,
            "Unexpected error",
            "An unexpected error occurred while running the LAMMPS simulation: Unexpected error",
        ),
    ],
)
def test_run_lammps_exceptions(
    mock_modal_function, exception_type, exception_message, expected_match
):
    input_file = "input_file.data"
    mock_from_name, mock_function = mock_modal_function
    mock_function.remote.side_effect = exception_type(exception_message)

    with pytest.raises((ValueError, Exception), match=expected_match):
        run_lammps.execute(input_file=input_file)


def test_run_lammps_with_none():
    with pytest.raises(ValueError, match="Input file path must not be None or empty."):
        run_lammps.execute(input_file=None)


def test_get_structure_from_mp_text():
    mp_id = "mp-149"
    file_path = "/results/Si.cif"

    # Mock Materials Project API and SpacegroupAnalyzer
    with (
        patch("mp_api.client.MPRester") as mock_mpr,
        patch("pymatgen.symmetry.analyzer.SpacegroupAnalyzer") as mock_sga,
        patch("modal.Function.from_name") as mock_function,
    ):
        # Mock structure
        mock_structure = MagicMock()
        mock_structure.to.return_value = "data_Si\n_cell_length_a 5.468"

        mock_doc = MagicMock()
        mock_doc.structure = mock_structure

        mock_mpr_instance = MagicMock()
        mock_mpr_instance.materials.summary.search.return_value = [mock_doc]
        mock_mpr.return_value.__enter__.return_value = mock_mpr_instance

        # Mock SpacegroupAnalyzer
        mock_sga_instance = MagicMock()
        mock_sga_instance.get_conventional_standard_structure.return_value = (
            mock_structure
        )
        mock_sga.return_value = mock_sga_instance

        # Mock write_file function
        mock_write = MagicMock()
        mock_function.return_value.remote = mock_write

        result = get_structure_from_mp_text.execute(mp_id=mp_id, file_path=file_path)

        assert result == f"Structure saved successfully at {file_path}"
        mock_write.assert_called_once()


def test_get_structure_from_mp_text_invalid_id():
    invalid_mp_id = "mp-9999999"
    file_path = "/results/fake.cif"

    with patch("mp_api.client.MPRester") as mock_mpr:
        mock_mpr_instance = MagicMock()
        mock_mpr_instance.materials.summary.search.side_effect = IndexError(
            "list index out of range"
        )
        mock_mpr.return_value.__enter__.return_value = mock_mpr_instance

        result = get_structure_from_mp_text.execute(
            mp_id=invalid_mp_id, file_path=file_path
        )

        assert isinstance(result, str)
        assert result.lower().startswith("failed to retrieve or save structure")


def test_convert_structure_to_lammps_data(mock_modal_function):
    structure_path = "/results/Si.cif"
    output_file = "/results/Si.data"
    mock_from_name, mock_function = mock_modal_function

    result = convert_structure_to_lammps_data.execute(
        structure_path=structure_path, output_file=output_file, atom_style="full"
    )

    assert result == f"LAMMPS data file successfully written to: {output_file}"
    mock_function.remote.assert_called_once_with(structure_path, output_file, "full")


def test_convert_structure_to_lammps_data_invalid_input(mock_modal_function):
    invalid_structure_path = "/nonexistent/path/invalid.cif"
    output_file = "/results/invalid.data"
    mock_from_name, mock_function = mock_modal_function
    mock_function.remote.side_effect = Exception("No such file or directory")

    with pytest.raises(Exception) as exc_info:
        convert_structure_to_lammps_data.execute(
            structure_path=invalid_structure_path,
            output_file=output_file,
            atom_style="atomic",
        )

    msg = str(exc_info.value).lower()
    assert "unexpected error" in msg


def test_execute_python_script_success(mock_modal_function):
    script_path = "/path/to/script.py"
    args = ["--input", "data.json"]
    mock_from_name, mock_function = mock_modal_function
    mock_function.remote.return_value = '{"success": true, "stdout": "Done"}'

    result = execute_python_script.execute(
        script_path=script_path, args=args, timeout=300
    )

    assert "success" in result
    mock_function.remote.assert_called_once()


def test_execute_python_script_failure(mock_modal_function):
    script_path = "/nonexistent/script.py"
    mock_from_name, mock_function = mock_modal_function
    mock_function.remote.side_effect = Exception("Script not found")

    with pytest.raises(Exception) as exc_info:
        execute_python_script.execute(script_path=script_path)

    assert "unexpected error" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main()
