import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import modal
import pytest
from corral_md.tools import (
    convert_structure_to_lammps_data,
    get_potential_metadata,
    get_structure_from_mp_text,
    run_lammps,
)
from pymatgen.core import Structure


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


def test_run_lammps_success():
    input_file = "input_file.data"
    expected_log_file = f"{Path(input_file).stem}.log"

    # Create a mock for the remote function
    mock_remote = MagicMock()

    # Patch 'modal.Function.lookup' to return an object with a 'remote' method (the mock_remote)
    with patch("modal.Function.lookup") as mock_lookup:
        mock_lookup.return_value.remote = mock_remote

        result = run_lammps.execute(input_file=input_file)

        # Assert the remote method was called once with the right arguments
        mock_remote.assert_called_once_with(input_file, expected_log_file)

        # Assert the return message is as expected
        assert (
            result
            == f"Simulation ran successfully using input: {input_file}, log saved at: {expected_log_file}"
        )


def test_run_lammps_value_error():
    input_file = "input_file.data"

    # Patch 'modal.Function.lookup' and make its remote method raise ValueError
    with patch("modal.Function.lookup") as mock_lookup:
        mock_remote = MagicMock()
        mock_remote.remote.side_effect = ValueError("Some value error")
        mock_lookup.return_value = mock_remote

        with pytest.raises(
            ValueError,
            match="The LAMMPS simulation failed with a ValueError: Some value error",
        ):
            run_lammps.execute(input_file=input_file)


def test_run_lammps_unexpected_exception():
    input_file = "input_file.data"

    # Patch 'modal.Function.lookup' and make its remote method raise a generic Exception
    with patch("modal.Function.lookup") as mock_lookup:
        mock_remote = MagicMock()
        mock_remote.remote.side_effect = RuntimeError("Unexpected error")
        mock_lookup.return_value = mock_remote

        with pytest.raises(
            Exception,
            match="An unexpected error occurred while running the LAMMPS simulation: Unexpected error",
        ):
            run_lammps.execute(input_file=input_file)


def test_run_lammps_with_real_file():
    # Replace this with the actual existing file path on your system
    actual_input_file = "/test_files/test_minimise/input.in"

    # Run the function without mocking - this will execute the real modal call
    result = run_lammps.execute(input_file=actual_input_file)

    expected_log_file = f"{Path(actual_input_file).stem}.log"
    expected_message = f"Simulation ran successfully using input: {actual_input_file}, log saved at: {expected_log_file}"

    assert result == expected_message


def test_run_lammps_with_none():
    with pytest.raises(ValueError, match="Input file path must not be None or empty."):
        run_lammps.execute(input_file=None)


def test_run_lammps_with_nonexistent_file():
    invalid_path = "/path/to/nonexistent/file.lammps"

    with pytest.raises(ValueError) as exc_info:
        run_lammps.execute(input_file=invalid_path)

    assert "lammps simulation failed" in str(exc_info.value).lower()
    assert "no such file or directory" in str(exc_info.value).lower()


@pytest.mark.skipif(
    not os.getenv("MP_API_KEY"),
    reason="MP_API_KEY not available in environment",
)
def test_get_structure_from_mp_text_real():
    mp_id = "mp-149"  # Silicon
    file_path = "/results/Si.cif"

    result = get_structure_from_mp_text.execute(mp_id=mp_id, file_path=file_path)

    assert result == f"Structure saved successfully at {file_path}"

    # Optionally, validate the content

    read_file = modal.Function.lookup("simagent", "read_file")
    cif_content = read_file.remote(file_path)

    assert "data_Si" in cif_content

    struct = Structure.from_str(cif_content, fmt="cif")
    assert struct.composition.reduced_formula == "Si"


@pytest.mark.skipif(
    not os.getenv("MP_API_KEY"),
    reason="MP_API_KEY not available in environment",
)
def test_get_structure_from_mp_text_invalid_id():
    invalid_mp_id = "mp-9999999"
    file_path = "/results/fake.cif"

    result = get_structure_from_mp_text.execute(
        mp_id=invalid_mp_id, file_path=file_path
    )

    assert isinstance(result, str)
    assert result.lower().startswith("failed to retrieve or save structure")
    assert (
        "mp-9999999" in result
        or "not found" in result.lower()
        or "no documents" in result.lower()
        or "list index out of range" in result.lower()
        or "no such material" in result.lower()
    )


def test_convert_structure_to_lammps_data_valid():
    structure_path = "/results/Si.cif"
    output_file = "/results/Si.data"

    result = convert_structure_to_lammps_data.execute(
        structure_path=structure_path, output_file=output_file, atom_style="full"
    )

    assert result == f"LAMMPS data file successfully written to: {output_file}"

    # Optionally check file content via modal API
    read_file = modal.Function.lookup("simagent", "read_file")
    data_content = read_file.remote(output_file)
    assert "Masses" in data_content or "Atoms" in data_content  # LAMMPS style sections


def test_convert_structure_to_lammps_data_invalid_input():
    invalid_structure_path = "/nonexistent/path/invalid.cif"
    output_file = "/results/invalid.data"

    with pytest.raises(Exception) as exc_info:
        convert_structure_to_lammps_data.execute(
            structure_path=invalid_structure_path,
            output_file=output_file,
            atom_style="atomic",
        )

    msg = str(exc_info.value).lower()
    assert "unexpected error" in msg
    assert "no such file" in msg or "not found" in msg or "failed" in msg


if __name__ == "__main__":
    pytest.main()
