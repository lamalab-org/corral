from pathlib import Path
from unittest.mock import MagicMock, patch

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

    # Patch modal.Function.lookup to return an object whose remote is mock_remote
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_lookup.return_value.remote = mock_remote

        # Call the function
        result = run_lammps.execute(input_file=input_file)

        # Verify the remote was called correctly
        mock_remote.assert_called_once_with(input_file, expected_log_file)

        # Verify return value
        assert (
            result
            == f"Simulation ran successfully using input: {input_file}, log saved at: {expected_log_file}"
        )


def test_run_lammps_value_error():
    input_file = "input_file.data"

    # Patch modal.Function.lookup to return a mock object whose 'remote' raises ValueError
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()
        mock_func.remote.side_effect = ValueError("Some value error")
        mock_lookup.return_value = mock_func

        with pytest.raises(
            ValueError,
            match="The LAMMPS simulation failed with a ValueError: Some value error",
        ):
            run_lammps.execute(input_file=input_file)


def test_run_lammps_unexpected_exception():
    input_file = "input_file.data"

    # Patch 'modal.Function.lookup' to return a mock object whose remote raises RuntimeError
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()
        mock_func.remote.side_effect = RuntimeError("Unexpected error")
        mock_lookup.return_value = mock_func

        with pytest.raises(
            Exception,
            match="An unexpected error occurred while running the LAMMPS simulation: Unexpected error",
        ):
            run_lammps.execute(input_file=input_file)


def test_run_lammps_with_mocked_real_file():
    actual_input_file = "/test_files/test_minimise/input.in"
    expected_log_file = f"{Path(actual_input_file).stem}.log"

    # Patch Modal so we don't run real simulation
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()
        mock_func.remote.return_value = None  # simulate successful remote call
        mock_lookup.return_value = mock_func

        result = run_lammps.execute(input_file=actual_input_file)

        assert (
            result
            == f"Simulation ran successfully using input: {actual_input_file}, log saved at: {expected_log_file}"
        )

        # Ensure remote was called correctly
        mock_func.remote.assert_called_once_with(actual_input_file, expected_log_file)


def test_run_lammps_with_none():
    with pytest.raises(ValueError, match="Input file path must not be None or empty."):
        run_lammps.execute(input_file=None)


def test_run_lammps_with_nonexistent_file():
    invalid_path = "/path/to/nonexistent/file.lammps"

    with pytest.raises(ValueError) as exc_info:
        run_lammps.execute(input_file=invalid_path)

    assert "lammps simulation failed" in str(exc_info.value).lower()
    assert "no such file or directory" in str(exc_info.value).lower()


def test_get_structure_from_mp_text_mocked():
    mp_id = "mp-149"
    file_path = "/results/Si.cif"

    # Full valid CIF content for Silicon
    cif_content = """# generated using pymatgen
data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.44370237
_cell_length_b   5.44370237
_cell_length_c   5.44370237
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   Si8
_cell_volume   161.31810739
_cell_formula_units_Z   8
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_type_symbol
 _atom_type_oxidation_number
  Si0+  0.0
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si0+  Si0  1  0.75000000  0.75000000  0.25000000  1
  Si0+  Si1  1  0.00000000  0.50000000  0.50000000  1
  Si0+  Si2  1  0.75000000  0.25000000  0.75000000  1
  Si0+  Si3  1  0.00000000  0.00000000  0.00000000  1
  Si0+  Si4  1  0.25000000  0.75000000  0.75000000  1
  Si0+  Si5  1  0.50000000  0.50000000  0.00000000  1
  Si0+  Si6  1  0.25000000  0.25000000  0.25000000  1
  Si0+  Si7  1  0.50000000  0.00000000  0.50000000  1
"""

    # Patch Modal to simulate reading the CIF file
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()
        mock_func.remote.return_value = cif_content
        mock_lookup.return_value = mock_func

        # Call the function (mocked)
        result = get_structure_from_mp_text.execute(mp_id=mp_id, file_path=file_path)

        # Validate return message
        assert result == f"Structure saved successfully at {file_path}"

        # Simulate reading the file via Modal
        content = mock_lookup.return_value.remote(file_path)
        assert "data_Si" in content

        # Parse CIF content and check composition
        struct = Structure.from_str(content, fmt="cif")
        assert struct.composition.reduced_formula == "Si"


def test_get_structure_from_mp_text_invalid_id_mocked():
    invalid_mp_id = "mp-9999999"
    file_path = "/results/fake.cif"

    # Patch the Modal lookup to simulate failure
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()

        # Simulate raising an exception when trying to save a CIF
        def fake_remote(*_args, **_kwargs):
            raise ValueError(f"Material {invalid_mp_id} not found")

        mock_func.remote.side_effect = fake_remote
        mock_lookup.return_value = mock_func

        # Call the function
        result = get_structure_from_mp_text.execute(
            mp_id=invalid_mp_id, file_path=file_path
        )

        # Validate the returned error message
        assert isinstance(result, str)
        assert result.lower().startswith("failed to retrieve or save structure")
        assert (
            invalid_mp_id in result
            or "not found" in result.lower()
            or "no documents" in result.lower()
            or "list index out of range" in result.lower()
            or "no such material" in result.lower()
        )


def test_convert_structure_to_lammps_data_mocked():
    structure_path = "/results/Si.cif"
    output_file = "/results/Si.data"

    # Dummy LAMMPS data content
    lammps_data_content = """LAMMPS data file via pymatgen
1 atoms
1 atom types

Masses

1 28.0855

Atoms

1 1 0.0 0.0 0.0
"""

    # Patch Modal lookup to simulate reading/writing
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()
        mock_func.remote.return_value = lammps_data_content
        mock_lookup.return_value = mock_func

        # Call the function
        result = convert_structure_to_lammps_data.execute(
            structure_path=structure_path, output_file=output_file, atom_style="full"
        )

        # Check returned message
        assert result == f"LAMMPS data file successfully written to: {output_file}"

        # Simulate reading file content via modal
        data_content = mock_lookup.return_value.remote(output_file)
        assert "Masses" in data_content or "Atoms" in data_content


def test_convert_structure_to_lammps_data_invalid_input_mocked():
    invalid_structure_path = "/nonexistent/path/invalid.cif"
    output_file = "/results/invalid.data"

    # Patch Modal lookup to simulate failure when reading invalid structure
    with patch("corral_md.tools.modal.Function.lookup") as mock_lookup:
        mock_func = MagicMock()

        # Simulate remote call raising FileNotFoundError
        mock_func.remote.side_effect = FileNotFoundError(
            f"No such file: {invalid_structure_path}"
        )
        mock_lookup.return_value = mock_func

        # Expect exception from convert_structure_to_lammps_data
        with pytest.raises(Exception) as exc_info:
            convert_structure_to_lammps_data.execute(
                structure_path=invalid_structure_path,
                output_file=output_file,
                atom_style="atomic",
            )

        msg = str(exc_info.value).lower()
        assert "unexpected error" in msg or "failed" in msg
        assert "no such file" in msg or "not found" in msg


if __name__ == "__main__":
    pytest.main()
