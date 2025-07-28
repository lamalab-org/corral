import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import sys
import pytest
import pathlib
import modal
from dotenv import load_dotenv
import re
from pymatgen.core.structure import Structure
from io import StringIO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from md_simulations.tools import get_potential_metadata, get_structure_from_mp_text, convert_structure_to_lammps_data, run_lammps

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pymatgen.core import Molecule, Structure

def skip_if_no_api_key():
    """Skip test if MP_API_KEY is not available"""
    return pytest.mark.skipif(
        not os.getenv("MP_API_KEY"), reason="MP_API_KEY not available in environment"
    )

@skip_if_no_api_key()
def test_get_structure_from_mp_text():
    status_message = get_structure_from_mp_text.execute(mp_id="mp-149", file_path="/results/Si.cif")
    
    # Extract path from status message
    file_path = status_message.split(" at ", 1)[1].strip()

    # Remote read the file via modal
    read_file = modal.Function.lookup("simagent", "read_file")
    cif_data = read_file.remote(file_path)

    # Save to temporary file
    temp_path = "/tmp/retrieved_Si.cif"
    with open(temp_path, "w") as f:
        f.write(cif_data)

    # Load structures
    retrieved_structure = Structure.from_file(temp_path)
    ground_truth_structure = Structure.from_file("./structures/Si.cif")

    # Compare using pymatgen's structure matcher
    assert retrieved_structure == ground_truth_structure, "The structures are not equivalent!"

@pytest.mark.parametrize("filename,expected_metadata", [
    ("Al99.eam.alloy", "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}"),
])
def test_potential_metadata(filename, expected_metadata):
    metadata = get_potential_metadata.execute(filename)
    assert metadata == expected_metadata

def test_potential_metadata_invalid():
    with pytest.raises(ValueError, match="Unrecognized potential file: Unknown.eam"):
        get_potential_metadata.execute("Unknown.eam")

if __name__ == "__main__":
    import pytest
    # Run only tests with "test_potential_metadata" in their name
    pytest.main([__file__, "-v", "-k", "test_potential_metadata_invalid"])


