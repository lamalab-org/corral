
from corral.base import Tool, ToolArgument
from corral.utils import tool
from typing import Dict
from dotenv import load_dotenv
import os
import modal
from loguru import logger
import subprocess

@tool
def get_potential_metadata(file_path: str) -> str:
    """
    Get the metadata of a potential file. This function reads the potential file and extracts
    the metadata, and other relevant information.

    Args:
        file_path: Path to the potential file.

    Returns:
        A string containing the metadata of the potential file.
    """

    potential_name = file_path.split("/")[-1]
    if potential_name == "ffield.reax":
        return "{potential type : reax, elements supported : Carbon (C), Hydrogen (H), Oxygen (O), Calcium (Ca), Silicon (Si), pair_style : reaxff}"
    elif potential_name == "Al99.eam.alloy":
        return "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}"
    elif potential_name == "Cu_Zhou04.eam.alloy":
        return "{potential type : EAM, elements supported : Cu (Copper), pair_style : eam/alloy}"
    elif potential_name == "Mg_Zhou04.eam.alloy":
        return "{potential type : EAM, elements supported : Mg (Magnesium), pair_style : eam/alloy}"
    elif potential_name == "Fe-C_Hepburn_Ackland.eam.fs":
        return "{potential type : EAM, elements supported : Fe (Iron), C (Carbon), pair_style : eam/fs}"
    else:
        raise ValueError(f"Unrecognized potential file: {potential_name}")
    
@tool
def get_structure_from_mp_text(mp_id: str, file_path: str) -> str:
    """
    Retrieve the conventional unit cell structure of a material from the Materials Project
    and return its CIF content as a text string. This function queries the Materials Project database for a given material ID,
    obtains the primitive structure, converts it to the conventional crystallographic
    structure using symmetry analysis, and then saves it in the given file_path in cif format.

    Args:
        mp_id: Materials Project id.
        file_path: The path where the CIF file will be saved.

    Returns:
        Returns the status message.
    """
    try:
        from mp_api.client import MPRester
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with MPRester("3pKn435e6fN6hfcOEfpYX4OqbnfV1MKB") as mpr:
            docs = mpr.materials.summary.search(
                material_ids=[str(mp_id)], fields=["structure"]
            )
            structure = docs[0].structure

        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()
        structure_cif = structure.to(fmt="cif")

        write_file_sim = modal.Function.lookup("simagent", "write_file")
        write_file_sim.remote(file_path, structure_cif)

        return f"Structure saved successfully at {file_path}"

    except Exception as e:
        return f"Failed to retrieve or save structure: {str(e)}"
    

@tool
def convert_structure_to_lammps_data(structure: str, output_file: str, atom_style: str = "charge") -> str:
    """
    Convert a CIF-format structure file into a LAMMPS data file. This function takes the file path of the CIF structure as input,
    converts it to a pymatgen Structure object, and then generates the corresponding LAMMPS data
    file.

    Args:
        structure: The file path of the CIF-format structure.
        output_file: The path to the output LAMMPS data file to be written.
        atom_style: The atom style for LAMMPS. Common values include:
            - "atomic": Basic atomic positions and masses.
            - "charge": Atoms with charge data (default).

    Returns:
        None. Writes the LAMMPS data to the specified output file.
    """

    try:
        convert_structure_to_lammps_data_sim = modal.Function.lookup("simagent", "convert_structure_to_lammps_data")
        convert_structure_to_lammps_data_sim.remote(structure, output_file, atom_style)
        return f"LAMMPS data file successfully written to: {output_file}"
    except Exception as e:
        # Handle unexpected errors
        raise Exception(f"An unexpected error occurred while converting structure to LAMMPS data: {str(e)}")
    
@tool
def run_lammps(input_file: str, num_cpus: int = 1) -> str:
    """Run a LAMMPS simulation. A LAMMPS log file is generated in the same directory as the input file, with the same name but a .log extension. For example, if the input file is named input.lammps, the log file will be named input.log.

    Args:
        input_file: Path to the LAMMPS input script file.
        num_cpus: Number of CPUs to use for the simulation. Default is 1. Maximum allowed CPUs is 4.
    """
    try:
        file_name_without_extension = os.path.splitext(os.path.basename(input_file))[0]
        log_file = f"{file_name_without_extension}.log"
        run_lammps_sim = modal.Function.lookup("simagent", "run_lammps")
        run_lammps_sim.remote(input_file, log_file, num_cpus)
        return f"Simulation ran successfully using input: {input_file}, log saved at: {log_file}"

    except ValueError as e:
        # Raise a ValueError with more context about the failure
        raise ValueError(f"The LAMMPS simulation failed with a ValueError: {str(e)}")
    except Exception as e:
        # Handle unexpected errors
        raise Exception(f"An unexpected error occurred while running the LAMMPS simulation: {str(e)}")

@tool
def extract_max_stress(file_path: str) -> float:
    """Extracts the maximum tensile stress (in GPa) along the x-direction from a space-delimited stress-strain data file with a header.

    Args:
        file_path (str): Path to the .txt file containing stress-strain data.

    Returns:
        float: Maximum stress_xx value in GPa.
    """
    try: 
        extract_max_stress_ = modal.Function.lookup("simagent", "extract_max_stress")  
        return extract_max_stress_.remote(file_path)
    except Exception as e:
        raise Exception(f"An unexpected error occurred while extracting maximum stress value: {str(e)}")
