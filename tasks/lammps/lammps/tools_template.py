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
    [BRIEF] Returns metadata from a known potential file. 
    [/BRIEF]

    [DETAILED] This tool provides a quick and reliable way to identify the type and supported elements 
    of a LAMMPS potential file based solely on its filename. It eliminates the need to parse the often 
    large and complex contents of the potential files, which can often exceed the processing limits of 
    many systems or applications. By returning a structured description, this tool enables the user to 
    determine whether a given potential file is appropriate for a specific molecular dynamics (MD) 
    simulation. This is especially useful when selecting the correct interatomic potential for a system 
    involving specific elements, without having to inspect the file manually or load it entirely. 
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to quickly determine the type and supported elements of a LAMMPS potential file based on its filename.
    - Best suited for selecting an appropriate potential file for a specific molecular dynamics (MD) simulation without reading or parsing the full file contents.
    - Recommended for gaining a fast, structured overview of a potential file’s applicability to specific element combinations or simulation scenarios.
    [/PROCEDURAL]

    [CONTEXTUAL] 
    How this tool works:
    - Extracts the file name from the provided file path
    - Matches it against a set of known file names
    - Returns a structured metadata string for recognized files
    - Raises a ValueError if the file name is unrecognized
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1: get_potential_metadata("sim_data/ffield.reax")
    [/SYNTACTICAL]

    
    Args:
        file_path: [BRIEF] Absolute path to the potential file. [/BRIEF]
                   [DETAILED] This is the absolute path to a LAMMPS-compatible potential file (e.g., ReaxFF or EAM formats). The file name is used to determine metadata, so it must match one of the known patterns. [/DETAILED]
                   [SYNTACTIC] Format: string ending in a recognized potential filename [/SYNTACTIC]
                   [EXAMPLES] Examples: "/path/to/file/ffield.reax", "ffield_UTA1.ITT" [/EXAMPLES]                      
    Returns:
        str : [BRIEF] Structured metadata string describing the potential file. [/BRIEF]
              [DETAILED] The returned string includes the type of interatomic potential and a list of chemical elements that it supports. This helps in choosing suitable potentials for simulations involving specific atoms. [/DETAILED]
              [EXAMPLES] Example outputs: "{potential type : reax, elements supported : Carbon (C), Hydrogen (H), Oxygen (O), Calcium (Ca), Silicon (Si), pair_style : reaxff}" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [BRIEF] If the file name is not recognized. [/BRIEF]
                    [DETAILED] Raised when the filename does not match any known potential files. This helps prevent silent failures and makes debugging easier in automated workflows. [/DETAILED]
    [/RAISES]

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
    [BRIEF] Retrieves and saves the conventional crystal structure of a material from the Materials Project as a CIF file.
    [/BRIEF]

    [DETAILED] This tool retrieves the conventional unit cell structure of a material from the Materials Project 
    using its material ID. It transforms the primitive structure returned by the database into its conventional 
    crystallographic form using symmetry operations, and exports the result in CIF format to a specified file path.
    This tool is valuable for workflows that require standardized crystal structure representations — such as 
    simulations, visualization, structure matching, or publication. It avoids the need for manual structure 
    transformation or dealing with primitive cells when conventional representation is needed.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need the conventional crystallographic (not primitive) structure of a material from the Materials Project.
    - Best suited for preparing simulation-ready input files, visualizing crystal structures, or storing standardized CIFs.
    - Recommended for quick and automated generation of conventional structure files for structure-based computation or crystallographic analysis.
    [/PROCEDURAL] 

    [CONTEXTUAL] How this tool works:
    - Step 1: Queries the Materials Project database using the provided material ID to retrieve the material's primitive crystal structure.
    - Step 2: Applies symmetry analysis to convert the primitive structure into its conventional crystallographic form.
    - Step 3: Converts the conventional structure into CIF (Crystallographic Information File) format.
    - Step 4: Saves the CIF content to the specified file path using an external `write_file` function.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1: get_structure_from_mp_text("mp-149", "/workspace/Si_conventional.cif")
    - Example 2: get_structure_from_mp_text("mp-13", "outputs/Aluminum_structure.cif")
    [/SYNTACTICAL]

    Args:
        mp_id:  [BRIEF] Materials Project ID of the material. [/BRIEF]
                [DETAILED] A unique identifier used by the Materials Project database to reference a material. The ID typically starts with "mp-" followed by digits. It must correspond to an existing entry. [/DETAILED]
                [SYNTACTIC] Format: "mp-XXXX" where X is a digit [/SYNTACTIC]
                [EXAMPLES] Examples: "mp-149", "mp-13", "mp-1234567" [/EXAMPLES]
        file_path: [BRIEF] Destination path for saving the CIF file. [/BRIEF]
                   [DETAILED] Absolute path to the file where the CIF content will be written. [/DETAILED]
                   [SYNTACTIC] Format: string path ending in ".cif" [/SYNTACTIC]
                   [EXAMPLES] Examples: "/tmp/output.cif", "structure_files/Al.cif" [/EXAMPLES]                    
    Returns:
        str : [BRIEF] Status message indicating successful structure retrieval and saving at required path. [/BRIEF]
              [DETAILED] If successful, the tool retrieves the conventional crystallographic structure for the given Materials Project ID,
              converts it into CIF format, saves it at the specified path, 
              and returns a confirmation message. If any step fails, a descriptive error message is returned instead. [/DETAILED]
              [EXAMPLES] Example outputs:
                - "Structure saved successfully at /workspace/data/structure.cif"
                - "Failed to retrieve or save structure: Invalid Materials Project ID" [/EXAMPLES]
                   
    [RAISES] Exceptions:
        Exception: [BRIEF] Raised if structure retrieval or file saving fails. [/BRIEF]
                   [DETAILED] This generic exception is returned if any error occurs during Materials Project API access, 
                   structure conversion, or remote file write. The error message is descriptive and includes the failure reason. [/DETAILED]
    [/RAISES]


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
    [BRIEF] Converts a CIF-format structure file into a LAMMPS data file using a specified atom style. [/BRIEF]

    [DETAILED] This tool converts a crystal structure provided in CIF format (as a file) into a LAMMPS-compatible data file. Internally, it reads the CIF structure file using pymatgen, transforms it into a Structure object, and then serializes it to a LAMMPS data format using the LammpsData class. It supports configurable atom styles such as "atomic" or "charge", allowing flexibility based on simulation requirements. This enables seamless transformation of standardized crystallographic data into simulation-ready LAMMPS input files, streamlining the setup process for molecular dynamics workflows. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to convert crystallographic structure data in CIF format into a LAMMPS-compatible `.data` file.
    - Best suited for setting up molecular dynamics simulations where LAMMPS is the engine, and structure data is sourced from databases like Materials Project or experimental CIFs.
    - Avoid when your structure data is already in LAMMPS format or requires extensive pre-processing (e.g., force field assignments).
    - Recommended for automating simulation pipelines that begin with standardized structural data and end in LAMMPS-ready input formats.  
    [/PROCEDURAL] 


    [CONTEXTUAL] How this tool works:
    - Step 1: The tool reads the `.cif` file using `pymatgen.core.Structure.from_file` to create a Structure object.
    - Step 2: The `LammpsData.from_structure` function is called, using the chosen `atom_style` to format the atomic data accordingly.
    - Step 3: The resulting LAMMPS data file is written to the path specified by `output_file`.
    - Behavior note: The tool does not apply force fields or assign charges explicitly; it assumes the structure is already complete and appropriate for the selected `atom_style`. 
    [/CONTEXTUAL]

    [SYNTACTICAL] 
    - Example 1: convert_structure_to_lammps_data("/workspace/graphene.cif", "/workspace/output/graphene.data")
    - Example 2: convert_structure_to_lammps_data("./structures/NaCl.cif", "./data/NaCl.data", atom_style="atomic")
    [/SYNTACTICAL]

    Args:
        structure: [BRIEF] Path to the CIF-format structure file. [/BRIEF]
                   [DETAILED] Path to the file containing the crystallographic structure. This file is read and 
                    converted into a pymatgen `Structure` object internally before being serialized to LAMMPS data format. [/DETAILED]
                   [SYNTACTIC] Format: string ending in ".cif" [/SYNTACTIC]
                   [EXAMPLES] Examples: "/workspace/graphene.cif", "./data/SiO2.cif" [/EXAMPLES]
        output_file: [BRIEF] Path where the LAMMPS data file will be saved. [/BRIEF]
                     [DETAILED] This is the destination file path where the generated LAMMPS-compatible data file will be written. 
                     The output file will contain the atomic positions, types, and other necessary information formatted for LAMMPS simulations. [/DETAILED]
                     [SYNTACTIC] Format: valid string representing a writable `.data` file path. [/SYNTACTIC]
                     [EXAMPLES] Examples:  
                         - "/workspace/output/graphene.data",  
                         -"./converted_data/SiO2.data" [/EXAMPLES]
        atom_style: [BRIEF] Atom style to be used in the LAMMPS data file, defaults to "charge". [/BRIEF]
                    [DETAILED] Specifies the LAMMPS atom style to use when formatting the data file. Common values include:
                    - "atomic": Includes atomic positions and mass, no charges.
                    - "charge": Includes atomic charges in addition to position and mass.
                    The choice of style should match the `atom_style` directive in the LAMMPS input script. Defaults to "charge" if not provided. [/DETAILED]
                    [SYNTACTIC] Format: one of the predefined LAMMPS atom styles as a lowercase string. [/SYNTACTIC]
                    [EXAMPLES] Examples: "real", "metal". [/EXAMPLES]
                    [CHOICES] Valid values: "real", "metal", "si", "cgs", "electron", "micro", "nano", "". [/CHOICES]

    [RETURNS]
        str:
            [BRIEF] Status message indicating successful LAMMPS data file generation. [/BRIEF]
            [DETAILED] If the conversion is successful, returns a confirmation message specifying the path where the LAMMPS data file has been saved. This message can be used for logging or downstream validation in automated simulation workflows. [/DETAILED]
            [EXAMPLES] Example outputs:  
                - "LAMMPS data file successfully written to: /workspace/output/graphene.data"  
                - "LAMMPS data file successfully written to: ./converted_data/SiO2.data" [/EXAMPLES]
    [/RETURNS]

    [RAISES] Exceptions:
        Exception:
            [BRIEF] Raised if an unexpected error occurs during structure conversion. [/BRIEF]
            [DETAILED] If the structure cannot be parsed or the LAMMPS data file generation fails (e.g., due to malformed CIF, file I/O issues, or internal conversion errors), a generic Exception is raised with a descriptive message for debugging. [/DETAILED]
    [/RAISES]

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
    """
    [BRIEF] Runs a LAMMPS simulation based on the provided input script and generates a corresponding log file. [/BRIEF]
    
    [DETAILED] This tool executes a LAMMPS molecular dynamics simulation using a specified input script.  
    It takes the path to a LAMMPS input file and automatically triggers the simulation run through a remote execution backend with specified CPU resources.  
    The tool also generates a corresponding log file (named after the input script, with `.log` extension replacing the original extension)  
    which contains detailed simulation output including thermodynamic data, errors (if any), and runtime diagnostics. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute a LAMMPS molecular dynamics simulation using a predefined input script.
    - Recommended for production simulations, high-throughput screening, and automated pipelines where simulation setup is complete and ready to run.
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Step 1: The input LAMMPS script (typically ending in `.in`) is passed to the remote backend, along with the number of CPUs to use for the simulation.
    - Step 2: The tool constructs a log file name by replacing the script’s extension with `.log`.
    - Step 3: A remote function is invoked with the input file, the log file path, and the number of CPUs.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - Example 1: run_lammps("/workspace/lammps_inputs/graphene_sim.in", 1)
    - Example 2: run_lammps("./data/sio2_minimize.in", 2)
    [/SYNTACTICAL]

    Args:
        input_file: [BRIEF] Path to the LAMMPS input script file. [/BRIEF]
                    [DETAILED] This parameter specifies the absolute or relative path to the input script used by LAMMPS. 
                    The script typically contains simulation settings such as atom style, force field parameters, boundary conditions, and compute directives.
                    The file must be in LAMMPS-compatible format (`.in` extension is conventional but not required) and should not require interactive input during execution. [/DETAILED]
                    [SYNTACTIC] Format: string representing a file path; must be readable by the backend LAMMPS engine. [/SYNTACTIC]
                    [EXAMPLES] Examples: 
                        - "/workspace/lammps_inputs/graphene_sim.in"  
                        - "./simulations/liquid_water.in"  
                        - "minimize_bulk_sio2.in" [/EXAMPLES]
        num_cpus: [BRIEF] Number of CPUs to use for the simulation, default is 1. [/BRIEF]
                  [DETAILED] Specifies how many CPU cores to allocate for the LAMMPS simulation. 
                    This parameter allows parallel execution of the simulation, which can significantly speed up computations for large systems. 
                    The default value is 1, meaning the simulation will run on a single CPU core. 
                    If the backend supports it, this can be increased to utilize multiple cores (up to 4 in this case). [/DETAILED]
                  [SYNTACTIC] Format: integer value representing the number of CPU cores to allocate. [/SYNTACTIC]
                  [EXAMPLES] Examples: 
                        - 1 (default, single-core execution)
                        - 2 (dual-core execution)
                        - 4 (quad-core execution, maximum allowed)
                  [/EXAMPLES]

    Returns:
        str: 
            [BRIEF] Message indicating simulation completion with log file location. [/BRIEF] 
            [DETAILED] On success, returns a message confirming the simulation run, the path to the latest input script used, and the corresponding log file. 
            The log file contains detailed runtime diagnostics and output for verification. [/DETAILED]
            [EXAMPLES] 
                - "Simulation ran successfully using input: simulations/run_graphene.in, log saved at: run_graphene.log"
                - "Simulation ran successfully using input: ./jobs/job123.lmp, log saved at: job123.log" [/EXAMPLES]


    [RAISES] Exceptions:
        ValueError: 
            [BRIEF] Raised if LAMMPS simulation fails due to invalid input or LAMMPS-specific error. [/BRIEF]
            [DETAILED] Triggered when the underlying LAMMPS execution raises a ValueError — for example, due to missing sections in the input file or invalid parameters. [/DETAILED]
        
        Exception: 
            [BRIEF] Raised on unexpected backend or runtime errors. [/BRIEF]
            [DETAILED] Captures unanticipated issues such as file system errors, backend unavailability, or misconfigured modal runtime. 
            Returns a descriptive error message for debugging. [/DETAILED]
    [/RAISES]

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
def extract_max_stress(file_path) -> float:
    """
    [BRIEF] Extracts the maximum tensile stress (GPa) along the x-direction from a stress-strain data file. [/BRIEF]

    [DETAILED] This tool processes a space-delimited text file containing stress-strain data with a header row.  
    It reads the file, parses the stress tensor components, and identifies the maximum tensile stress component along the x-axis (stress_xx) in units of GPa.  
    This value is important for materials mechanical property analysis and failure prediction. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when analyzing output stress-strain data from simulations or experiments.
    - Best suited for datasets formatted as space-delimited text files with a header.
    - Recommended for quick extraction of peak stress values for materials screening. [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Step 1: Reads the stress-strain data file as text.
    - Step 2: Parses numeric columns identifying stress_xx.
    - Step 3: Computes and returns the maximum tensile stress_xx value (in GPa).
    - Handles exceptions for file read errors or unexpected file formatting. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    - extract_max_stress("results/stress_strain_01.txt")
    - extract_max_stress("/data/graphene_stress.txt") [/SYNTACTICAL]

    Args:
        file_path: 
            [BRIEF] Absolute Path to the stress-strain data text file. [/BRIEF]
            [DETAILED] Absolute path, and should point to a valid, space-delimited `.txt` file containing stress-strain data with a header. 
            The file must include a stress_xx column or equivalent. [/DETAILED]
            [SYNTACTIC] Format: string file path ending in `.txt` [/SYNTACTIC]
            [EXAMPLES] Examples: "data/Al_stress.txt", "/sim_outputs/sample_stress.txt" [/EXAMPLES]

    Returns:
        float: 
            [BRIEF] Maximum tensile stress_xx value in GPa. [/BRIEF]
            [DETAILED] Returns the peak stress along the x-direction extracted from the dataset. 
            This is a floating-point value representing gigapascals (GPa). [/DETAILED]
            [EXAMPLES] Example outputs: 12.45, 56.78, 98.12 [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [BRIEF] Raised if file reading or parsing fails. [/BRIEF]
            [DETAILED] This may occur due to missing files, invalid formatting, or unexpected data structures in the input file. [/DETAILED]
    [/RAISES]
    """
    try: 
        extract_max_stress_ = modal.Function.lookup("simagent", "extract_max_stress")  
        return extract_max_stress_.remote(file_path)
    except Exception as e:
        raise Exception(f"An unexpected error occurred while extracting maximum stress value: {str(e)}")