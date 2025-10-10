"""
MD Tutorials Tools Module

This module provides a comprehensive set of tools for molecular dynamics (MD) simulations
using LAMMPS. It includes functionality for structure preparation, simulation execution,
and results analysis in materials science workflows.
"""

import os
from pathlib import Path

import modal
from loguru import logger

from corral.backend.tool import tool


@tool
def execute_python_script(
    script_path: str,
    args: list | None = None,
    timeout: int = 600,
    working_dir: str | None = None,
) -> str:
    """[BRIEF] Execute a Python script file with arguments in a controlled environment. [/BRIEF]

    [DETAILED] This tool executes existing Python script files with command-line arguments, providing a controlled environment for running complex analysis workflows, data processing pipelines, or computational simulations.
    It captures all output streams and provides comprehensive execution monitoring with timeout protection.
    This is essential for integrating existing Python scripts into automated workflows and materials analysis pipelines. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute existing Python scripts with specific arguments. You can also use io tool to write a script and then execute it.
    - Best suited for running complex analysis workflows or simulations
    - Essential for integrating external Python tools into automated pipelines
    - Recommended for batch processing and computational workflows
    - Avoid for simple code execution
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Validates script file existence and accessibility
    - Constructs command with script path and provided arguments
    - Executes script in subprocess with timeout protection
    - Captures standard output, error streams, and return codes
    - Provides comprehensive execution monitoring and error reporting
    - Supports custom working directory for script execution
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Ensure script file exists and is executable with proper dependencies [/PREREQUISITE]
    2. [CURRENT] Execute script with appropriate arguments and timeout [/CURRENT]
    3. [FOLLOW_UP] Process script output and results for further analysis. Can be used to process json script as required [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `execute_python_script("analysis.py", ["--input", "data.json", "--output", "results.json"], 300)`,
    `execute_python_script("simulation.py", ["--steps", "1000", "--temp", "300"], 1800, "/path/to/workdir")`,
    `execute_python_script("processing.py", None, 600, None)`,
    [/SYNTACTICAL]

    Args:
        script_path: [BRIEF] Path to the Python script file to execute. [/BRIEF]
                    [DETAILED] Complete file path to the Python script that should be executed.
                    The script must exist and be readable.
                    The path can be relative to the current working directory or absolute.
                    The script should be a valid Python file with appropriate shebang or run using the Python interpreter. [/DETAILED]
                    [SYNTACTIC] "Valid file path to Python script" [/SYNTACTIC]
                    [EXAMPLES] "scripts/analysis.py", "/home/user/simulations/run_sim.py", "data_processing.py" [/EXAMPLES]
        args: [BRIEF] Optional list of command-line arguments for the script. [/BRIEF]
             [DETAILED] A list of strings representing command-line arguments to pass to the script.
             These arguments will be passed to the script in the order provided.
             Common arguments include input files, output paths, configuration parameters, and processing options.
             If None, the script will be executed without arguments. [/DETAILED]
             [SYNTACTIC] ["arg1", "arg2", "arg3", ...] or None [/SYNTACTIC]
             [EXAMPLES] ["--input", "data.json"], ["--verbose", "--output", "results.csv"], None [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 600. [/BRIEF]
                [DETAILED] The maximum time in seconds the script is allowed to run before being terminated.
                This prevents runaway processes and ensures resource management.
                Choose appropriate values based on expected script execution time.
                For computational simulations, longer timeouts may be necessary. [/DETAILED]
                [SYNTACTIC] positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] 300 (5 minutes), 600 (10 minutes), 3600 (1 hour) [/EXAMPLES]
        working_dir: [BRIEF] Optional working directory for script execution. [/BRIEF]
                    [DETAILED] The directory from which the script should be executed.
                    This affects relative path resolution and file I/O operations within the script.
                    If None, the current working directory will be used.
                    This is useful when scripts expect to run from specific directories or access relative files. [/DETAILED]
                    [SYNTACTIC] Valid directory path or None [/SYNTACTIC]
                    [EXAMPLES] "/path/to/project", "data/analysis", None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive execution results and monitoring data. [/BRIEF]
             [DETAILED] A JSON-formatted string containing execution status, captured output streams, error messages, return code, and the complete command that was executed.
             This provides full visibility into the script execution process and enables debugging and monitoring of automated workflows. [/DETAILED]
             [EXAMPLES] "{"success": true, "stdout": "Processing complete", "stderr": "", "return_code": 0, "command": "python script.py --input data.json"}" [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When the specified script file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Script path is invalid or file is not accessible [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify script path exists and is readable [/ERROR_RECOVERY]
        TimeoutExpired: [ERROR_WHEN] When script execution exceeds the specified timeout [/ERROR_WHEN]
                       [ERROR_DETAILS] Script terminated due to timeout limit [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase timeout value or optimize script performance [/ERROR_RECOVERY]
        PermissionError: [ERROR_WHEN] When script file lacks execute permissions [/ERROR_WHEN]
                        [ERROR_DETAILS] Insufficient permissions to execute the script [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check file permissions and ensure script is executable [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Cannot modify script execution environment beyond working directory
    - Limited to Python scripts and available system Python installation
    - No real-time output streaming during execution
    - Cannot interact with scripts requiring user input
    [/LIMITATIONS]
    """
    try:
        logger.info(f"script path {script_path}")
        execute_code_script = modal.Function.lookup("simagent", "execute_python_script")
        return execute_code_script.remote(
            script_path=script_path, args=args, timeout=timeout, working_dir=working_dir
        )
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while executing the code: {e!s}"
        ) from e


@tool
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    """[BRIEF] Execute Python code in a secure environment with data input/output capabilities. [/BRIEF]

    [DETAILED] This tool provides a secure execution environment for custom Python code, essential for data analysis, custom calculations, and algorithm development in materials science workflows.
    It supports data injection, output capture, and file saving capabilities while maintaining security through process isolation and timeout controls.
    This enables flexible custom analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute custom Python analysis or calculations
    - Best suited for data processing and custom algorithm development
    - Essential for implementing custom filtering, analysis, or transformation logic
    - Recommended for prototyping and testing analysis workflows
    - Avoid for simple operations that can be done with existing tools
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Creates isolated subprocess environment for secure code execution
    - Injects input data as JSON-parsed variable if provided
    - Captures standard output, error streams, and execution results
    - Implements timeout protection to prevent infinite loops
    - Extracts variables from executed code for result capture
    - Saves results to file if requested for persistence
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Prepare input data and ensure code is syntactically correct [/PREREQUISITE]
    2. [CURRENT] Execute custom Python code with data processing or analysis [/CURRENT]
    3. [FOLLOW_UP] Use captured results for further analysis or save to files [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `execute_python_code("result = sum([1, 2, 3, 4, 5])", None, None, 30)`,
    `execute_python_code("filtered_data = [x for x in input_data if x > 0.5]", json_data, "output.json")`,
    `execute_python_code("import numpy as np; result = np.mean(input_data)", array_data, None, 60)`,
    [/SYNTACTICAL]

    Args:
        python_code: [BRIEF] Python code string to be executed. [/BRIEF]
                    [DETAILED] A string containing valid Python code to be executed in this environment.
                    For best results, assign your main output to a variable named 'result' or 'output'.
                    The code can import standard libraries and perform complex calculations.
                    The tool will attempt to capture user-defined variables as execution results. [/DETAILED]
                    [SYNTACTIC] Valid Python code string [/SYNTACTIC]
                    [EXAMPLES] "result = 2 + 2", "import json; result = json.loads(data)", "filtered = [x for x in data if x > threshold]" [/EXAMPLES]
        input_data: [BRIEF] Optional JSON string to inject as input_data variable. [/BRIEF]
                   [DETAILED] An optional JSON string that will be loaded into a Python variable named 'input_data' within the executed script.
                   This allows the script to process external data.
                   The JSON will be parsed and made available as a Python object (dict, list, etc.) depending on the JSON structure. [/DETAILED]
                   [SYNTACTIC] Valid JSON string or None [/SYNTACTIC]
                   [EXAMPLES] "{"data": [1, 2, 3]}", '[1, 2, 3, 4, 5]', "{"threshold": 0.5, "values": [...]}" [/EXAMPLES]
        save_output_to: [BRIEF] Optional file path to save execution results. [/BRIEF]
                       [DETAILED] An optional file path where the captured execution results will be saved as a JSON file.
                       If provided and execution is successful, the results will be written to this file for persistence and later use.
                       The directory will be created if it doesn't exist. [/DETAILED]
                       [SYNTACTIC] "Valid file path or None" [/SYNTACTIC]
                       [EXAMPLES] "results.json", "output/analysis_results.json", "data/processed_output.json" [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 300. [/BRIEF]
                [DETAILED] The maximum time in seconds the subprocess is allowed to run before being terminated.
                This prevents infinite loops and runaway processes from consuming system resources.
                If the execution exceeds this limit, a timeout error will be returned.
                Choose appropriate values based on expected computation time. [/DETAILED]
                [SYNTACTIC] positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] 30 (quick calculations), 300 (standard), 1800 (long processing) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with detailed execution results and captured output. [/BRIEF]
             [DETAILED] A comprehensive JSON string containing execution status, standard output, error messages, return code, captured execution results, and file save status.
             The execution_result field contains variables captured from the executed code.
             This enables full visibility into the execution process and results. [/DETAILED]
             [EXAMPLES] "{"success": true, "execution_result": {"result": 10}, "stdout": "...", "stderr": "", "return_code": 0}" [/EXAMPLES]

    [RAISES] Exceptions:
        TimeoutExpired: [ERROR_WHEN] When code execution exceeds the specified timeout [/ERROR_WHEN]
                       [ERROR_DETAILS] Process terminated due to timeout limit [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase timeout value or optimize code for faster execution [/ERROR_RECOVERY]
        SyntaxError: [ERROR_WHEN] When the Python code contains syntax errors [/ERROR_WHEN]
                    [ERROR_DETAILS] Invalid Python syntax in the code string [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check code syntax and fix any errors [/ERROR_RECOVERY]
        RuntimeError: [ERROR_WHEN] When code execution fails due to runtime errors [/ERROR_WHEN]
                     [ERROR_DETAILS] Errors during code execution such as undefined variables [/ERROR_DETAILS]
                     [ERROR_RECOVERY] Debug code logic and ensure all required variables are defined [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to Python standard library and commonly available packages
    - Cannot access external network resources or file system outside working directory
    - Cannot install new packages during execution
    - Does not persist state between executions
    [/LIMITATIONS]
    """
    try:
        execute_code = modal.Function.lookup("simagent", "execute_python_code")
        return execute_code.remote(
            python_code=python_code,
            input_data=input_data,
            save_output_to=save_output_to,
            timeout=timeout,
        )
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while executing the code: {e!s}"
        ) from e


@tool
def get_potential_metadata(file_path: str) -> str:
    """
    [BRIEF] Returns metadata from a known LAMMPS potential file given the file path. The metadata includes potential type, elements supported, and the LAMMPS compatible pair style keyword. [/BRIEF]

    [DETAILED] This tool provides a quick and reliable way to identify the type and supported elements of a LAMMPS potential file based solely on its filename.
    It eliminates the need to parse the often large and complex contents of the potential files, which can often exceed the processing limits of many systems or applications.
    By returning a structured description, this tool enables the user to determine whether a given potential file is appropriate for a specific molecular dynamics (MD) simulation.
    This is especially useful when selecting the correct interatomic potential for a system involving specific elements, without having to inspect the file manually or load it entirely.
    The metadata includes the type of potential, which elements are supported, as well as the LAMMPS pair style to be specified while writing simulation script.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to quickly determine the type, supported elements or the LAMMPS pair style of a LAMMPS potential file based on its filename.
        - Best suited for selecting an appropriate potential file for a specific molecular dynamics (MD) simulation without reading or parsing the full file contents.
        - Recommended for gaining a fast, structured overview of a potential file's applicability to specific element combinations or simulation scenarios.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Select the potential files that might be relevant to your simulation task. [/PREREQUISITE]
        2. [CURRENT] Use this tool to retrieve metadata for each potential file based on its filename. This will help you quickly identify which potentials are suitable for your simulation needs. [/CURRENT]
        3. [FOLLOW_UP] Based on the metadata returned, choose the appropriate potential file for your simulation setup, and run the simulation using the potential file and the tool `run_lammps`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Extracts the file name from the provided file path
        - Matches it against a set of known file names
        - Returns a structured metadata string for recognized files
        - Raises a ValueError if the file name is unrecognized
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_potential_metadata("sim_data/ffield.reax")`,
        `get_potential_metadata("/path/to/potentials/Al99.eam.alloy")`,
        `get_potential_metadata("Mg_Zhou04.eam.alloy`,
        `get_potential_metadata("/data/Fe-C_Hepburn_Ackland.eam.fs")`,
        `get_potential_metadata("Cu_Zhou04.eam.alloy`
    ]
    [/SYNTACTICAL]


    Args:
        file_path:
            [BRIEF] Absolute path to the potential file. [/BRIEF]
            [DETAILED] This is the absolute path to a LAMMPS-compatible potential file (e.g., ReaxFF or EAM formats). The file name is used to determine metadata, so it must match one of the known patterns. [/DETAILED]
            [SYNTACTICAL] Format: "string ending in a recognized potential filename". [/SYNTACTICAL]
            [EXAMPLES] Examples: "/path/to/file/ffield.reax", "ffield_UTA1.ITT" [/EXAMPLES]

    Returns:
        str :
            [BRIEF] Structured metadata string describing the potential file. [/BRIEF]
            [DETAILED] The returned string includes the type of interatomic potential and a list of chemical elements that it supports. This helps in choosing suitable potentials for simulations involving specific atoms. [/DETAILED]
            [EXAMPLES] Example outputs: "{potential type : reax, elements supported : Carbon (C), Hydrogen (H), Oxygen (O), Calcium (Ca), Silicon (Si), pair_style : reaxff}" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the file name is not recognized or is empty. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the filename does not match any known potential files.
            This helps prevent silent failures and makes debugging easier in automated workflows. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, ensure the file name matches one of the known potential files or update the tool to include new potential file names as needed. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Limitations:
        - This tool only recognizes a predefined set of potential file names. If the file name does not match any of the known patterns, it will raise a ValueError.
        - The metadata returned is static and does not include dynamic information from the file contents, such as specific parameters or coefficients used in the potential.
        - The tool does not validate the actual contents of the potential file; it relies solely on the file name for metadata extraction.
    [/LIMITATIONS]
    """
    if not file_path:
        raise ValueError("File path must not be None or empty.")

    potential_name = file_path.split("/")[-1]

    if potential_name == "Si.sw":
        return "{potential type : Stillinger Weber (SW), elements supported : Si (Silicon), pair_style : sw}"
    elif potential_name == "2007_SiO.tersoff":
        return "{potential type : tersoff, elements supported : Si (Silicon), Oxygen (O), pair_style : tersoff}"
    elif potential_name == "Al99.eam.alloy":
        return "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}"
    elif potential_name == "Cu_Zhou04.eam.alloy":
        return "{potential type : EAM, elements supported : Cu (Copper), pair_style : eam/alloy}"
    elif potential_name == "Mg_Zhou04.eam.alloy":
        return "{potential type : EAM, elements supported : Mg (Magnesium), pair_style : eam/alloy}"
    elif potential_name == "Fe-C_Hepburn_Ackland.eam.fs":
        return (
            "{potential type : EAM, elements supported : "
            "Fe (Iron), C (Carbon), pair_style : eam/fs}"
        )
    else:
        raise ValueError(f"Unrecognized potential file: {potential_name}")


@tool
def get_structure_from_mp_text(mp_id: str, file_path: str) -> str:
    """
    [BRIEF] Retrieves and saves the conventional crystal structure of a material from the Materials Project as a CIF file. [/BRIEF]

    [DETAILED] This tool retrieves the conventional unit cell structure of a material from the Materials Project using its material ID. It transforms the primitive structure returned by the database into its conventional crystallographic form using symmetry operations, and exports the result in CIF format to a specified file path. This tool is valuable for workflows that require standardized crystal structure representations — such as simulations, visualization, structure matching, or publication. It avoids the need for manual structure transformation or dealing with primitive cells when conventional representation is needed. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need the conventional crystallographic (not primitive) structure of a material from the Materials Project.
        - Best suited for preparing simulation-ready input files, visualizing crystal structures, or storing standardized CIFs.
        - Recommended for quick and automated generation of conventional structure files for structure-based computation or crystallographic analysis.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure you have the Materials Project ID of the material you want to retrieve. Sometimes the MP ID is in the task description so be sure to fully capture the information there. [/PREREQUISITE]
        2. [CURRENT] Use this tool to fetch the conventional structure in CIF format by providing the MP ID and desired file path. [/CURRENT]
        3. [FOLLOW_UP] The resulting CIF file can be used in subsequent steps such as molecular dynamics simulations (using the tool `run_lammps`), structure visualization, or crystallographic analysis. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Queries the Materials Project database using the provided material ID to retrieve the material's primitive crystal structure.
        - Applies symmetry analysis to convert the primitive structure into its conventional crystallographic form.
        - Converts the conventional structure into CIF (Crystallographic Information File) format.
        - Saves the CIF content to the specified file path using an external function.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_structure_from_mp_text("mp-149", "/workspace/Si_conventional.cif")`,
        `get_structure_from_mp_text("mp-13", "outputs/Aluminum_structure.cif")`,
        `get_structure_from_mp_text("mp-1692", "/data/structures/CuO_conventional.cif")`,
        `get_structure_from_mp_text("mp-19770", "structure_files/Fe2O3.cif")`,
        `get_structure_from_mp_text("mp-1143", "/tmp/Al2O3_structure.cif")`,
    ]
    [/SYNTACTICAL]

    Args:
        mp_id (str):
            [BRIEF] Materials Project ID of the material. [/BRIEF]
            [DETAILED] A unique identifier used by the Materials Project database to reference a material. The ID typically starts with "mp-" followed by digits. It must correspond to an existing entry. [/DETAILED]
            [SYNTACTIC] Format: '"mp-XXXX" where X is a digit'. [/SYNTACTIC]
            [EXAMPLES] Examples: "mp-149", "mp-13", "mp-1234567" [/EXAMPLES]
        file_path (str):
            [BRIEF] Destination path for saving the CIF file. [/BRIEF]
            [DETAILED] Absolute path to the file where the CIF content will be written. [/DETAILED]
            [SYNTACTIC] Format: 'string path ending in ".cif" corresponding to the path of the CIF file'. [/SYNTACTIC]
            [EXAMPLES] Examples: "/tmp/output.cif", "structure_files/Al.cif" [/EXAMPLES]

    Returns:
        str :
            [BRIEF] Status message indicating successful structure retrieval and saving at required path. [/BRIEF]
            [DETAILED] If successful, the tool retrieves the conventional crystallographic structure for the given Materials Project ID, converts it into CIF format, saves it at the specified path, and returns a confirmation message.
            If any step fails, a descriptive error message is returned instead. [/DETAILED]
            [EXAMPLES]
                - "Structure saved successfully at /workspace/data/structure.cif"
                - "Failed to retrieve or save structure: Invalid Materials Project ID" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised if structure retrieval or file saving fails. [/ERROR_WHEN]
            [ERROR_DETAILS] This generic exception is returned if any error occurs during Materials Project API access, structure conversion, or remote file write. The error message is descriptive and includes the failure reason. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, check the MP ID for correctness, and verify that the specified file path is writable. If the MP ID is invalid or does not exist, you may need to use a different ID or check the Materials Project database for available materials. If the file path is incorrect or inaccessible, ensure that the directory exists and has the correct permissions for writing files. If the API key is invalid or the Materials Project service is down, you may need to use another tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The tool requires a Materials Project API key to access the database, which is hardcoded in the function.
        - It assumes that the MP ID provided corresponds to a valid material in the Materials Project database.
        - The tool does not handle cases where the material has multiple structures or polymorphs; it retrieves only the first available structure.
    [/LIMITATIONS]
    """

    try:
        from mp_api.client import MPRester
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        with MPRester(os.getenv("MP_API_KEY")) as mpr:
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
        return f"Failed to retrieve or save structure: {e!s}"


@tool
def convert_structure_to_lammps_data(
    structure_path: str, output_file: str, atom_style: str = "charge"
) -> str:
    """
    [BRIEF] Converts a CIF-format structure file into a LAMMPS data file using a specified atom style. [/BRIEF]

    [DETAILED] This tool converts a crystal structure provided in CIF format (as a file) into a LAMMPS-compatible data file. Internally, it reads the CIF structure file using pymatgen, transforms it into a Structure object, and then serializes it to a LAMMPS data format using the LammpsData class. It supports configurable atom styles such as "atomic" or "charge", allowing flexibility based on simulation requirements.
    This enables seamless transformation of standardized crystallographic data into simulation-ready LAMMPS input files, streamlining the setup process for molecular dynamics workflows. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to convert crystallographic structure data in CIF format into a LAMMPS-compatible `.data` file.
        - Best suited for setting up molecular dynamics simulations where LAMMPS is the engine, and structure data is sourced from databases like Materials Project or experimental CIFs.
        - Avoid when your structure data is already in LAMMPS format or requires extensive pre-processing (e.g., force field assignments).
        - Recommended for automating simulation pipelines that begin with standardized structural data and end in LAMMPS-ready input formats.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure you have a valid CIF file containing the crystallographic structure of the material you want to simulate. Use the tool `get_structure_from_mp_text` to obtain one. [/PREREQUISITE]
        2. [CURRENT] Use this tool to convert the CIF file into a LAMMPS data file by providing the path to the CIF file, the desired output file path, and the atom style (if different from the default "charge"). [/CURRENT]
        3. [FOLLOW_UP] The resulting LAMMPS data file can be used directly in LAMMPS simulations, using the tool `run_lammps` to run the simulation. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - The tool reads the `.cif` file using `pymatgen.core.Structure.from_file` to create a Structure object.
        - The `LammpsData.from_structure` function is called, using the chosen `atom_style` to format the atomic data accordingly.
        - The resulting LAMMPS data file is written to the path specified by `output_file`.
        - The tool does not apply force fields or assign charges explicitly; it assumes the structure is already complete and appropriate for the selected `atom_style`.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `convert_structure_to_lammps_data("/workspace/graphene.cif", "/workspace/output/graphene.data")`,
        `convert_structure_to_lammps_data("./structures/NaCl.cif", "./data/NaCl.data", atom_style="atomic")`,
        `convert_structure_to_lammps_data("/data/SiO2.cif", "/converted_data/SiO2.data", atom_style="charge")`,
        `convert_structure_to_lammps_data("MgO.cif", "MgO.data", atom_style="charge")`,
        `convert_structure_to_lammps_data("/tmp/Al2O3.cif", "/tmp/Al2O3.data", atom_style="atomic")`
    ]
    [/SYNTACTICAL]

    Args:
        structure_path (str):
            [BRIEF] Path to the CIF-format structure file. [/BRIEF]
            [DETAILED] Path to the file containing the crystallographic structure.
            This file is read and converted into a pymatgen `Structure` object internally before being serialized to LAMMPS data format. [/DETAILED]
            [SYNTACTIC] Format: 'string ending in ".cif" correspoding to the path of the CIF file.' [/SYNTACTIC]
            [EXAMPLES] Examples: "/workspace/graphene.cif", "./data/SiO2.cif" [/EXAMPLES]
        output_file (str):
            [BRIEF] Path where the LAMMPS data file will be saved. [/BRIEF]
            [DETAILED] This is the destination file path where the generated LAMMPS-compatible data file will be written.
            The output file will contain the atomic positions, types, and other necessary information formatted for LAMMPS simulations. [/DETAILED]
            [SYNTACTIC] Format: 'valid string representing a writable `.data` file path'. [/SYNTACTIC]
            [EXAMPLES] Examples:
                - "/workspace/output/graphene.data",
                -"./converted_data/SiO2.data" [/EXAMPLES]
        atom_style (str):
            [BRIEF] Atom style to be used in the LAMMPS data file, defaults to "charge". [/BRIEF]
            [DETAILED] Specifies the LAMMPS atom style to use when formatting the data file.
            Common values include:
                - "atomic": Includes atomic positions and mass, no charges.
                - "charge": Includes atomic charges in addition to position and mass.
            The choice of style should match the `atom_style` directive in the LAMMPS input script.
            Defaults to "charge" if nothing provided.
            Valid values: "real", "metal", "si", "cgs", "electron", "micro", "nano", "full", "". [/DETAILED]
            [SYNTACTIC] Format: "one of the predefined LAMMPS atom styles as a lowercase string". [/SYNTACTIC]
            [EXAMPLES] Examples: "real", "metal". [/EXAMPLES]

    Returns:
        str:
            [BRIEF] Status message indicating successful LAMMPS data file generation. [/BRIEF]
            [DETAILED] If the conversion is successful, returns a confirmation message specifying the path where the LAMMPS data file has been saved. This message can be used for logging or downstream validation in automated simulation workflows. [/DETAILED]
            [EXAMPLES] Example outputs:
                - "LAMMPS data file successfully written to: /workspace/output/graphene.data"
                - "LAMMPS data file successfully written to: ./converted_data/SiO2.data" [/EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised if structure conversion or file writing fails. [/ERROR_WHEN]
            [ERROR_DETAILS] This may occur due to issues such as:
                - Invalid or malformed CIF file that cannot be parsed.
                - File I/O errors when writing the output file (e.g., permission issues, invalid paths).
                - Internal errors in the conversion process (e.g., unsupported atom styles, missing dependencies).
            The error message will provide context for debugging the issue. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, ensure the CIF file is well-formed and accessible, check that the output file path is valid and writable, and verify that the specified atom style is supported by LAMMPS. If the CIF file is malformed, you may need to correct it or use a different file.
            If the atom style is unsupported, choose a valid atom style from the LAMMPS documentation. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The tool assumes the CIF file is well-formed and contains a valid crystallographic structure.
        - It does not perform any validation on the structure content beyond what pymatgen provides.
        - If the CIF file contains unsupported features or is malformed, it may raise an error during parsing.
        - The atom style must be one of the recognized LAMMPS styles; otherwise, it will raise an error.
    [/LIMITATIONS]
    """
    try:
        convert_structure_to_lammps_data_sim = modal.Function.lookup(
            "simagent", "convert_structure_to_lammps_data"
        )
        convert_structure_to_lammps_data_sim.remote(
            structure_path, output_file, atom_style
        )
        return f"LAMMPS data file successfully written to: {output_file}"
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while converting structure to LAMMPS data: {e!s}"
        ) from e


@tool
def run_lammps(input_file: str) -> str:
    """
    [BRIEF] Runs a LAMMPS simulation based on the provided input script and generates a corresponding log file. [/BRIEF]

    [DETAILED] This tool executes a LAMMPS molecular dynamics simulation using a specified input script. It takes the path to a LAMMPS input file and automatically triggers the simulation run through a remote execution backend. The tool also generates a corresponding log file (named after the input script, with `.log` extension replacing the original extension) which contains detailed simulation output including thermodynamic data, errors (if any), and runtime diagnostics. [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to execute a LAMMPS molecular dynamics simulation using a predefined input script.
        - Use to run a molecular dynamics simulation without needing to manually start LAMMPS or handle the command line interface.
        - Recommended for production simulations, high-throughput screening, and automated pipelines where simulation setup is complete and ready to run.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure you have a valid LAMMPS input script ready for execution.
        This script should contain all necessary simulation parameters, atom definitions, force fields, and commands. Additionally, ensure that the .data file is in place. You can create one using the function `convert_structure_to_lammps_data`.  [/PREREQUISITE]
        2. [CURRENT] Use this tool to run the LAMMPS simulation by providing the path to the input script. The tool will handle the remote execution and log file generation. [/CURRENT]
        3. [FOLLOW_UP] After the simulation completes, check the generated log file for results, diagnostics, and any errors. The log file will be named based on the input script, with a `.log` extension. You can then proceed to analyze the results or use the output data in subsequent steps of your workflow. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - The input LAMMPS script (typically ending in `.in`) is passed to the remote backend.
        - The tool constructs a log file name by replacing the script's extension with `.log`.
        - A remote function is invoked with the input file, the log file path.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `run_lammps("/workspace/lammps_inputs/graphene_sim.in")`,
        `run_lammps("./data/sio2_minimize.in")`,
        `run_lammps("minimize_bulk_sio2.in")`,
        `run_lammps("/path/to/simulation/input_script.in")`,
        `run_lammps("job123.lmp", 2)`,
    ]
    [/SYNTACTICAL]

    Args:
        input_file (str):
            [BRIEF] Path to the LAMMPS input script file. [/BRIEF]
            [DETAILED] This parameter specifies the absolute or relative path to the input script used by LAMMPS. The script typically contains simulation settings such as atom style, force field parameters, boundary conditions, and compute directives. The file must be in LAMMPS-compatible format (`.in` extension is conventional but not required) and should not require interactive input during execution. [/DETAILED]
            [SYNTACTIC] Format: string representing a file path; must be readable by the backend LAMMPS engine. [/SYNTACTIC]
            [EXAMPLES] Examples:
                - "/workspace/lammps_inputs/graphene_sim.in"
                - "./simulations/liquid_water.in"
                - "minimize_bulk_sio2.in" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] Message indicating simulation completion with log file location. [/BRIEF]
            [DETAILED] On success, returns a message confirming the simulation run, the path to the latest input script used, and the corresponding log file. The log file contains detailed runtime diagnostics and output for verification. [/DETAILED]
            [EXAMPLES]
                - "Simulation ran successfully using input: simulations/run_graphene.in, log saved at: run_graphene.log"
                - "Simulation ran successfully using input: ./jobs/job123.lmp, log saved at: job123.log" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] Raised if the LAMMPS simulation fails due to invalid input or LAMMPS-specific error. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised when the underlying LAMMPS execution raises a ValueError, which can occur due to issues such as missing sections in the input file, invalid parameters, or other LAMMPS-specific errors that prevent the simulation from running successfully.
            The error message will provide context about the failure, such as missing commands or unsupported features in the input script. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, check the input file for correctness, ensuring that all required sections are present and properly formatted.
            Verify that the parameters used in the input script are valid for the LAMMPS version being used. If the error persists, consult the LAMMPS documentation or community forums for guidance on the specific error encountered. [/ERROR_RECOVERY]

        Exception:
            [ERROR_WHEN] Raised on unexpected backend or runtime errors. [/ERROR_WHEN]
            [ERROR_DETAILS] This generic exception is raised for any unexpected issues that occur during the execution of the LAMMPS simulation, such as backend unavailability, file system errors, or misconfigured modal runtime.
            The error message will include details about the failure, which can help in debugging the issue. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, check the backend configuration to ensure it is correctly set up and available.
            Verify that the input file path is correct and accessible.
            If the backend is misconfigured or unavailable, you may need to adjust the modal settings or ensure that the modal service is running properly.
            If the error persists, consult the modal documentation or support resources for further assistance. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - The tool assumes that the input file is correctly formatted and does not require interactive input during execution.
        - It does not validate the contents of the input file beyond basic file existence checks; any errors in the LAMMPS script will result in a runtime error during execution.
        - The tool is designed to work with a specific backend (modal) and may not function correctly if the backend is misconfigured or unavailable.
    [/LIMITATIONS]
    """
    if not input_file:
        raise ValueError("Input file path must not be None or empty.")

    try:
        file_name_without_extension = Path(input_file).stem
        log_file = f"{file_name_without_extension}.log"
        run_lammps_sim = modal.Function.lookup("simagent", "run_lammps")
        run_lammps_sim.remote(input_file, log_file)
        return f"Simulation ran successfully using input: {input_file}, log saved at: {log_file}"

    except ValueError as e:
        # Raise a ValueError with more context about the failure
        raise ValueError(
            f"The LAMMPS simulation failed with a ValueError: {e!s}"
        ) from e
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while running the LAMMPS simulation: {e!s}"
        ) from e
