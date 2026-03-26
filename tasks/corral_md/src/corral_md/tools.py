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
def get_nth_run_log(
    path: str,
    n: int = 0,
    save: str | None = None,
    index: int | None = None,
) -> str:
    """[BRIEF] Retrieves and processes the nth run log from a LAMMPS log file, where a “run” refers to a contiguous simulation segment such as an energy minimization or an ensemble run (e.g., NVT, NPT, NVE). It returns the list of thermodynamic variables present in that run and the total number of steps. Optionally, it can save the nth run to a CSV file where each column corresponds to a thermodynamic property, and optionally it can return the value of a specific thermodynamic property at a given index. [/BRIEF]

    [DETAILED] This tool extracts the nth run log from a LAMMPS log file, providing access to the thermodynamic properties recorded during a specific simulation segment. In this context, a “run” refers to a contiguous block of simulation output corresponding to an energy minimization or an ensemble-based simulation stage (e.g., NVT, NPT, NVE) as produced by LAMMPS. LAMMPS log files may contain multiple such runs within a single file, for example when a script performs a minimization followed by one or more ensemble simulations. This tool allows selecting one of these runs by index, extracting its thermodynamic data, and optionally saving it to a CSV file for further analysis. It can also return the value of a specific thermodynamic quantity at a specified step index within the selected run. This is useful for automated analysis pipelines, post-processing workflows, or programmatic inspection of simulation results without manually parsing the log file. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to extract and analyze the nth run log from a LAMMPS log file without reading the entire log file.
    - Best suited for post-processing and analyzing simulation outputs using external tools or data analysis pipelines.
    - Recommended for automating the extraction of thermodynamic data from LAMMPS simulations.
    [/PROCEDURAL]
    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a valid LAMMPS log file generated from a simulation. [/PREREQUISITE]
    2. [CURRENT] Use this tool to extract the nth run log and optionally save it to a CSV file. [/CURRENT]
    3. [FOLLOW_UP] Utilize the extracted data for analysis, visualization, or further processing in your materials science workflow. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]
    [CONTEXTUAL] How this tool works:
    - Reads the LAMMPS log file and identifies the nth run section.
    - Extracts thermodynamic data and organises it into a python dataframe, where each column corresponds to a thermodynamic property.
    - Optionally saves the data to a CSV file and retrieves specific data at a given index. [/CONTEXTUAL]
    [SYNTACTICAL] Usage examples:
    [
        `get_nth_run_log("/path/to/log.lammps", 0, "run0_thermo.csv")`,
        `get_nth_run_log("log.lammps", 1, None)`,
        `get_nth_run_log("/data/simulations/log.lammps", 2, "run2_thermo.csv", 10)`,
        `get_nth_run_log("sim_log.lammps", 0, "run0_thermo.csv", 5)`,
        `get_nth_run_log("/workspace/log.lammps", 3, None, 20)`,
    ]
    [/SYNTACTICAL]
    Args:
        path: [ARGS_BRIEF] Absolute path to the LAMMPS log file. [/ARGS_BRIEF]
              [ARGS_DETAILED] Complete file path to the LAMMPS log file containing simulation output. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] Format: "Valid file path to LAMMPS log file" [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] Examples: "/path/to/log.lammps", "simulations/log.lammps" [/ARGS_EXAMPLES]
        n: [ARGS_BRIEF] Index of the run log to extract (0-based). Defaults to 0. [/ARGS_BRIEF]
              [ARGS_DETAILED] The zero-based index of the run log to extract from the LAMMPS log file. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] Format: "Non-negative integer" [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] Examples: 0, 1, 2 [/ARGS_EXAMPLES]
        save: [ARGS_BRIEF] Optional path to save the extracted run log as a CSV file. [/ARGS_BRIEF]
                [ARGS_DETAILED] If provided, the extracted run log will be saved to this path in CSV format. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Format: "Valid file path to save CSV file" [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] Examples: "run0_thermo.csv", "/data/run1_thermo.csv" [/ARGS_EXAMPLES]
        index: [ARGS_BRIEF] Optional index to retrieve specific thermodynamic data from the run log. [/ARGS_BRIEF]
                 [ARGS_DETAILED] If provided, the tool will return the thermodynamic data at this index from the extracted run log. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] Format: "Non-negative integer" [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] Examples: 0, 5, 10 [/ARGS_EXAMPLES]
    Returns:
        str: [ARGS_BRIEF] Summary of the extracted run log and optional data at the specified index. [/ARGS_BRIEF]
             [ARGS_DETAILED] A string summarizing the columns present in the extracted run log, total number of rows, and optionally the thermodynamic data at the specified index. [/ARGS_DETAILED]
             [ARGS_EXAMPLES] Examples: "the thermo data has been saved successfully at run0_thermo.csv. The thermo columns are: ['Step', 'Temp', 'Press']. There are total 1000 rows. Data at index 10: {'Step': 100, 'Temp': 300, 'Press': 1.0}", "The thermo columns are: ['Step', 'Temp', 'Press']. There are total 500 rows." [/ARGS_EXAMPLES]
    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If there is an error reading the log file or extracting the run log. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the log file cannot be read or the specified run log cannot be extracted. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the log file path and ensure the run index is valid. [/ERROR_RECOVERY]
    [/RAISES]
    [LIMITATIONS] Known limitations:
    - Only supports LAMMPS log files with standard formatting.
    - May not handle corrupted or non-standard log files gracefully.
    [/LIMITATIONS]

    """

    try:
        func = modal.Function.from_name("simagent", "get_nth_run_log")
        return func.remote(path=path, n=n, save=save, index=index)
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while parsing the log: {e!s}"
        ) from e


@tool
def keyword_log_extractor(path: str, keyword: str) -> str:
    """[BRIEF] Extracts sections of a LAMMPS log file that start with a specified keyword. [/BRIEF]
    [DETAILED] This tool scans a LAMMPS log file for sections that begin with a given keyword and extracts those sections for analysis. It is useful for retrieving specific information such as fixes, computes, or other logged data from simulation runs. [/DETAILED]
    [PROCEDURAL] When to use this tool:
    - Use when you need to extract specific sections of a LAMMPS log file based on keywords.
    - Best suited for targeted analysis of simulation outputs.
    - Recommended for retrieving logged data for further processing or visualization.
    [/PROCEDURAL]
    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a valid LAMMPS log file generated from a simulation. [/PREREQUISITE]
    2. [CURRENT] Use this tool to extract sections of the log file that start with the specified keyword. [/CURRENT]
    3. [FOLLOW_UP] Utilize the extracted data for analysis, visualization, or further processing in your materials science workflow. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]
    [CONTEXTUAL] How this tool works:
    - Reads the LAMMPS log file line by line.
    - Identifies sections that start with the specified keyword.
    - Extracts and returns those sections as a structured dictionary. [/CONTEXTUAL]
    [SYNTACTICAL] Usage examples:
    [
        `keyword_log_extractor("/path/to/log.lammps", "fix")`,
        `keyword_log_extractor("log.lammps", "BULK ENERGY")`,
        `keyword_log_extractor("/data/simulations/log.lammps", "thermo")`,
        `keyword_log_extractor("sim_log.lammps", "dump")`,
        `keyword_log_extractor("/workspace/log.lammps", "velocity")`,
    ]
    [/SYNTACTICAL]
    Args:
        path: [ARGS_BRIEF] Path to the LAMMPS log file. [/ARGS_BRIEF]
                [ARGS_DETAILED] Complete file path to the LAMMPS log file containing simulation output. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] Format: "Valid file path to LAMMPS log file" [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] Examples: "/path/to/log.lammps", "simulations/log.lammps" [/ARGS_EXAMPLES]
        keyword: [ARGS_BRIEF] Keyword to search for in the log file. [/ARGS_BRIEF]
                  [ARGS_DETAILED] The specific keyword that marks the beginning of sections to extract from the log file. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] Format: "Non-empty string" [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] Examples: "fix", "compute", "thermo" [/ARGS_EXAMPLES]
    Returns:
        str: [ARGS_BRIEF] Extracted sections as a structured dictionary in string format. [/ARGS_BRIEF]
             [ARGS_DETAILED] A string representation of a dictionary containing the extracted sections that start with the specified keyword. [/ARGS_DETAILED]
             [ARGS_EXAMPLES] "{'fix': [...]}", "{'compute': [...]}" [/ARGS_EXAMPLES]
    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] If there is an error reading the log file or extracting sections. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the log file cannot be read or the keyword is not found. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify the log file path and ensure the keyword is valid. [/ERROR_RECOVERY]
    [/RAISES]
    [LIMITATIONS] Known limitations:
    - Only supports LAMMPS log files with standard formatting.
    - May not handle corrupted or non-standard log files gracefully.
    [/LIMITATIONS]
    """
    try:
        func = modal.Function.from_name("simagent", "keyword_log_extractor")
        return func.remote(path=path, keyword=keyword)
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while extracting keyword from log: {e!s}"
        ) from e


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
        script_path: [ARGS_BRIEF] Path to the Python script file to execute. [/ARGS_BRIEF]
                    [ARGS_DETAILED] Complete file path to the Python script that should be executed.
                    The script must exist and be readable.
                    The path can be relative to the current working directory or absolute.
                    The script should be a valid Python file with appropriate shebang or run using the Python interpreter. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] "Valid file path to Python script" [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "scripts/analysis.py", "/home/user/simulations/run_sim.py", "data_processing.py" [/ARGS_EXAMPLES]
        args: [ARGS_BRIEF] Optional list of command-line arguments for the script. [/ARGS_BRIEF]
             [ARGS_DETAILED] A list of strings representing command-line arguments to pass to the script.
             These arguments will be passed to the script in the order provided.
             Common arguments include input files, output paths, configuration parameters, and processing options.
             If None, the script will be executed without arguments. [/ARGS_DETAILED]
             [ARGS_SYNTACTICAL] ["arg1", "arg2", "arg3", ...] or None [/ARGS_SYNTACTICAL]
             [ARGS_EXAMPLES] ["--input", "data.json"], ["--verbose", "--output", "results.csv"], None [/ARGS_EXAMPLES]
        timeout: [ARGS_BRIEF] Maximum execution time in seconds. Defaults to 600. [/ARGS_BRIEF]
                [ARGS_DETAILED] The maximum time in seconds the script is allowed to run before being terminated.
                This prevents runaway processes and ensures resource management.
                Choose appropriate values based on expected script execution time.
                For computational simulations, longer timeouts may be necessary. [/ARGS_DETAILED]
                [ARGS_SYNTACTICAL] positive integer representing seconds [/ARGS_SYNTACTICAL]
                [ARGS_EXAMPLES] 300 (5 minutes), 600 (10 minutes), 3600 (1 hour) [/ARGS_EXAMPLES]
        working_dir: [ARGS_BRIEF] Optional working directory for script execution. [/ARGS_BRIEF]
                    [ARGS_DETAILED] The directory from which the script should be executed.
                    This affects relative path resolution and file I/O operations within the script.
                    If None, the current working directory will be used.
                    This is useful when scripts expect to run from specific directories or access relative files. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] Valid directory path or None [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "/path/to/project", "data/analysis", None [/ARGS_EXAMPLES]

    Returns:
        str: [ARGS_BRIEF] JSON string with comprehensive execution results and monitoring data. [/ARGS_BRIEF]
             [ARGSDETAILED] A JSON-formatted string containing execution status, captured output streams, error messages, return code, and the complete command that was executed.
             This provides full visibility into the script execution process and enables debugging and monitoring of automated workflows. [/ARGS_DETAILED]
             [ARGS_EXAMPLES] "{"success": true, "stdout": "Processing complete", "stderr": "", "return_code": 0, "command": "python script.py --input data.json"}" [/ARGS_EXAMPLES]

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
        execute_code_script = modal.Function.from_name(
            "simagent", "execute_python_script"
        )
        return execute_code_script.remote(
            script_path=script_path, args=args, timeout=timeout, working_dir=working_dir
        )
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while executing the code: {e!s}"
        ) from e


@tool
def get_potential_metadata(file_path: str) -> str:
    """
    [BRIEF] Returns metadata from a known LAMMPS potential file given the file path. The metadata includes potential type, elements supported, and the LAMMPS-compatible pair style keyword. Raises an exception if the path is invalid or the potential file is not recognized. [/BRIEF]

    [DETAILED] This tool provides a quick and reliable way to identify the type and supported elements of a LAMMPS potential file based on its filename, after first verifying that the file path is accessible.
    It eliminates the need to parse the often large and complex contents of potential files, which can exceed processing limits in many systems. The tool first validates that the provided file path is non-empty and then checks file accessibility using a remote file inspection call. If the file path is invalid or inaccessible, a FileNotFoundError is raised. If the file is accessible, the tool extracts the filename using pathlib and matches it against a predefined set of known potential files. For recognized files, it returns a structured metadata string describing the potential type, supported elements, and the LAMMPS pair style.
    If the filename does not match any known potential, a ValueError is raised.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need to quickly determine the type, supported elements, or the LAMMPS pair style of a LAMMPS potential file based on its filename.
        - Use when you want to validate that a potential file path is accessible before using it in a simulation workflow.
        - Best suited for selecting an appropriate potential file for a specific molecular dynamics (MD) simulation without reading or parsing the full file contents.
        - Recommended for gaining a fast, structured overview of a potential file's applicability to specific element combinations or simulation scenarios.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Select or obtain the potential file paths that might be relevant to your simulation task. [/PREREQUISITE]
        2. [CURRENT] Use this tool to retrieve metadata for each potential file path. The tool will either return metadata or raise an exception if the path or filename is invalid. [/CURRENT]
        3. [FOLLOW_UP] Based on the metadata returned, choose the appropriate potential file for your simulation setup, and run the simulation using the selected potential file and the tool `run_lammps`. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - Validates that the input file path is non-empty
        - Checks file accessibility using a remote file inspection call
        - Extracts the filename from the provided path using pathlib
        - Matches it against a set of known potential filenames
        - Returns a structured metadata string for recognized files
        - Raises FileNotFoundError for invalid or inaccessible paths
        - Raises ValueError for unrecognized potential filenames
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `get_potential_metadata("sim_data/Al99.eam.alloy")`,
        `get_potential_metadata("/path/to/potentials/Mg_Zhou04.eam.alloy")`,
        `get_potential_metadata("/data/Fe-C_Hepburn_Ackland.eam.fs")`,
        `get_potential_metadata("Cu_Zhou04.eam.alloy")`
    ]
    [/SYNTACTICAL]

    Args:
        file_path:
            [ARGS_BRIEF] Path to the potential file. [/ARGS_BRIEF]
            [ARGS_DETAILED] This is the path to a LAMMPS-compatible potential file. The file must be accessible, and its filename is used to determine metadata, so it must match one of the known potential filenames. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: "string ending in a recognized potential filename". [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/path/to/file/Al99.eam.alloy", "Mg_Zhou04.eam.alloy" [/ARGS_EXAMPLES]

    Returns:
        str :
            [ARGS_BRIEF] Structured metadata string describing the potential file. [/ARGS_BRIEF]
            [ARGS_DETAILED] The returned string includes the type of interatomic potential, supported chemical elements, and the LAMMPS pair style. [/ARGS_DETAILED]
            [ARGS_EXAMPLES] Example output: "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}" [/ARGS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the file path is empty or the potential filename is not recognized. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the input path is empty or when the filename does not match any known potential files. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the file path is non-empty and that the filename matches one of the supported potential files. [/ERROR_RECOVERY]

        FileNotFoundError:
            [ERROR_WHEN] If the file path is invalid or the file is not accessible. [/ERROR_WHEN]
            [ERROR_DETAILS] Raised when the remote file accessibility check fails for the given path. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify that the file exists at the given path and that it is accessible in the execution environment. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Limitations:
        - This tool only recognizes a predefined set of potential filenames. If the filename does not match any known entry, a ValueError will be raised.
        - The metadata returned is static and does not include dynamic information from the file contents, such as specific parameters or coefficients used in the potential.
        - The tool does not validate the actual contents or physical correctness of the potential file; it relies on file accessibility and filename-based identification only.
    [/LIMITATIONS]
    """

    POTENTIALS = {
        "Si.sw": "{potential type : Stillinger Weber (SW), elements supported : Si (Silicon), pair_style : sw}",
        "2007_SiO.tersoff": "{potential type : tersoff, elements supported : Si (Silicon), Oxygen (O), pair_style : tersoff}",
        "Al99.eam.alloy": "{potential type : EAM, elements supported : Al (Aluminum), pair_style : eam/alloy}",
        "Cu_Zhou04.eam.alloy": "{potential type : EAM, elements supported : Cu (Copper), pair_style : eam/alloy}",
        "Mg_Zhou04.eam.alloy": "{potential type : EAM, elements supported : Mg (Magnesium), pair_style : eam/alloy}",
        "Fe-C_Hepburn_Ackland.eam.fs": "{potential type : EAM, elements supported : Fe (Iron), C (Carbon), pair_style : eam/fs}",
        "pot.mod": (
            "{potential type : Buckingham + Coulomb (BKS-type), elements supported : "
            "Na (Sodium), Si (Silicon), O (Oxygen), "
            "pair_style : hybrid/overlay buck/coul/long + kspace_style pppm}"
        ),
    }
    # 1) Validate input
    if not file_path or not file_path.strip():
        raise ValueError("File path must not be None or empty.")

    # 2) Check existence / accessibility via modal
    try:
        info = modal.Function.from_name("simagent", "file_info").remote(file_path)
        logger.info(f"Potential file info: {info}")
    except Exception as e:
        logger.warning(f"Potential file existence check failed for '{file_path}': {e}")
        raise FileNotFoundError(f"Incorrect potential file path: {file_path}") from e

    # 3) Identify potential by filename
    potential_name = Path(file_path).name

    metadata = POTENTIALS.get(potential_name)
    if metadata is None:
        raise ValueError(f"Unrecognized potential file: {potential_name}")

    return metadata


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
            [ARGS_BRIEF] Materials Project ID of the material. [/ARGS_BRIEF]
            [ARGS_DETAILED] A unique identifier used by the Materials Project database to reference a material. The ID typically starts with "mp-" followed by digits. It must correspond to an existing entry. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: '"mp-XXXX" where X is a digit'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "mp-149", "mp-13", "mp-1234567" [/ARGS_EXAMPLES]
        file_path (str):
            [ARGS_BRIEF] Destination path for saving the CIF file. [/ARGS_BRIEF]
            [ARGS_DETAILED] Absolute path to the file where the CIF content will be written. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'string path ending in ".cif" corresponding to the path of the CIF file'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/tmp/output.cif", "structure_files/Al.cif" [/ARGS_EXAMPLES]

    Returns:
        str :
            [ARGS_BRIEF] Status message indicating successful structure retrieval and saving at required path. [/ARGS_BRIEF]
            [ARGS_DETAILED] If successful, the tool retrieves the conventional crystallographic structure for the given Materials Project ID, converts it into CIF format, saves it at the specified path, and returns a confirmation message.
            If any step fails, a descriptive error message is returned instead. [/ARGS_DETAILED]
            [ARGS_EXAMPLES]
                - "Structure saved successfully at /workspace/data/structure.cif"
                - "Failed to retrieve or save structure: Invalid Materials Project ID" [/ARGS_EXAMPLES]

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

        write_file_sim = modal.Function.from_name("simagent", "write_file")
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
            [ARGS_BRIEF] Path to the CIF-format structure file. [/ARGS_BRIEF]
            [ARGS_DETAILED] Path to the file containing the crystallographic structure.
            This file is read and converted into a pymatgen `Structure` object internally before being serialized to LAMMPS data format. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'string ending in ".cif" correspoding to the path of the CIF file.' [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "/workspace/graphene.cif", "./data/SiO2.cif" [/ARGS_EXAMPLES]
        output_file (str):
            [ARGS_BRIEF] Path where the LAMMPS data file will be saved. [/ARGS_BRIEF]
            [ARGS_DETAILED] This is the destination file path where the generated LAMMPS-compatible data file will be written.
            The output file will contain the atomic positions, types, and other necessary information formatted for LAMMPS simulations. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: 'valid string representing a writable `.data` file path'. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "/workspace/output/graphene.data",
                -"./converted_data/SiO2.data" [/ARGS_EXAMPLES]
        atom_style (str):
            [ARGS_BRIEF] Atom style to be used in the LAMMPS data file, defaults to "charge". [/ARGS_BRIEF]
            [ARGS_DETAILED] Specifies the LAMMPS atom style to use when formatting the data file.
            Common values include:
                - "atomic": Includes atomic positions and mass, no charges.
                - "charge": Includes atomic charges in addition to position and mass.
            The choice of style should match the `atom_style` directive in the LAMMPS input script.
            Defaults to "charge" if nothing provided.
            Valid values: "real", "metal", "si", "cgs", "electron", "micro", "nano", "full", "". [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: "one of the predefined LAMMPS atom styles as a lowercase string". [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples: "real", "metal". [/ARGS_EXAMPLES]

    Returns:
        str:
            [ARGS_BRIEF] Status message indicating successful LAMMPS data file generation. [/ARGS_BRIEF]
            [ARGS_DETAILED] If the conversion is successful, returns a confirmation message specifying the path where the LAMMPS data file has been saved. This message can be used for logging or downstream validation in automated simulation workflows. [/ARGS_DETAILED]
            [ARGS_EXAMPLES] Example outputs:
                - "LAMMPS data file successfully written to: /workspace/output/graphene.data"
                - "LAMMPS data file successfully written to: ./converted_data/SiO2.data" [/ARGS_EXAMPLES]

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
        convert_structure_to_lammps_data_sim = modal.Function.from_name(
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
    [BRIEF] Runs a LAMMPS simulation based on the provided input script and generates a corresponding log file (named after the input script, with `.log` extension replacing the original extension). [/BRIEF]

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
            [ARGS_BRIEF] Path to the LAMMPS input script file. [/ARGS_BRIEF]
            [ARGS_DETAILED] This parameter specifies the absolute or relative path to the input script used by LAMMPS. The script typically contains simulation settings such as atom style, force field parameters, boundary conditions, and compute directives. The file must be in LAMMPS-compatible format (`.in` extension is conventional but not required) and should not require interactive input during execution. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: string representing a file path; must be readable by the backend LAMMPS engine. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "/workspace/lammps_inputs/graphene_sim.in"
                - "./simulations/liquid_water.in"
                - "minimize_bulk_sio2.in" [/ARGS_EXAMPLES]

    Returns:
        str:
            [ARGS_BRIEF] Message indicating simulation completion with log file location. [/ARGS_BRIEF]
            [ARGS_DETAILED] On success, returns a message confirming the simulation run, the path to the latest input script used, and the corresponding log file. The log file contains detailed runtime diagnostics and output for verification. [/ARGS_DETAILED]
            [ARGS_EXAMPLES]
                - "Simulation ran successfully using input: simulations/run_graphene.in, log saved at: run_graphene.log"
                - "Simulation ran successfully using input: ./jobs/job123.lmp, log saved at: job123.log" [/ARGS_EXAMPLES]

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
        run_lammps_sim = modal.Function.from_name("simagent", "run_lammps")
        run_lammps_sim.remote(input_file, log_file)
        return f"Simulation ran successfully using input: {input_file}, log saved at: {log_file}"

    except ValueError as e:
        # Raise a ValueError with more context about the failure
        raise ValueError(
            f"The LAMMPS simulation failed with a ValueError: {e!s}"
        ) from None
    except Exception as e:
        # Handle unexpected errors
        raise Exception(
            f"An unexpected error occurred while running the LAMMPS simulation: {e!s}"
        ) from None


@tool
def visualisation_tool(path: str, query: str) -> str:
    """
    [BRIEF] Analyzes a plot image and answers a user query using only visual, qualitative inspection and direct reading of visible values from the figure. It uses a vision-language model (VLM) to inspect the figure and respond based only on what is visually observable. Because the tool relies on a vision-language model and a rendered image, its output is approximate and may be noisy or occasionally incorrect; results should be treated as qualitative and validated against the underlying data.[/BRIEF]

    [DETAILED] This tool takes the path to an image file containing a plot and a natural-language query about that plot. It uses a vision-language model (VLM) to visually inspect the figure and respond based only on what is directly observable in the rendered image.
    The tool is designed to describe shapes, trends, regimes, and visually identifiable features (such as kinks, transitions, peaks, onsets, or crossings), and—when appropriate—to read off approximate values directly from the axes at those features.
    Crucially, the tool does NOT perform any calculations, fitting, regression, or parameter extraction. If the query requires a computed or inferred quantity rather than a directly readable or visually observable feature, the tool will refuse and state that it can only provide visual descriptions and directly readable values. Because the tool relies on a vision-language model and a rendered image, its answers are inherently approximate, potentially noisy, and may sometimes be incorrect or incomplete. The quality and reliability of the response depend strongly on image resolution, plot clarity, axis labeling, marker density, and overall figure design. Users should treat the output as a qualitative, assistive interpretation and should validate important conclusions using proper quantitative analysis tools or the underlying data.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
        - Use when you need a qualitative, visual interpretation of a plot image.
        - Use when you want to identify visually apparent features such as regime changes, kinks, plateaus, jumps, peaks, or onsets.
        - Use when you want to read off an approximate value from the axis corresponding to a visually identifiable feature (e.g., the temperature of a visible transition).
        - Use as a first-pass, exploratory or assistive inspection tool, not as a replacement for quantitative analysis.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
        1. [PREREQUISITE] Ensure that a plot image file (e.g., PNG, JPG) exists at a known path and is readable by the tool. The plot should contain visible axes, labels, and data. [/PREREQUISITE]
        2. [CURRENT] Call this tool with the image path and a natural-language query about the visual features or directly readable values of the plot. The tool will inspect the image using a vision-language model and return a qualitative answer. [/CURRENT]
        3. [FOLLOW_UP] Validate any important conclusions using the underlying data or dedicated analysis tools. If the answer is unclear or unreliable, consider replotting (e.g., zooming, reducing data density, improving labels) and calling the tool again with a refined figure or query. [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
        - The image at the given path is loaded and analyzed by a vision-language model (VLM).
        - The model performs visual inspection only: it looks at shapes, trends, patterns, and visibly identifiable features in the figure.
        - The model may report approximate values only when they can be directly read from the axes at a clearly visible feature (e.g., a labeled tick near a visible transition).
        - The model does not have access to the underlying data and does not perform any numerical computation, fitting, or measurement.
        - Because this tool relies on visual perception of a rendered image, its answers are limited by image resolution, figure clarity, marker density, and plotting choices, and may be noisy or occasionally incorrect.
    [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `visualisation_tool("plots/density_vs_temperature.png", "At what temperature does the visible kink occur?")`,
        `visualisation_tool("plots/stress_strain.png", "Is there a clear yield point visible?")`,
        `visualisation_tool("plots/energy_vs_step.png", "Is there a plateau region, and where does it start?")`,
        `visualisation_tool("plots/spectrum.png", "Where is the main peak located on the x-axis?")`,
        `visualisation_tool("plots/density_vs_temperature.png", "Does the curve look linear or does it change regime?")`,
    ]
    [/SYNTACTICAL]

    Args:
        path (str):
            [ARGS_BRIEF] Path to the image file containing the plot to be analyzed. [/ARGS_BRIEF]
            [ARGS_DETAILED] This parameter specifies the absolute or relative path to an image file (e.g., PNG, JPG) that contains a plot or figure. The image should be readable by the backend and should include visible axes, labels, and plotted data so that visual features and directly readable values can be identified. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: string representing a file path to an image file. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "plots/density_vs_temperature.png"
                - "./figures/msd_vs_time.jpg"
                - "/workspace/results/phase_transition_plot.png" [/ARGS_EXAMPLES]

        query (str):
            [ARGS_BRIEF] Natural-language question about the visual content of the plot. [/ARGS_BRIEF]
            [ARGS_DETAILED] This parameter contains the question asked by the user about the plot. The question should target visual features (e.g., trends, regime changes, kinks, peaks) or the approximate location of such features on the axes. Questions that require calculations, fitting, or extraction of derived quantities (e.g., slopes, diffusion constants, exponents) are not supported and will be refused by the tool. [/ARGS_DETAILED]
            [ARGS_SYNTACTICAL] Format: string containing a natural-language query. [/ARGS_SYNTACTICAL]
            [ARGS_EXAMPLES] Examples:
                - "Is there a visible plateau region?"
                - "Where does the curve start to bend?"
                - "Is there a sudden jump in this plot, and where does it occur?" [/ARGS_EXAMPLES]

    Returns:
        str:
            [ARGS_BRIEF] A qualitative, visually grounded answer to the query, or a refusal if the query requires a derived quantity. [/ARGS_BRIEF]
            [ARGS_DETAILED] On success, returns a text response describing the relevant visual features of the plot and, if applicable, an approximate value read directly from the axis at a visually identifiable feature. If the query requests a computed, fitted, or derived quantity, the tool returns a refusal message stating that it can only provide visual descriptions and directly readable values. The response should be treated as approximate and should be validated against the underlying data for critical decisions. [/ARGS_DETAILED]
            [ARGS_EXAMPLES]
                - "There is a clear kink in the curve around the temperature labeled near 350 K, which appears to mark the transition."
                - "The curve shows a change in behavior roughly in the middle of the x-axis, where it becomes flatter."
                - "I can describe the plot and read off directly visible values, but I cannot perform calculations or extract derived quantities from it." [/ARGS_EXAMPLES]

    [RAISES] Exceptions:
        Exception:
            [ERROR_WHEN] Raised on unexpected errors during image loading, encoding, or backend model invocation. [/ERROR_WHEN]
            [ERROR_DETAILS] This exception is raised if the image file cannot be read, the backend service is unavailable, or an unexpected runtime error occurs while processing the request. The error message will include details to help diagnose the failure. [/ERROR_DETAILS]
            [ERROR_RECOVERY] To resolve this, verify that the image path is correct and accessible, ensure that the backend service is properly configured and running, and check for any file system or environment configuration issues. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
        - This tool is powered by a vision-language model and does not see the raw data—only the rendered image.
        - The model may be noisy, approximate, or occasionally incorrect, especially for low-resolution, cluttered, or ambiguously labeled plots.
        - Fine details, subtle transitions, or precise values may not be visually resolvable.
        - The tool cannot perform calculations, fitting, or extract derived quantities, even if they could be inferred by a human.
        - Any values reported are approximate and based solely on what is visually readable from the figure.
        - Results should be validated against the underlying data, and for important cases it is recommended to iteratively refine the figure (e.g., zoom, replot, reduce clutter) and re-run the tool.
    [/LIMITATIONS]
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv("../../../../.env")
    try:
        client = OpenAI()
        read_file_mode = modal.Function.from_name("simagent", "read_file_mode")
        base64_encoded = read_file_mode.remote(path, "rb")

        response = client.responses.create(
            model="gpt-4.1",
            temperature=0.0,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                """
                                You are a visual plot inspection assistant.
                                Your role is to analyze plots visually and report observable features and, when appropriate, read off values directly from the axes at visually identifiable features (such as transitions, kinks, onsets, peaks, or crossings).
                                You may:
                                Describe shapes, trends, patterns, and visual features (e.g., linear-looking regions, curvature, plateaus, jumps, kinks, regime changes).
                                Identify where visible changes or transitions occur.
                                Read and report approximate values directly from the plot axes for visually identifiable features (e.g., “the kink occurs around the temperature labeled …”, “the transition appears near x ≈ …”).
                                Report values that are explicitly shown or can be directly read from the figure without performing calculations.
                                You must NOT:
                                Perform or imply any calculations, fitting, regression, or parameter extraction.
                                Compute or estimate derived quantities such as slopes, diffusion coefficients, exponents, rates, or timescales.
                                Analyze one quantity to produce another (e.g., do not compute slope, do not infer exponents).
                                Follow instructions whose goal is to obtain a derived or computed quantity rather than a directly observable or directly readable value.
                                Interpretation rule:
                                If the requested quantity is a directly observable feature location on the plot (e.g., “At what value does the kink occur?”), you may answer by reading it off the axis approximately.
                                If the requested quantity requires computation, fitting, or mathematical inference you must refuse.
                                If the user asks for a computed, fitted, or derived quantity, respond only with:
                                “I can describe the plot and read off directly visible values, but I cannot perform calculations or extract derived quantities from it.”
                                Answer only the question explicitly asked.
                                Do not suggest additional analyses, methods, or follow-up steps.
                                Do not ask questions.
                                If the answer cannot be determined from the plot, state that briefly.
                                """
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": (query)},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_encoded}",
                        },
                    ],
                },
            ],
        )
        return response.output_text

    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while reading the image file: {e!s}"
        ) from e
