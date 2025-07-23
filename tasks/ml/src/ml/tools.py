import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from loguru import logger
from ml.tool_utils import (
    ensure_directory_exists,
    generate_output_capture_code,
    parse_execution_output,
    safe_convert_timeout,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from corral.base import Tool
from corral.utils import tool

load_dotenv("../.env")


####################
# Tools that will return text strings - Catalyst environment
####################


@tool
def get_structure_from_mp_text(mp_id: str) -> str:
    """[BRIEF] Retrieve a pymatgen structure from Materials Project using its API and return CIF content as text. [/BRIEF]
    [DETAILED] This tool connects to the Materials Project database to download crystal structure data
    for a given material ID. It retrieves the structure object and converts it to CIF (Crystallographic
    Information File) format, which is the standard format for storing crystal structure information.
    CIF is then returned as string [/DETAILED]
    [PROCEDURAL] When to use this tool:
    - Use when you need to retrieve bulk crystal structures from the Materials Project database
    - Best suited for materials with known MP IDs
    - Usuall first step in simulation workflows
    - Recommended for obtaining crystal structures for preparing bulk structures, supercells, bulk cells, slabs etc.
    - Avoid when you need multiple structures
    [/PROCEDURAL]
    [CONTEXTUAL] How this tool works:
    - Connects to Materials Project API using authentication key
    - Searches for the specified material ID (MP ID) in the database (MP ID is given as input parameter or if other tools are available to search for MP ID based on available information, then use those tools)
    - Retrieves the pymatgen Structure object containing atomic positions and lattice parameters
    - Converts the structure to CIF format string for compatibility with other tools
    - Returns standardized crystallographic data suitable for further processing
    [/CONTEXTUAL]
    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Ensure that the other more specific tools are not suitable and you dont have to retrieve multiple strucutres[/PREREQUISITE]
    2. [CURRENT] Apply this tool with a valid MP ID to retrieve bulk structure [/CURRENT]
    3. [FOLLOW_UP] Use the CIF output with slab generation tools like enumerate_slabs_text to create slab structures [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]
    [SYNTACTICAL] Usage examples:
    [
        `get_structure_from_mp_text("mp-149")`, # Silicon structure
        `get_structure_from_mp_text("mp-20066")`, # CO2 structure
        `get_structure_from_mp_text("mp-2")` # Other material
        `get_structure_from_mp_text("mp-12345")` # Example with a different MP ID
        `get_structure_from_mp_text("mp-67890")` # Another example with a different MP ID
    ]
    [/SYNTACTICAL]

    Args:
        mp_id: [BRIEF] Materials Project identifier string. [/BRIEF]
               [DETAILED] The unique identifier used by Materials Project to catalog materials.
               Should be in the format "mp-XXXXX" where XXXXX is a numerical ID. This ID
               corresponds to a specific material entry in the Materials Project database. [/DETAILED]
               [SYNTACTIC] Format: "mp-" followed by digits (e.g., "mp-149", "mp-20066") [/SYNTACTIC]
               [EXAMPLES] Examples: "mp-149" (Silicon), "mp-20066" (CO2), "mp-2" (Li) [/EXAMPLES]

    Returns:
        str: [BRIEF] CIF content string containing the crystal structure data. [/BRIEF]
             [DETAILED] A properly formatted CIF (Crystallographic Information File) string
             containing all necessary information about the crystal structure including lattice
             parameters, atomic positions, space group, and symmetry operations. This format
             is widely compatible with crystallographic software and other structure analysis tools. [/DETAILED]
             [EXAMPLES] Example output: "\n_chemical_formula_structural Si\n_cell_length_a 5.468..." [/EXAMPLES]

    [RAISES] Exceptions:
        ConnectionError: [ERROR_WHEN] When unable to connect to Materials Project API [/ERROR_WHEN]
                        [ERROR_DETAILS] Network connectivity issues or API server downtime [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check internet connection and MP_API_KEY environment variable [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When the specified MP ID is not found in the database [/ERROR_WHEN]
                 [ERROR_DETAILS] Invalid or non-existent material ID provided [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Verify MP ID exists on Materials Project website or check MP ID syntax[/ERROR_RECOVERY]
        AuthenticationError: [ERROR_WHEN] When API key is invalid or missing [/ERROR_WHEN]
                             [ERROR_DETAILS] MP_API_KEY environment variable not set or expired [/ERROR_DETAILS]
                             [ERROR_RECOVERY] Obtain valid API key from Materials Project and set environment variable [/ERROR_RECOVERY]
    [/RAISES]
    [LIMITATIONS] Known limitations:
    - Requires valid Materials Project API key to be set and internet connection
    - Limited to materials available in the Materials Project database
    - May not include the most recent experimental structures
    [/LIMITATIONS]
    """
    from mp_api.client import MPRester

    with MPRester(os.getenv("MP_API_KEY")) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[str(mp_id)], fields=["structure"]
        )
        structure = docs[0].structure

    return structure.to(fmt="cif")


####################################
### Tools relevant for ml training
####################################


# utility function
def get_bulk_polymorphs_data_func(composition: str) -> str:
    """
    Query the Materials Project database to find polymorphs for a given composition. This function returns
    a JSON string containing polymorph data including MP IDs, structures (CIF),  structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities,
    volumes, number of sites, symmetry, and stability. The results are sorted by energy above hull.

    Args:
        composition: Chemical composition (e.g., 'TiO2')
        api_key: Materials Project API key (optional if set in environment)

    Returns:
        JSON string containing polymorph data including MP IDs, structures (CIF),
        energies above hull, formation_energy_per_atom, band gaps, densities,
        volumes, number of sites, symmetry, and stability. (sorted by energy above hull)
    """

    from mp_api.client import MPRester

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Query for materials with the given composition
        docs = mpr.materials.summary.search(
            formula=composition,
            fields=[
                "material_id",
                "structure",
                "energy_above_hull",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "symmetry",
                "is_stable",
            ],
        )

        # Convert structures to CIF for easy storage
        polymorph_data = []
        for doc in docs:
            structure_cif = doc.structure.to(fmt="cif")

            polymorph_data.append(
                {
                    "material_id": doc.material_id,
                    "cif": structure_cif,
                    "energy_above_hull": doc.energy_above_hull,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap,
                    "density": doc.density,
                    "volume": doc.volume,
                    "nsites": doc.nsites,
                    "space_group": doc.symmetry.symbol,
                    "is_stable": doc.is_stable,
                }
            )

        # Sort by energy above hull (stability)
        polymorph_data = sorted(polymorph_data, key=lambda x: x["energy_above_hull"])

        return json.dumps(polymorph_data, indent=2)


def execute_python_code_given_code(code: str) -> str:
    """
    Executes a given Python code string.
    This is a placeholder and should be replaced with your actual implementation.
    """
    try:
        # Create a dictionary to hold local variables during execution
        exec_globals = {}
        exec_locals = {}
        exec(code, exec_globals, exec_locals)
        # Assuming the filtering code will produce a 'output' variable
        return json.dumps(
            {"success": True, "execution_result": {"output": exec_locals.get("output")}}
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@tool
def get_bulk_polymorphs_data(composition: str) -> str:
    """[BRIEF] Query Materials Project database to find all polymorphs for a given chemical composition. [/BRIEF]

    [DETAILED] This tool retrieves comprehensive polymorph data from the Materials Project database
    for a specific chemical composition. Polymorphs are different crystal structures with the same
    chemical formula but different atomic arrangements, leading to distinct physical and chemical
    properties. This tool could be relevant for retrieving structures of the same compoisition.
    Apart from structure for each polymorph Materials Project ID (MP ID), CIF structure, energy above hull,
    formation energy per atom, band gap, density, volume, number of sites, space group, and stability information is also retrieved. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to explore all known structural variants of a single material composition
    - Suitable for identifying thermodynamically stable and metastable phases and other properties like band gap, density, volume, number of sites, space group of the structure.
    - Recommended for retrieving structure and Materials Project ID (MP ID), CIF structure, energy above hull,
    formation energy per atom, band gap, density, volume, number of sites, space group of one single compoisition
    - Avoid when you only need a single, well-known structure (use get_structure_from_mp_text instead)
    - Avoid when you need data for multiple composition (use batch_retrieve_polymorphs instead)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Connects to Materials Project API using authentication credentials
    - Searches for all materials matching the specified chemical composition
    - Retrieves comprehensive data including energetics, structural, and electronic properties
    - Converts crystal structures to CIF format for compatibility with other tools
    - Sorts results by energy above hull (thermodynamic stability) for easy analysis
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you need to retrieve data for only one composition and there is no better tool [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve all polymorphs for a composition [/CURRENT]
    3. [FOLLOW_UP] Use select_polymorphs_with_strategy to filter results or batch_retrieve_polymorphs for multiple compositions [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [
    `get_bulk_polymorphs_data("TiO2")`,  # Retrieve polymorphs for titanium dioxide
    `get_bulk_polymorphs_data("SiO2")`,  # Retrieve polymorphs for silicon dioxide
    `get_bulk_polymorphs_data("Al2O3
    ]
    [/SYNTACTICAL]

    Args:
        composition: [BRIEF] Chemical composition formula. [/BRIEF]
                    [DETAILED] Chemical formula specifying the composition for which polymorphs
                    should be retrieved. Should follow standard chemical notation with element
                    symbols. The tool will find all known crystal structures
                    with this exact composition in the Materials Project database. [/DETAILED]
                    [SYNTACTIC] Format: "Standard chemical formula (e.g., TiO2, Al2O3, CaTiO3)" [/SYNTACTIC]
                    [EXAMPLES] Examples: "TiO2" (rutile, anatase, brookite), "SiO2" (quartz, cristobalite), "Fe2O3" (hematite, maghemite) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string containing comprehensive polymorph data sorted by stability. [/BRIEF]
             [DETAILED] A JSON-formatted string containing a list of dictionaries, each representing
             a polymorph with properties including Materials Project ID, CIF structure, energy above hull,
             formation energy per atom, band gap, density, volume, number of sites, space group, and
             stability information. Results are sorted by energy above hull for easy identification
             of the most stable phases. [/DETAILED]
             [EXAMPLES] Example output: '[{"material_id": "mp-2657", "cif": "...", "energy_above_hull": 0.0, "formation_energy_per_atom": -4.2, ...}]' [/EXAMPLES]

    [RAISES] Exceptions:
        KeyError: [ERROR_WHEN] When the specified compoisiton is not found in the database [/ERROR_WHEN]
            [ERROR_DETAILS] Invalid or non-existent material composition is provided [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify if the composition is valid[/ERROR_RECOVERY]
        ValueError: [ERROR_WHEN] When Materials Project API key is not available [/ERROR_WHEN]
                   [ERROR_DETAILS] MP_API_KEY environment variable not set or invalid [/ERROR_DETAILS]
                   [ERROR_RECOVERY] If API key is not set, the tool might not work, hence choose some other tool [/ERROR_RECOVERY]
        ConnectionError: [ERROR_WHEN] When unable to connect to Materials Project API [/ERROR_WHEN]
                        [ERROR_DETAILS] Network connectivity issues or API server downtime [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check internet connection if not working the tool might not work, hence choose some other tool [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to materials available in the Materials Project database
    - Requires valid API key and internet connection
    - Can retrieve polymorphs only for one composition at a time
    [/LIMITATIONS]
    """
    from mp_api.client import MPRester

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Query for materials with the given composition
        docs = mpr.materials.summary.search(
            formula=composition,
            fields=[
                "material_id",
                "structure",
                "energy_above_hull",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "symmetry",
                "is_stable",
            ],
        )

        # Convert structures to CIF for easy storage
        polymorph_data = []
        for doc in docs:
            structure_cif = doc.structure.to(fmt="cif")

            polymorph_data.append(
                {
                    "material_id": doc.material_id,
                    "cif": structure_cif,
                    "energy_above_hull": doc.energy_above_hull,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap,
                    "density": doc.density,
                    "volume": doc.volume,
                    "nsites": doc.nsites,
                    "space_group": doc.symmetry.symbol,
                    "is_stable": doc.is_stable,
                }
            )

        # Sort by energy above hull (stability)
        polymorph_data = sorted(polymorph_data, key=lambda x: x["energy_above_hull"])

        return json.dumps(polymorph_data, indent=2)


@tool
def get_bulk_polymorphs_data_to_file(
    composition: str, save_path: str | None = None
) -> str:
    """[BRIEF] Query Materials Project for polymorphs and save comprehensive data to a JSON file to give path. [/BRIEF]

     [DETAILED] This tool performs the same comprehensive polymorph retrieval as get_bulk_polymorphs_data
     but saves the results directly to a JSON file for persistent storage and later analysis.
    This tool retrieves comprehensive polymorph data from the Materials Project database
    for a specific chemical composition. Polymorphs are different crystal structures with the same
    chemical formula but different atomic arrangements, leading to distinct physical and chemical
    properties. This tool could be relevant for retrieving structures of the same compoisition.
    Apart from structure for each polymorph Materials Project ID (MP ID), CIF structure, energy above hull,
    formation energy per atom, band gap, density, volume, number of sites, space group, and stability information is also retrieved.
    The result is then saved a JSON file at the specified save_path.
    The file-based approach allows for efficient handling of large datasets and facilitates
    saving context of llm. [/DETAILED]

     [PROCEDURAL] When to use this tool:
     - Use when you need to explore all known structural variants of a single material composition.
     - Suitable for identifying thermodynamically stable and metastable phases and other properties like band gap, density, volume, number of sites, space group of the structure.
     - Use when you need to store polymorph data for later analysis or sharing or if you want to save context of llm.
     - Best suited for building persistent datasets and material databases
     - Highly recommended if the number of polymorphs for a compoisition could be very big
     - Avoid when you only need temporary data access (use get_bulk_polymorphs_data instead)
     - Avoid when you only need a single, well-known structure (use get_structure_from_mp_text instead)
     - Avoid when you need data for multiple composition (use batch_retrieve_polymorphs instead)
     [/PROCEDURAL]

     [CONTEXTUAL] How this tool works:
     - Connects to Materials Project API using authentication credentials
     - Searches for all materials matching the specified chemical composition
     - Retrieves comprehensive data including energetics, structural, and electronic properties
     - Converts crystal structures to CIF format for compatibility with other tools
     - Sorts results by energy above hull (thermodynamic stability) for easy analysis
     - Saves results to specified file path in JSON format with proper formatting
     - Ensures data persistence and enables later processing by other tools
     - Validates file path and creates directories as needed
     [/CONTEXTUAL]

     [WORKFLOW_INTEGRATION] Typical workflow integration:
     1. [PREREQUISITE] Ensure you need to retrieve data for only one composition and there is no better tool  [/PREREQUISITE]
     2. [CURRENT] Apply this tool to retrieve and save polymorph data and save to a file[/CURRENT]
     3. [FOLLOW_UP] Use consolidate_polymorph_datasets to combine multiple files of different composition or prepare_tabular_dataset[/FOLLOW_UP]
     [/WORKFLOW_INTEGRATION]

     [SYNTACTICAL] Usage examples:
     [
     `get_bulk_polymorphs_data_to_file("TiO2", "data/tio2_polymorphs.json")`,  # Save polymorphs for titanium dioxide
     `get_bulk_polymorphs_data_to_file("SiO2", "data/si2_polymorphs.json")`,  # Save polymorphs for silicon dioxide
     `get_bulk_polymorphs_data_to_file("Al2O3", "data/al2o3_polymorphs.json")`,  # Save polymorphs for aluminum oxide
     `get_bulk_polymorphs_data_to_file("Fe2O3", "data/fe2o3_polymorphs.json")`,  # Save polymorphs for iron
     ]
     [/SYNTACTICAL]

     Args:
         composition: [BRIEF] Chemical composition formula. [/BRIEF]
                     [DETAILED] Chemical formula specifying the composition for which polymorphs
                     should be retrieved and saved. Should follow standard chemical notation with
                     element symbols and subscripts. The tool will find all known crystal structures
                     with this exact composition in the Materials Project database. [/DETAILED]
                     [SYNTACTIC] Format: "Standard chemical formula (e.g., TiO2, Al2O3, CaTiO3)" [/SYNTACTIC]
                     [EXAMPLES] Examples: "TiO2" (titanium dioxide), "SiO2" (silicon dioxide), "Fe2O3" (iron oxide) [/EXAMPLES]
         save_path: [BRIEF] File path where JSON data will be saved. [/BRIEF]
                   [DETAILED] Complete file path including filename and extension where the polymorph
                   data will be saved. The path should be writable and the directory will be created
                   if it doesn't exist. Using .json extension is recommended for clarity. If None,
                   the tool will raise an error as the file path is required. [/DETAILED]
                   [SYNTACTIC] Format: "Valid file path with .json extension" [/SYNTACTIC]
                   [EXAMPLES] Examples: "data/tio2_polymorphs.json", "save_path/tio2_polymorphs.json", "results/Cu2O_polymorphsides.json" [/EXAMPLES]

     Returns:
         str: [BRIEF] File path where the polymorph data was saved. [/BRIEF]
              [DETAILED] Returns the exact file path where the JSON data was successfully written.
              This path can be used by subsequent tools for data loading and processing. The file
              contains comprehensive polymorph data in JSON format, sorted by thermodynamic stability. [/DETAILED]
              [EXAMPLES] Example output: "data/tio2_polymorphs.json" [/EXAMPLES]

     [RAISES] Exceptions:
         ValueError: [ERROR_WHEN] When save_path is None or API key is not available [/ERROR_WHEN]
                    [ERROR_DETAILS] Either save_path parameter is not provided or MP_API_KEY environment variable is missing [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Provide valid save_path. If  MP_API_KEY is not set, tool might not work and use a different tool [/ERROR_RECOVERY]
         IOError: [ERROR_WHEN] When unable to write to the specified file path [/ERROR_WHEN]
                 [ERROR_DETAILS] File path is not writable or directory doesn't exist [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check file permissions and ensure directory exists [/ERROR_RECOVERY]
     [/RAISES]

     [LIMITATIONS] Known limitations:
     - Requires writable file system access
     - Limited to materials available in the Materials Project database
     - Can retrieve polymorphs only for one composition at a time
     - Does not validate file format compatibility with other tools
     [/LIMITATIONS]
    """
    from mp_api.client import MPRester

    if save_path is None:
        raise ValueError("save_path must be provided to save the JSON data")

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Query for materials with the given composition
        docs = mpr.materials.summary.search(
            formula=composition,
            fields=[
                "material_id",
                "structure",
                "energy_above_hull",
                "formation_energy_per_atom",
                "band_gap",
                "density",
                "volume",
                "nsites",
                "symmetry",
                "is_stable",
            ],
        )

        # Convert structures to CIF for easy storage
        polymorph_data = []
        for doc in docs:
            structure_cif = doc.structure.to(fmt="cif")

            polymorph_data.append(
                {
                    "material_id": doc.material_id,
                    "cif": structure_cif,
                    "energy_above_hull": doc.energy_above_hull,
                    "formation_energy_per_atom": doc.formation_energy_per_atom,
                    "band_gap": doc.band_gap,
                    "density": doc.density,
                    "volume": doc.volume,
                    "nsites": doc.nsites,
                    "space_group": doc.symmetry.symbol,
                    "is_stable": doc.is_stable,
                }
            )

        # Sort by energy above hull (stability)
        polymorph_data = sorted(polymorph_data, key=lambda x: x["energy_above_hull"])
        json_str = json.dumps(polymorph_data, indent=2)

        # Save to file if path is provided
        if save_path:
            with Path(save_path).open("w") as f:
                f.write(json_str)

        return save_path


@tool
def batch_retrieve_polymorphs(
    compositions: list[str],
    max_energy_above_hull: float = 0.5,
    max_per_composition: int = 10,
    save_directory: str = "polymorph_data",
) -> str:
    """[BRIEF] Retrieve polymorphs for multiple chemical compositions efficiently in batch mode and save it to given directory as json. [/BRIEF]

    [DETAILED] This tool performs polymorph retrieval for multiple chemical compositions
    simultaneouslys. The tool applies energy and count filters to focus on thermodynamically relevant phases.
    This tool retrieves comprehensive polymorph data from the Materials Project database
    for each chemical composition and save in the user inputted save_directory. Polymorphs are different crystal structures with the same
    chemical formula but different atomic arrangements, leading to distinct physical and chemical
    properties. This tool could be relevant for retrieving structures of the same compoisition for multiple composition.
    Apart from structure for each polymorph Materials Project ID (MP ID), CIF structure, energy above hull,
    formation energy per atom, band gap, density, volume, number of sites, space group, and stability information is also retrieved.
    The result is then saved a JSON file at the specified directory in the format <composition>_polymorphs.json.
    [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to systematically explore multiple chemical compositions
    - Best suited for high-throughput materials screening and dataset preperation
    - Use when you need to store polymorph data for later analysis or sharing or if you want to save context of llm.
    - Best suited for building persistent datasets and material databases
    - Avoid when you only need detailed analysis of a single composition
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Iterates through list of compositions using robust error handling
    - Connects to Materials Project API using authentication credentials
    - Searches for all materials matching the specified chemical composition
    - Retrieves comprehensive data including energetics, structural, and electronic properties
    - Converts crystal structures to CIF format for compatibility with other tools
    - Applies energy filtering to focus on thermodynamically accessible phases
    - Limits number of structures per composition to prevent data explosion
    - Creates organized directory structure for systematic data storage
    - Provides comprehensive success/failure reporting for quality control
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE]Ensure you need to retrieve data for more than one composition and there is no better tool  [/PREREQUISITE]
    2. [CURRENT] Apply this tool to retrieve polymorphs for multiple compositions [/CURRENT]
    3. [FOLLOW_UP] Use consolidate_polymorph_datasets to combine results or select_polymorphs_with_strategy for filtering [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [
    `batch_retrieve_polymorphs(["TiO2", "SiO2", "Al2O3"], 0.3, 5, "oxides_data")`,  # Retrieve polymorphs for multiple oxides
    `batch_retrieve_polymorphs(["CaTiO3", "SrTiO3", "BaTiO3"], 0.5, 10, "perovskites")`,  # Retrieve polymorphs for perovskites
    `batch_retrieve_polymorphs(["FeO", "Fe2O3", "Fe3O4"], 0.2, 8, "iron_oxides")`,  # Retrieve polymorph
    ]
    [/SYNTACTICAL]

    Args:
        compositions: [BRIEF] List of chemical compositions to retrieve. [/BRIEF]
                     [DETAILED] List of chemical formulas for which polymorphs should be retrieved.
                     Each composition should follow standard chemical notation. The tool will process
                     each composition independently and provide detailed success/failure reporting.
                     Large lists are supported but may take significant time to process. [/DETAILED]
                     [SYNTACTIC] Format: ["composition1", "composition2", ...] [/SYNTACTIC]
                     [EXAMPLES] Examples: ["TiO2", "SiO2", "Al2O3"], ["CaTiO3", "SrTiO3"], ["FeO", "Fe2O3"] [/EXAMPLES]
        max_energy_above_hull: [BRIEF] Maximum energy above hull threshold in eV/atom. Defaults to 0.5. [/BRIEF]
                              [DETAILED] Energy threshold above the convex hull for including polymorphs.
                              Only phases with energy above hull less than or equal to this value will
                              be included. This filters out highly unstable phases while retaining
                              potentially accessible metastable phases. Lower values give more stable
                              phases but may miss interesting metastable structures. [/DETAILED]
                              [SYNTACTIC] Format: positive float representing energy in eV/atom [/SYNTACTIC]
                              [EXAMPLES] Examples: 0.1 (very stable), 0.5 (standard), 1.0 (include metastable) [/EXAMPLES]
        max_per_composition: [BRIEF] Maximum number of polymorphs per composition. Defaults to 10. [/BRIEF]
                            [DETAILED] Maximum number of polymorphs to retrieve for each composition,
                            taken from the most stable phases first. This prevents data explosion for
                            compositions with many known phases while ensuring the most important
                            structures are captured. Higher values provide more comprehensive coverage
                            but increase dataset size and processing time. [/DETAILED]
                            [SYNTACTIC] Format: positive integer [/SYNTACTIC]
                            [EXAMPLES] Examples: 5 (focused), 10 (standard), 20 (comprehensive) [/EXAMPLES]
        save_directory: [BRIEF] Directory path for saving individual composition files. Defaults to "polymorph_data". [/BRIEF]
                       [DETAILED] Base directory where individual JSON files for each composition will be saved.
                       The directory will be created if it doesn't exist. Each composition will have its own
                       JSON file named with the composition formula. This organization facilitates easy
                       data management and selective loading of specific compositions. [/DETAILED]
                       [SYNTACTIC] Format: "Valid directory path" [/SYNTACTIC]
                       [EXAMPLES] Examples: "data/polymorphs", "materials/oxides", "results/batch_data" [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with batch retrieval results and statistics. [/BRIEF]
             [DETAILED] A comprehensive JSON report containing lists of successfully processed and
             failed compositions, total number of polymorphs retrieved, file paths for each composition,
             and summary statistics. This enables quality control and tracking of the batch processing
             workflow. [/DETAILED]
             [EXAMPLES] Example output: '{"successful_compositions": ["TiO2", "SiO2"], "failed_compositions": ["BadFormula"], "total_polymorphs": 15, "composition_files": {...}}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When parameters are invalid (negative energy, zero max_per_composition) [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid parameter values or missing API key [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Check parameter values and ensure MP_API_KEY is set [/ERROR_RECOVERY]
        IOError: [ERROR_WHEN] When unable to create save directory or write files [/ERROR_WHEN]
                [ERROR_DETAILS] Directory creation failed or insufficient write permissions [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check directory permissions and available disk space [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires writable file system access
    - Limited to materials available in the Materials Project database
    - Processing time scales linearly with number of compositions
    - Individual composition failures don't stop the entire batch
    - Does not validate file format compatibility with other tools
    [/LIMITATIONS]
    """
    from pathlib import Path

    if isinstance(max_energy_above_hull, str):
        max_energy_above_hull = float(max_energy_above_hull)
    if isinstance(max_per_composition, str):
        max_per_composition = int(max_per_composition)

    # Create save directory
    Path(save_directory).mkdir(exist_ok=True)

    results = {
        "successful_compositions": [],
        "failed_compositions": [],
        "total_polymorphs": 0,
        "composition_files": {},
    }

    for composition in compositions:
        try:
            # Get polymorphs for this composition
            polymorphs_json = get_bulk_polymorphs_data_func(composition)
            polymorphs = json.loads(polymorphs_json)

            # Filter by energy and limit count
            filtered_polymorphs = [
                p
                for p in polymorphs
                if p.get("energy_above_hull", 999) <= max_energy_above_hull
            ][:max_per_composition]

            if filtered_polymorphs:
                # Save to individual file
                save_path = Path(save_directory) / f"{composition}_polymorphs.json"
                with save_path.open("w") as f:
                    json.dump(filtered_polymorphs, f, indent=2)

                results["successful_compositions"].append(composition)
                results["composition_files"][composition] = str(save_path)
                results["total_polymorphs"] += len(filtered_polymorphs)
            else:
                results["failed_compositions"].append(composition)

        except Exception as e:
            results["failed_compositions"].append(composition)
            logger.error(f"Failed to retrieve polymorphs for {composition}: {e}")

    return json.dumps(results, indent=2)


@tool
def sort_and_get_first_from_json(
    polymorph_data_json: str, sort_key: str, return_key: str
) -> str:
    """[BRIEF] Sort JSON data by specified key and return the first element's specified value. [/BRIEF]

    [DETAILED] This utility tool provides flexible sorting and extraction capabilities for JSON data,
    particularly useful for materials data analysis where you need to identify optimal structures
    based on specific criteria. It enables quick identification of the best material according to
    any numerical property, such as finding the most stable phase, highest band gap material, or
    densest structure. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to quickly identify the best material from a dataset
    - Best suited for extracting optimal values from sorted lists
    - Recommended for picking materials with desired properties like lowest energy, highest band gap, etc.
    - Avoid when you need multiple values or complex filtering criteria
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses JSON string into Python data structure
    - Applies sorting based on specified key using numerical comparison
    - Extracts the first element after sorting (best/optimal value)
    - Returns the specified property value from the optimal element
    - Handles various data types and provides robust error handling
    - To find the right keys from polymorph data maybe use io tools or python tools
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First obtain JSON data from polymorph retrieval tools [/PREREQUISITE]
    2. [CURRENT] Apply this tool to identify optimal material based on specific criteria [/CURRENT]
    3. [FOLLOW_UP] Use the returned value for further analysis or material selection [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `sort_and_get_first_from_json(polymorphs_json, "energy_above_hull", "material_id")`,
    `sort_and_get_first_from_json(polymorphs_json, "band_gap", "cif")`,
    `sort_and_get_first_from_json(polymorphs_json, "density", "formation_energy_per_atom")`,
    [/SYNTACTICAL]

    Args:
        polymorph_data_json: [BRIEF] JSON string containing the data to be sorted. [/BRIEF]
                            [DETAILED] A JSON-formatted string containing a list of dictionaries,
                            each representing a material or structure with various properties.
                            The data should be structured consistently with numerical values
                            for the sorting key. This is typically output from polymorph
                            retrieval tools. [/DETAILED]
                            [SYNTACTIC] Format: "Valid JSON string containing list of dictionaries" [/SYNTACTIC]
                            [EXAMPLES] Examples: JSON from get_bulk_polymorphs_data output [/EXAMPLES]
        sort_key: [BRIEF] Property name to sort the data by. [/BRIEF]
                 [DETAILED] The dictionary key name that will be used for sorting the data.
                 This should correspond to a numerical property in the JSON data. The sorting
                 is performed in ascending order, so the first element will have the smallest
                 value for this property. Common keys include energy_above_hull, band_gap,
                 density, formation_energy_per_atom. [/DETAILED]
                 [SYNTACTIC] Format: "String matching a key in the JSON data dictionaries" [/SYNTACTIC]
                 [EXAMPLES] Examples: "energy_above_hull", "band_gap", "density"[/EXAMPLES]
        return_key: [BRIEF] Property name to return from the first element after sorting. [/BRIEF]
                   [DETAILED] The dictionary key name for the value that should be returned
                   from the first (optimal) element after sorting. This allows extraction
                   of any property from the optimal structure, such as material_id for
                   identification, cif for structure, or any other calculated property. [/DETAILED]
                   [SYNTACTIC] Format: "String matching a key in the JSON data dictionaries" [/SYNTACTIC]
                   [EXAMPLES] Examples: "material_id", "cif", "formation_energy_per_atom"[/EXAMPLES]

    Returns:
        str: [BRIEF] Value of the specified return_key from the first element after sorting. [/BRIEF]
             [DETAILED] The value corresponding to the return_key from the material that has
             the smallest value for the sort_key. This could be a string (like material_id
             or CIF), a number (like energy or band gap), or any other data type stored
             in the JSON. The returned value represents the optimal material according
             to the specified sorting criterion. [/DETAILED]
             [EXAMPLES] Example outputs: "mp-2657" (material ID), "1.23" (energy value), CIF structure string [/EXAMPLES]

    [RAISES] Exceptions:
        JSONDecodeError: [ERROR_WHEN] When the polymorph_data_json string is not valid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON string or incorrect format [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and ensure proper string escaping [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When sort_key or return_key is not found in the data [/ERROR_WHEN]
                 [ERROR_DETAILS] Specified keys don't exist in the JSON data dictionaries [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check available keys in the JSON data and use valid key names [/ERROR_RECOVERY]
        IndexError: [ERROR_WHEN] When the JSON data is empty or contains no elements [/ERROR_WHEN]
                   [ERROR_DETAILS] Empty list or no valid data after parsing [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Ensure JSON data contains at least one element [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Only returns the first element after sorting (single optimal result)
    - Sorting is performed in ascending order only
    - Does not handle complex sorting criteria or multiple keys
    - May not work properly with non-numerical sort keys
    [/LIMITATIONS]
    """
    data = json.loads(polymorph_data_json)

    # Sort the data based on the given key
    sorted_data = sorted(data, key=lambda x: x[sort_key])

    # Return the value of the specified key from the first element
    return sorted_data[0][return_key]


@tool
def select_polymorphs_with_strategy(
    polymorphs_data: str,
    selection_strategy: str = "diverse_energy",
    max_polymorphs: int = 5,
    energy_threshold: float = 0.5,
    is_path: bool = False,
) -> str:
    """[BRIEF] Select polymorphs using strategic criteria for systematic materials analysis. [/BRIEF]

    [DETAILED] This tool implements intelligent selection strategies for polymorph datasets, enabling
    systematic reduction of large materials databases while preserving important structural and
    energetic diversity. The input can be json polymorph data as string or path to json file of polymorph data.
    It supports multiple selection algorithms designed for different research
    objectives, from stability-focused studies to comprehensive structural surveys. This is crucial
    for managing computational resources and focusing analysis on the most relevant materials. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to systematically reduce large polymorph datasets
    - Essential for creating representative training sets for machine learning
    - Recommended for comparative studies requiring diverse structural examples
    - Avoid when you need all available data or have specific material requirements
    - Avoid when you have a custom logic for selection (use custom Python code instead)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Applies energy threshold filtering to focus on accessible phases
    - Implements multiple selection algorithms based on different criteria
    - "diverse_energy" spreads selection across energy range for representative sampling
    - "most_stable" prioritizes thermodynamically favored phases
    - "diverse_structure" ensures different space groups are represented
    - Returns optimized subset maintaining important characteristics
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First obtain polymorph data using get_bulk_polymorphs_data or batch_retrieve_polymorphs [/PREREQUISITE]
    2. [CURRENT] Apply this tool to select representative subset based on strategy [/CURRENT]
    3. [FOLLOW_UP] Use selected polymorphs for slab generation, ML dataset preparation, or detailed analysis. You can even use this multiple times to prepare a set with different strategies [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `select_polymorphs_with_strategy(polymorphs_json, "most_stable", 3, 0.3, False)`,
    `select_polymorphs_with_strategy("data/polymorphs.json", "diverse_structure", 5, 0.5, True)`,
    `select_polymorphs_with_strategy(polymorphs_json, "diverse_energy", 8, 0.8, False)`,
    [/SYNTACTICAL]

    Args:
        polymorphs_data: [BRIEF] JSON string or file path containing polymorph data. [/BRIEF]
                        [DETAILED] Either a JSON-formatted string containing polymorph data or a file path
                        to a JSON file, depending on the is_path parameter. The data should contain
                        polymorphs with properties like energy_above_hull, space_group, and other
                        structural/energetic information. This is typically output from polymorph
                        retrieval tools. [/DETAILED]
                        [SYNTACTIC] Format: "JSON string or valid file path" [/SYNTACTIC]
                        [EXAMPLES] Examples: JSON string from get_bulk_polymorphs_data, "data/polymorphs.json" [/EXAMPLES]
        selection_strategy: [BRIEF] Strategy for polymorph selection. Defaults to "diverse_energy". [/BRIEF]
                           [DETAILED] The algorithm used for selecting polymorphs from the dataset.
                           "diverse_energy" selects polymorphs distributed across the energy range
                           for representative sampling. "most_stable" prioritizes the most
                           thermodynamically stable phases. "diverse_structure" ensures different
                           space groups are represented to capture structural diversity. [/DETAILED]
                           [SYNTACTIC] Format: "diverse_energy", "most_stable", or "diverse_structure" [/SYNTACTIC]
                           [EXAMPLES] Examples: "most_stable" (stability focus), "diverse_structure" (structural diversity), "diverse_energy" (energy sampling) [/EXAMPLES]
        max_polymorphs: [BRIEF] Maximum number of polymorphs to select. Defaults to 5. [/BRIEF]
                       [DETAILED] The maximum number of polymorphs to include in the final selection.
                       This parameter controls the size of the resulting dataset and should be chosen
                       based on computational resources and analysis requirements. Larger values
                       provide more comprehensive coverage but increase processing time and
                       computational cost. [/DETAILED]
                       [SYNTACTIC] Format: positive integer [/SYNTACTIC]
                       [EXAMPLES] Examples: 3 (focused), 5 (standard), 10 (comprehensive) [/EXAMPLES]
        energy_threshold: [BRIEF] Maximum energy above hull in eV/atom. Defaults to 0.5. [/BRIEF]
                         [DETAILED] Energy threshold above the convex hull for including polymorphs
                         in the selection process. Only phases with energy above hull less than
                         or equal to this value will be considered. This pre-filtering step ensures
                         that only thermodynamically accessible phases are included in the analysis. [/DETAILED]
                         [SYNTACTIC] Format: positive float representing energy in eV/atom [/SYNTACTIC]
                         [EXAMPLES] Examples: 0.1 (very stable), 0.5 (moderate), 1.0 (include metastable) [/EXAMPLES]
        is_path: [BRIEF] Whether polymorphs_data is a file path. Defaults to False. [/BRIEF]
                [DETAILED] Boolean flag indicating whether the polymorphs_data parameter should
                be treated as a file path (True) or as a JSON string (False). When True, the
                tool will read the JSON data from the specified file. When False, it will
                parse the data directly from the string. [/DETAILED]
                [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                [EXAMPLES] Examples: True (file path), False (JSON string) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string containing selected polymorphs based on the specified strategy. [/BRIEF]
             [DETAILED] A JSON-formatted string containing the selected subset of polymorphs,
             maintaining the same data structure as the input but with reduced number of entries.
             The selection preserves important characteristics according to the chosen strategy
             while reducing dataset size for efficient processing. [/DETAILED]
             [EXAMPLES] Example output: JSON string with 3-10 selected polymorphs based on strategy [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When invalid selection strategy is specified [/ERROR_WHEN]
                   [ERROR_DETAILS] Strategy name not recognized or invalid parameters [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Use valid strategy names: "diverse_energy", "most_stable", "diverse_structure" [/ERROR_RECOVERY]
        FileNotFoundError: [ERROR_WHEN] When is_path=True but file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Specified file path cannot be found or accessed [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Check file path and ensure file exists [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When polymorphs_data contains invalid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON string or corrupted file [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and data integrity [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Selection strategies are predefined and not customizable
    - Energy threshold applies uniformly to all polymorphs
    - Does not consider complex multi-objective optimization
    - May not preserve specific structural features of interest
    [/LIMITATIONS]
    """
    if is_path:
        with Path(polymorphs_data).open("r") as f:
            polymorphs = json.loads(f.read())
    else:
        polymorphs = json.loads(polymorphs_data)

    # Filter by energy threshold
    filtered = [p for p in polymorphs if p["energy_above_hull"] <= energy_threshold]

    if selection_strategy == "most_stable":
        # Sort by energy above hull, take most stable
        selected = sorted(filtered, key=lambda x: x["energy_above_hull"])[
            :max_polymorphs
        ]

    elif selection_strategy == "diverse_energy":
        # Select polymorphs with diverse energies
        sorted_polymorphs = sorted(filtered, key=lambda x: x["energy_above_hull"])
        selected = []
        if sorted_polymorphs:
            step = max(1, len(sorted_polymorphs) // max_polymorphs)
            for i in range(0, min(len(sorted_polymorphs), max_polymorphs * step), step):
                selected.append(sorted_polymorphs[i])

    elif selection_strategy == "diverse_structure":
        # Select polymorphs with different space groups
        selected = []
        seen_space_groups = set()
        for p in sorted(filtered, key=lambda x: x["energy_above_hull"]):
            if (
                p["space_group"] not in seen_space_groups
                and len(selected) < max_polymorphs
            ):
                selected.append(p)
                seen_space_groups.add(p["space_group"])

    else:
        # Default: take first max_polymorphs
        selected = filtered[:max_polymorphs]

    return json.dumps(selected, indent=2)


@tool
def consolidate_polymorph_datasets(
    composition_files: dict[str, str],
    output_path: str = "consolidated_polymorphs.json",
) -> str:
    """[BRIEF] Consolidate multiple polymorph JSON files into a single comprehensive dataset. [/BRIEF]

    [DETAILED] This tool combines multiple polymorph datasets from different compositions into a
    unified dataset suitable for dataset preperation and machine learning applications. It handles
    data integration and provides comprehensive statistics about
    the consolidated dataset. Often you have multiple polumorph json file and you want to combine them. You can use this tool to combine them [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to combine multiple datasets
    - Best suited for building comprehensive materials databases
    - Recommended for for combining results from batch_retrieve_polymorphs
    - Avoid when you need to maintain composition-specific organization
    - AVoid when you have a single polymorph file or no files to combine
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Reads multiple JSON files specified in the composition_files dictionary
    - Merges data while maintaining source composition information
    - Adds provenance metadata to track data origins
    - Provides comprehensive statistics about the consolidated dataset
    - Handles missing files and corrupted data gracefully
    - Saves the consolidated dataset to a specified output file
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First use batch_retrieve_polymorphs to create multiple composition files or run get_bulk_polymorphs_data multiple times to ahve data for multiple composition [/PREREQUISITE]
    2. [CURRENT] Apply this tool to consolidate separate files into unified dataset [/CURRENT]
    3. [FOLLOW_UP] Use prepare_tabular_dataset or prepare_neural_network_dataset for ML preparation [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `consolidate_polymorph_datasets({"TiO2": "data/tio2.json", "SiO2": "data/sio2.json"})`,
    `consolidate_polymorph_datasets(composition_files_dict, "materials_database.json")`,
    `consolidate_polymorph_datasets(batch_results["composition_files"], "consolidated.json")`,
    [/SYNTACTICAL]

    Args:
        composition_files: [BRIEF] Dictionary mapping compositions to their JSON file paths. [/BRIEF]
                          [DETAILED] A dictionary where keys are composition names/formulas and values
                          are file paths to their corresponding JSON files containing polymorph data.
                          This is typically the output from batch_retrieve_polymorphs. The tool will
                          attempt to read each file and integrate the data while maintaining composition
                          information. [/DETAILED]
                          [SYNTACTIC] Format: '{"composition1": "path1.json", "composition2": "path2.json", ...}' [/SYNTACTIC]
                          [EXAMPLES] Examples: {"TiO2": "data/tio2_polymorphs.json", "SiO2": "data/sio2_polymorphs.json"} [/EXAMPLES]
        output_path: [BRIEF] Path for the consolidated dataset file. Defaults to "consolidated_polymorphs.json". [/BRIEF]
                    [DETAILED] File path where the consolidated dataset will be saved. The file will
                    contain all polymorphs from all compositions in a single JSON structure with
                    added source composition information. The directory will be created if it doesn't
                    exist. Using .json extension is recommended for clarity. [/DETAILED]
                    [SYNTACTIC] Format: "Valid file path with .json extension" [/SYNTACTIC]
                    [EXAMPLES] Examples: "consolidated_polymorphs.json", "data/all_materials.json", "datasets/complete_set.json" [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with consolidation results and comprehensive statistics. [/BRIEF]
             [DETAILED] A JSON-formatted string containing consolidation status, output file path,
             and detailed statistics including total number of polymorphs, number of compositions
             successfully included, average polymorphs per composition, and any processing errors.
             This enables quality control and assessment of the consolidation process. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "output_path": "consolidated.json", "statistics": {"total_polymorphs": 150, "compositions_included": 15, ...}}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When one or more input files cannot be found [/ERROR_WHEN]
                          [ERROR_DETAILS] File paths in composition_files dictionary are invalid [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Check file paths and ensure all files exist [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When input files contain invalid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Corrupted or malformed JSON in input files [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format of input files [/ERROR_RECOVERY]
        IOError: [ERROR_WHEN] When unable to write to output path [/ERROR_WHEN]
                [ERROR_DETAILS] Output path is not writable or directory doesn't exist [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check write permissions and ensure output directory exists [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Memory usage scales with total dataset size
    - Does not perform deduplication of identical structures
    - May not handle very large individual files efficiently
    - Does not validate data consistency across files
    [/LIMITATIONS]
    """
    all_polymorphs = []
    stats = {
        "total_polymorphs": 0,
        "compositions_included": 0,
        "average_per_composition": 0,
    }

    for composition, file_path in composition_files.items():
        try:
            with Path(file_path).open() as f:
                polymorphs = json.load(f)

            # Add composition information to each polymorph
            for polymorph in polymorphs:
                polymorph["source_composition"] = composition

            # Extend the all_polymorphs list with the polymorphs from the current file
            all_polymorphs.extend(polymorphs)

            stats["compositions_included"] += 1

        except FileNotFoundError:
            logger.warning(
                f"File not found for {composition} at {file_path}. Skipping."
            )
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {file_path}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to process {composition} from {file_path}: {e}")

    stats["total_polymorphs"] = len(all_polymorphs)
    if stats["compositions_included"] > 0:
        stats["average_per_composition"] = int(
            stats["total_polymorphs"] / stats["compositions_included"]
        )
    else:
        stats["average_per_composition"] = 0

    # Save consolidated dataset
    try:
        with Path(output_path).open("w") as f:
            json.dump(all_polymorphs, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save consolidated dataset to {output_path}: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)

    return json.dumps(
        {"success": True, "output_path": output_path, "statistics": stats}, indent=2
    )


@tool
def select_polymorphs_with_strategy_to_file(
    polymorphs_data: str,
    save_path: str,
    selection_strategy: str = "diverse_energy",
    max_polymorphs: int = 5,
    energy_threshold: float = 0.5,
    is_path: bool = False,
) -> str:
    """[BRIEF] Select polymorphs using strategic criteria and save results to file for persistent storage. [/BRIEF]

    [DETAILED] This tool combines the strategic polymorph selection capabilities with direct file output for
    persistent storage and workflow automation. It implements the same selection algorithms as
    select_polymorphs_with_strategy but automatically saves results to a specified file path.
    The input can be json polymorph data as string or path to json file of polymorph data.
    It supports multiple selection algorithms designed for different research objectives, from stability-focused structures to comprehensive structures.
    This is essential for automated workflows, batch processing, and creating organized datasets where selected polymorphs need
    to be stored for later use or sharing. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need strategic polymorph selection with automatic file storage
    - Best suited for automated workflows and batch processing pipelines
    - Essential for creating organized datasets that will be shared or archived
    - Recommended when building systematic collections of selected materials
    - Avoid when you only need temporary selection results in memory
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Applies identical selection strategies as select_polymorphs_with_strategy
    - Processes energy threshold filtering and strategic selection algorithms
    - Automatically saves selected polymorphs to specified file path in JSON format
    - Ensures proper file formatting and directory creation as needed
    - Returns file path for integration with downstream tools
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First obtain polymorph data using get_bulk_polymorphs_data or load from file [/PREREQUISITE]
    2. [CURRENT] Apply strategic selection and save results to persistent file storage [/CURRENT]
    3. [FOLLOW_UP] Use saved file with find_all_unique_slabs_upto_millerindex or other structural analysis tools [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `select_polymorphs_with_strategy_to_file(polymorphs_json, "selected_tio2.json", "most_stable", 3, 0.3, False)`,
    `select_polymorphs_with_strategy_to_file("data/polymorphs.json", "output/diverse.json", "diverse_structure", 5, 0.5, True)`,
    `select_polymorphs_with_strategy_to_file(batch_data, "results/selected_materials.json", "diverse_energy", 8, 0.8, False)`,
    [/SYNTACTICAL]

    Args:
        polymorphs_data: [BRIEF] JSON string or file path containing polymorph data. [/BRIEF]
                        [DETAILED] Either a JSON-formatted string containing polymorph data or a file path
                        to a JSON file, depending on the is_path parameter. The data should contain
                        polymorphs with properties like energy_above_hull, space_group, and other
                        structural/energetic information for strategic selection. [/DETAILED]
                        [SYNTACTIC] Format: "JSON string or valid file path" [/SYNTACTIC]
                        [EXAMPLES] Examples: JSON string from get_bulk_polymorphs_data, "data/polymorphs.json" [/EXAMPLES]

        save_path: [BRIEF] File path where selected polymorphs will be saved. [/BRIEF]
                  [DETAILED] Complete file path where the selected polymorph subset will be saved in JSON format.
                  The directory will be created if it doesn't exist. This file can be used by subsequent tools
                  or shared with collaborators. Using .json extension is recommended for clarity. [/DETAILED]
                  [SYNTACTIC] Format: "Valid file path with .json extension" [/SYNTACTIC]
                  [EXAMPLES] Examples: "selected_polymorphs.json", "data/tio2_selected.json", "results/diverse_materials.json" [/EXAMPLES]

        selection_strategy: [BRIEF] Strategy for polymorph selection. Defaults to "diverse_energy". [/BRIEF]
                           [DETAILED] The algorithm used for selecting polymorphs from the dataset. Options include
                           "diverse_energy" for energy range sampling, "most_stable" for thermodynamic stability,
                           and "diverse_structure" for structural diversity. Each strategy optimizes for different
                           research objectives and analysis requirements. [/DETAILED]
                           [SYNTACTIC] Format: "diverse_energy", "most_stable", or "diverse_structure" [/SYNTACTIC]
                           [EXAMPLES] Examples: "most_stable" (stability focus), "diverse_structure" (structural variety), "diverse_energy" (representative sampling) [/EXAMPLES]

        max_polymorphs: [BRIEF] Maximum number of polymorphs to select. Defaults to 5. [/BRIEF]
                       [DETAILED] The maximum number of polymorphs to include in the final selection and save to file.
                       This parameter controls dataset size and should be chosen based on computational resources
                       and analysis requirements. Larger values provide more comprehensive coverage but increase
                       processing time. [/DETAILED]
                       [SYNTACTIC] Format: positive integer [/SYNTACTIC]
                       [EXAMPLES] Examples: 3 (focused selection), 5 (standard), 10 (comprehensive coverage) [/EXAMPLES]

        energy_threshold: [BRIEF] Maximum energy above hull in eV/atom. Defaults to 0.5. [/BRIEF]
                         [DETAILED] Energy threshold above the convex hull for including polymorphs in the selection
                         process. Only phases with energy above hull less than or equal to this value will be
                         considered. This pre-filtering ensures thermodynamic accessibility of selected phases. [/DETAILED]
                         [SYNTACTIC] Format: positive float representing energy in eV/atom [/SYNTACTIC]
                         [EXAMPLES] Examples: 0.1 (very stable only), 0.5 (moderate threshold), 1.0 (include metastable) [/EXAMPLES]

        is_path: [BRIEF] Whether polymorphs_data is a file path. Defaults to False. [/BRIEF]
                [DETAILED] Boolean flag indicating whether the polymorphs_data parameter should be treated as
                a file path (True) or as a JSON string (False). When True, the tool will read the JSON data
                from the specified file. This enables flexible input handling for different workflow patterns. [/DETAILED]
                [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                [EXAMPLES] Examples: True (file input), False (JSON string input) [/EXAMPLES]

    Returns:
        str: [BRIEF] File path where the selected polymorphs were saved. [/BRIEF]
             [DETAILED] The complete file path where the selected polymorphs have been successfully saved.
             This path can be used by subsequent tools for loading the selected dataset or for verification
             that the file was created correctly. The file contains the subset of polymorphs selected
             according to the specified strategy. [/DETAILED]
             [EXAMPLES] Example output: "data/selected_tio2_polymorphs.json" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When invalid selection strategy is specified [/ERROR_WHEN]
                   [ERROR_DETAILS] Strategy name not recognized or invalid parameters [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Use valid strategy names: "diverse_energy", "most_stable", "diverse_structure" [/ERROR_RECOVERY]
        FileNotFoundError: [ERROR_WHEN] When is_path=True but input file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Specified input file path cannot be found or accessed [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Check input file path and ensure file exists [/ERROR_RECOVERY]
        IOError: [ERROR_WHEN] When unable to write to the save_path location [/ERROR_WHEN]
                [ERROR_DETAILS] Output directory doesn't exist or insufficient write permissions [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check output directory permissions and ensure path is writable [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Selection strategies are predefined and not customizable
    - File overwriting occurs without warning if save_path already exists
    - Cannot validate file format compatibility with specific downstream tools
    - Energy threshold applies uniformly without consideration of composition differences
    [/LIMITATIONS]
    """
    if is_path:
        with Path(polymorphs_data).open("r") as f:
            polymorphs = json.loads(f.read())
    else:
        polymorphs = json.loads(polymorphs_data)

    # Filter by energy threshold
    filtered = [p for p in polymorphs if p["energy_above_hull"] <= energy_threshold]

    if selection_strategy == "most_stable":
        selected = sorted(filtered, key=lambda x: x["energy_above_hull"])[
            :max_polymorphs
        ]

    elif selection_strategy == "diverse_energy":
        sorted_polymorphs = sorted(filtered, key=lambda x: x["energy_above_hull"])
        selected = []
        if sorted_polymorphs:
            step = max(1, len(sorted_polymorphs) // max_polymorphs)
            for i in range(0, min(len(sorted_polymorphs), max_polymorphs * step), step):
                selected.append(sorted_polymorphs[i])

    elif selection_strategy == "diverse_structure":
        selected = []
        seen_space_groups = set()
        for p in sorted(filtered, key=lambda x: x["energy_above_hull"]):
            if (
                p["space_group"] not in seen_space_groups
                and len(selected) < max_polymorphs
            ):
                selected.append(p)
                seen_space_groups.add(p["space_group"])

    else:
        selected = filtered[:max_polymorphs]

    # Save to file
    with Path(save_path).open("w") as f:
        json.dump(selected, f, indent=2)

    return save_path


@tool
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    """[BRIEF] Execute Python code in a secure environment with data input/output capabilities. [/BRIEF]

    [DETAILED] This tool provides a secure execution environment for custom Python code, essential for
    data analysis, custom calculations, and algorithm development in materials science workflows. It
    supports data injection, output capture, and file saving capabilities while maintaining security
    through process isolation and timeout controls. This enables flexible custom analysis. [/DETAILED]

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
                    [DETAILED] A string containing valid Python code to be executed in thes
                    environment. For best results, assign your main output to a variable named
                    'result' or 'output'. The code can import standard libraries and perform
                    complex calculations. The tool will attempt to capture user-defined variables
                    as execution results. [/DETAILED]
                    [SYNTACTIC] Format: "Valid Python code string" [/SYNTACTIC]
                    [EXAMPLES] Examples: "result = 2 + 2", "import json; result = json.loads(data)", "filtered = [x for x in data if x > threshold]" [/EXAMPLES]
        input_data: [BRIEF] Optional JSON string to inject as input_data variable. [/BRIEF]
                   [DETAILED] An optional JSON string that will be loaded into a Python variable
                   named 'input_data' within the executed script. This allows the script to
                   process external data. The JSON will be parsed and made available as a Python
                   object (dict, list, etc.) depending on the JSON structure. [/DETAILED]
                   [SYNTACTIC] Format: "Valid JSON string or None" [/SYNTACTIC]
                   [EXAMPLES] Examples: '{"data": [1, 2, 3]}', '[1, 2, 3, 4, 5]', '{"threshold": 0.5, "values": [...]}' [/EXAMPLES]
        save_output_to: [BRIEF] Optional file path to save execution results. [/BRIEF]
                       [DETAILED] An optional file path where the captured execution results will
                       be saved as a JSON file. If provided and execution is successful, the
                       results will be written to this file for persistence and later use.
                       The directory will be created if it doesn't exist. [/DETAILED]
                       [SYNTACTIC] Format: "Valid file path or None" [/SYNTACTIC]
                       [EXAMPLES] Examples: "results.json", "output/analysis_results.json", "data/processed_output.json" [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 300. [/BRIEF]
                [DETAILED] The maximum time in seconds the subprocess is allowed to run before
                being terminated. This prevents infinite loops and runaway processes from
                consuming system resources. If the execution exceeds this limit, a timeout
                error will be returned. Choose appropriate values based on expected computation time. [/DETAILED]
                [SYNTACTIC] Format: positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] Examples: 30 (quick calculations), 300 (standard), 1800 (long processing) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with detailed execution results and captured output. [/BRIEF]
             [DETAILED] A comprehensive JSON string containing execution status, standard output,
             error messages, return code, captured execution results, and file save status.
             The execution_result field contains variables captured from the executed code.
             This enables full visibility into the execution process and results. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "execution_result": {"result": 10}, "stdout": "...", "stderr": "", "return_code": 0}' [/EXAMPLES]

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
    import json
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    try:
        # Ensure timeout is valid
        timeout = safe_convert_timeout(timeout)

        # Create directory if needed
        if save_output_to:
            ensure_directory_exists(save_output_to)

        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Prepare the code with input data if provided
            full_code = ""
            if input_data:
                full_code += "import json\n"
                full_code += f"input_data = json.loads({input_data!r})\n"

            full_code += python_code
            full_code += generate_output_capture_code()

            f.write(full_code)
            temp_file = f.name

        # Execute the code
        process = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
            check=False,
        )

        # Clean up
        Path(temp_file).unlink()

        # Parse output
        execution_result, output_lines = parse_execution_output(process.stdout)

        # Save output if requested
        saved_path = None
        if save_output_to and execution_result:
            try:
                with Path(save_output_to).open("w") as f:
                    json.dump(execution_result, f, indent=2)
                saved_path = save_output_to
            except Exception as e:
                logger.warning(
                    f"Warning: Failed to save output to {save_output_to}: {e}"
                )

        result = {
            "success": process.returncode == 0,
            "stdout": "\n".join(output_lines),
            "stderr": process.stderr,
            "return_code": process.returncode,
            "execution_result": execution_result,
            "saved_to": saved_path,
        }
        if process.returncode != 0:
            result["error"] = process.stderr or "Script execution failed"

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        from pathlib import Path

        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )
    except Exception as e:
        if "temp_file" in locals() and Path(temp_file).exists():
            Path(temp_file).unlink()
        return json.dumps(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "execution_result": {},
            }
        )


@tool
def execute_python_script(
    script_path: str,
    args: list | None = None,
    timeout: int = 600,
    working_dir: str | None = None,
) -> str:
    """[BRIEF] Execute a Python script file with arguments in a controlled environment. [/BRIEF]

    [DETAILED] This tool executes existing Python script files with command-line arguments, providing
    a controlled environment for running complex analysis workflows, data processing pipelines, or
    computational simulations. It captures all output streams and provides comprehensive execution
    monitoring with timeout protection. This is essential for integrating existing Python scripts
    into automated workflows and materials analysis pipelines. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to execute existing Python scripts with specific arguments. You can also use io tool to write a script and then execute it.
    - Best suited for running complex analysis workflows or simulations
    - Essential for integrating external Python tools into automated pipelines
    - Recommended for batch processing and computational workflows
    - Avoid for simple code execution (use execute_python_code instead)
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
                    The script must exist and be readable. The path can be relative to the
                    current working directory or absolute. The script should be a valid Python
                    file with appropriate shebang or run using the Python interpreter. [/DETAILED]
                    [SYNTACTIC] Format: "Valid file path to Python script" [/SYNTACTIC]
                    [EXAMPLES] Examples: "scripts/analysis.py", "/home/user/simulations/run_sim.py", "data_processing.py" [/EXAMPLES]
        args: [BRIEF] Optional list of command-line arguments for the script. [/BRIEF]
             [DETAILED] A list of strings representing command-line arguments to pass to the script.
             These arguments will be passed to the script in the order provided. Common arguments
             include input files, output paths, configuration parameters, and processing options.
             If None, the script will be executed without arguments. [/DETAILED]
             [SYNTACTIC] Format: ["arg1", "arg2", "arg3", ...] or None [/SYNTACTIC]
             [EXAMPLES] Examples: ["--input", "data.json"], ["--verbose", "--output", "results.csv"], None [/EXAMPLES]
        timeout: [BRIEF] Maximum execution time in seconds. Defaults to 600. [/BRIEF]
                [DETAILED] The maximum time in seconds the script is allowed to run before being
                terminated. This prevents runaway processes and ensures resource management.
                Choose appropriate values based on expected script execution time. For
                computational simulations, longer timeouts may be necessary. [/DETAILED]
                [SYNTACTIC] Format: positive integer representing seconds [/SYNTACTIC]
                [EXAMPLES] Examples: 300 (5 minutes), 600 (10 minutes), 3600 (1 hour) [/EXAMPLES]
        working_dir: [BRIEF] Optional working directory for script execution. [/BRIEF]
                    [DETAILED] The directory from which the script should be executed. This affects
                    relative path resolution and file I/O operations within the script. If None,
                    the current working directory will be used. This is useful when scripts
                    expect to run from specific directories or access relative files. [/DETAILED]
                    [SYNTACTIC] Format: "Valid directory path or None" [/SYNTACTIC]
                    [EXAMPLES] Examples: "/path/to/project", "data/analysis", None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive execution results and monitoring data. [/BRIEF]
             [DETAILED] A JSON-formatted string containing execution status, captured output streams,
             error messages, return code, and the complete command that was executed. This provides
             full visibility into the script execution process and enables debugging and monitoring
             of automated workflows. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "stdout": "Processing complete", "stderr": "", "return_code": 0, "command": "python script.py --input data.json"}' [/EXAMPLES]

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
        if not Path(script_path).exists():
            return json.dumps(
                {"success": False, "error": f"Script file not found: {script_path}"}
            )

        # Prepare command
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(str(arg) for arg in args)

        # Execute
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir or Path.cwd(),
            check=False,
        )

        result = {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "return_code": process.returncode,
            "command": " ".join(cmd),
        }

        return json.dumps(result, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": f"Script execution timed out after {timeout} seconds",
            }
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def filter_json_with_strategy(
    input_json_path: str,
    output_json_path: str,
    custom_code: str | None = None,
) -> str:
    """[BRIEF] Filter JSON data using custom Python code and save results to a new file. [/BRIEF]

    [DETAILED] This tool provides flexible JSON data filtering capabilities using custom Python logic,
    essential for data preprocessing, quality control, and custom analysis workflows. It enables
    sophisticated filtering operations that go beyond simple threshold-based selection, allowing
    for complex multi-criteria filtering, data validation, and custom transformations. This is
    crucial for preparing datasets for analysis and machine learning applications. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need custom filtering logic beyond standard threshold-based selection
    - Best suited for complex multi-criteria filtering and data validation
    - Essential for data preprocessing and quality control workflows
    - Recommended for custom data transformations and analysis pipelines
    - Avoid for simple filtering operations that can be done with existing tools
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads JSON data from input file into a 'data' variable
    - Executes custom Python code in an isolated environment
    - Expects filtering logic to produce results in a 'filtered_data' variable
    - Saves filtered results to output file with comprehensive statistics
    - Provides detailed reporting on filtering effectiveness and data reduction
    - Use io tools or python tool to see the  keys from json if required
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure input JSON file exists and custom filtering code is prepared [/PREREQUISITE]
    2. [CURRENT] Apply custom filtering logic to process and filter JSON data [/CURRENT]
    3. [FOLLOW_UP] Use filtered data for further analysis, ML preparation for example use prepare_tabular_dataset to prepare datset from the output file [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `filter_json_with_strategy("input.json", "output.json", "filtered_data = [x for x in data if x['energy'] < 0.5]")`,
    `filter_json_with_strategy("materials.json", "stable.json", "filtered_data = [x for x in data if x['is_stable']]")`,
    `filter_json_with_strategy("polymorphs.json", "filtered.json", "filtered_data = [x for x in data if x['band_gap'] > 1.0 and x['density'] < 5.0]")`,
    [/SYNTACTICAL]

    Args:
        input_json_path: [BRIEF] Path to the input JSON file to be filtered. [/BRIEF]
                        [DETAILED] Complete file path to the JSON file containing the data to be filtered.
                        The file should contain valid JSON data, typically a list of dictionaries
                        representing materials or structures with various properties. The file must
                        be readable and contain well-formed JSON. [/DETAILED]
                        [SYNTACTIC] Format: "Valid file path to JSON file" [/SYNTACTIC]
                        [EXAMPLES] Examples: "data/materials.json", "polymorphs/all_structures.json", "input/dataset.json" [/EXAMPLES]
        output_json_path: [BRIEF] Path where filtered JSON data will be saved. [/BRIEF]
                         [DETAILED] Complete file path where the filtered JSON data will be written.
                         The directory will be created if it doesn't exist. The output file will
                         contain the filtered subset of the input data in the same JSON format.
                         Using .json extension is recommended for clarity. [/DETAILED]
                         [SYNTACTIC] Format: "Valid file path with .json extension" [/SYNTACTIC]
                         [EXAMPLES] Examples: "output/filtered_materials.json", "results/stable_phases.json", "processed/selected_data.json" [/EXAMPLES]
        custom_code: [BRIEF] Python code string defining the filtering logic. [/BRIEF]
                    [DETAILED] A string containing Python code that defines the filtering logic.
                    The code should expect the input data in a variable named 'data' and store
                    the filtered results in a variable named 'filtered_data'. The code can use
                    any Python constructs including list comprehensions, complex conditions,
                    and data transformations. [/DETAILED]
                    [SYNTACTIC] Format: "Valid Python code string with 'data' input and 'filtered_data' output" [/SYNTACTIC]
                    [EXAMPLES] Examples: "filtered_data = [x for x in data if x['energy'] < threshold]", "filtered_data = [x for x in data if x.get('stable', False)]" [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with filtering results and comprehensive statistics. [/BRIEF]
             [DETAILED] A JSON-formatted string containing filtering status, original and filtered
             data counts, output file path, and percentage reduction achieved. This provides
             comprehensive information about the filtering operation's effectiveness and enables
             quality control of the data processing pipeline. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "original_count": 100, "filtered_count": 25, "output_path": "filtered.json", "reduction_percentage": 75.0}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When the input JSON file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Input file path is invalid or file is not accessible [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify input file path exists and is readable [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When the input file contains invalid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON in the input file [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and fix any syntax errors [/ERROR_RECOVERY]
        SyntaxError: [ERROR_WHEN] When the custom filtering code contains syntax errors [/ERROR_WHEN]
                    [ERROR_DETAILS] Invalid Python syntax in the custom_code parameter [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Check and fix Python syntax in the filtering code [/ERROR_RECOVERY]
        RuntimeError: [ERROR_WHEN] When the custom filtering code fails during execution [/ERROR_WHEN]
                     [ERROR_DETAILS] Runtime errors in the filtering logic [/ERROR_DETAILS]
                     [ERROR_RECOVERY] Debug filtering code and ensure all variables are properly defined [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Custom code execution is isolated and cannot import external libraries
    - Cannot validate filtered data structure or content
    - Limited error reporting for complex filtering logic
    [/LIMITATIONS]
    """
    try:
        with Path(input_json_path).open("r") as f:
            data = json.load(f)

        # Create execution environment with data available
        exec_globals = {"data": data}
        exec_locals = {}

        # Execute the custom filtering code
        exec(custom_code, exec_globals, exec_locals)

        # Get the filtered data
        if "filtered_data" not in exec_locals:
            return json.dumps(
                {
                    "success": False,
                    "error": "Custom code must define 'filtered_data' variable",
                },
                indent=2,
            )

        filtered_data = exec_locals["filtered_data"]

        # Save filtered data
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_json_path).open("w") as f:
            json.dump(filtered_data, f, indent=2)

        result = {
            "success": True,
            "original_count": len(data),
            "filtered_count": len(filtered_data),
            "output_path": output_json_path,
            "reduction_percentage": (1 - len(filtered_data) / len(data)) * 100
            if data
            else 0,
        }

        return json.dumps(result, indent=2)

    except FileNotFoundError:
        return json.dumps(
            {"success": False, "error": f"Input file not found: {input_json_path}"},
            indent=2,
        )
    except json.JSONDecodeError:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid JSON format in file: {input_json_path}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()},
            indent=2,
        )


"""
Dataset preparation tools for different ML model types.
"""


@tool
def prepare_tabular_dataset(
    polymorphs_json_path: str,
    output_path: str,
    target_property: str = "formation_energy_per_atom",
    feature_engineering: str = "basic",
    test_split: float = 0.2,
    normalize: bool = True,
) -> str:
    """[BRIEF] Prepare tabular dataset for traditional ML models with feature engineering. [/BRIEF]

    [DETAILED] This tool creates ML-ready tabular datasets from materials data with simple feature engineering
    capabilities suitable for traditional machine learning models like XGBoost. It implements multiple feature engineering strategies such as basic property extraction to
    structural descriptors, handles data preprocessing, normalization, and train/test splitting. This is essential
    for building property prediction models. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when preparing data for traditional ML models (XGBoost, Random Forest, etc.)
    - Best suited for structured materials property prediction tasks
    - Essential for creating feature-engineered datasets from raw materials data
    - Recommended for establishing baseline models before deep learning approaches
    - Avoid when working with graph-structured or sequential data
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads materials data (json data) and extracts basic properties (density, volume, composition)
    - Implements advanced feature engineering including structural and electronic properties
    - Handles categorical encoding and missing value imputation automatically
    - Performs data normalization using standard scaling.
    - Creates train/test splits with proper randomization and metadata tracking
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First consolidate materials data using consolidate_polymorph_datasets [/PREREQUISITE]
    2. [CURRENT] Prepare comprehensive tabular dataset with engineered features [/CURRENT]
    3. [FOLLOW_UP] Use train_xgboost_model or other ML training tools with prepared dataset [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `prepare_tabular_dataset("consolidated.json", "ml_data", "formation_energy_per_atom", "advanced", 0.2, True)`,
    `prepare_tabular_dataset("polymorphs.json", "datasets", "band_gap", "basic", 0.15, False)`,
    `prepare_tabular_dataset("materials.json", "output", "bulk_modulus", "custom", 0.25, True)`,
    [/SYNTACTICAL]

    Args:
        polymorphs_json_path: [BRIEF] Path to consolidated polymorphs JSON file. [/BRIEF]
                             [DETAILED] Complete file path to a JSON file containing consolidated polymorph data with
                             materials properties and crystal structures. This should be the output from
                             consolidate_polymorph_datasets or similar tools containing comprehensive materials
                             information including CIF structures and calculated properties. [/DETAILED]
                             [SYNTACTIC] Format: "Valid file path to JSON file with materials data" [/SYNTACTIC]
                             [EXAMPLES] Examples: "consolidated_polymorphs.json", "data/materials_database.json", "datasets/all_oxides.json" [/EXAMPLES]

        output_path: [BRIEF] Base path for saving dataset files. [/BRIEF]
                    [DETAILED] Base directory and filename prefix where the prepared dataset files will be saved.
                    Multiple files will be created including training data, test data, and metadata. The tool
                    will create the directory structure if it doesn't exist. [/DETAILED]
                    [SYNTACTIC] Format: "Valid directory path and filename prefix" [/SYNTACTIC]
                    [EXAMPLES] Examples: "ml_datasets/formation_energy", "data/tabular", "output/materials_ml" [/EXAMPLES]

        target_property: [BRIEF] Property to predict. Defaults to "formation_energy_per_atom". [/BRIEF]
                        [DETAILED] The materials property that will serve as the prediction target for machine learning
                        models. This should be a key present in the polymorphs data with numerical values. Common
                        targets include formation energy, band gap, bulk modulus, and other calculated properties. [/DETAILED]
                        [SYNTACTIC] Format: "String matching property key in JSON data" [/SYNTACTIC]
                        [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "bulk_modulus", "density" [/EXAMPLES]

        feature_engineering: [BRIEF] Feature engineering strategy. Defaults to "basic". [/BRIEF]
                            [DETAILED] The level of feature engineering to apply to the materials data. "basic" extracts
                            fundamental properties, "advanced" includes structural and electronic descriptors, and "custom"
                            applies specialized feature extraction.[/DETAILED]
                            [SYNTACTIC] Format: "basic", "advanced", or "custom" [/SYNTACTIC]
                            [EXAMPLES] Examples: "basic", "advanced" , "custom" [/EXAMPLES]

        test_split: [BRIEF] Fraction of data for test set. Defaults to 0.2. [/BRIEF]
                   [DETAILED] The proportion of the dataset to reserve for testing, expressed as a decimal fraction.
                   The remaining data will be used for training. Common values range from 0.1 to 0.3 depending on
                   dataset size and validation strategy. Larger test sets provide more reliable evaluation but reduce
                   training data. [/DETAILED]
                   [SYNTACTIC] Format: float between 0.0 and 1.0 [/SYNTACTIC]
                   [EXAMPLES] Examples: 0.1 (small test set), 0.2 (standard), 0.3 (large test set) [/EXAMPLES]

        normalize: [BRIEF] Whether to normalize features. Defaults to True. [/BRIEF]
                  [DETAILED] Boolean flag controlling whether features should be normalized using standard scaling
                  (zero mean, unit variance). Normalization is generally recommended for most ML algorithms as it
                  ensures features have similar scales and prevents any single feature from dominating the model. [/DETAILED]
                  [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                  [EXAMPLES] Examples: True (recommended), False (when features already normalized) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive dataset preparation results and file paths. [/BRIEF]
             [DETAILED] A detailed JSON-formatted string containing preparation success status, file paths for training
             and test data, normalization parameters, dataset statistics, feature information, and metadata. This
             provides complete information about the prepared dataset for subsequent ML workflows. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "train_path": "ml_data_train.csv", "test_path": "ml_data_test.csv", "dataset_info": {...}}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When polymorphs JSON file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Invalid file path or missing input data file [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify file path and ensure input data file exists [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When target property is not found in the data [/ERROR_WHEN]
                 [ERROR_DETAILS] Specified target property doesn't exist in materials data [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check available properties in data and use valid target property name [/ERROR_RECOVERY]
        ValueError: [ERROR_WHEN] When feature engineering fails or data format is invalid [/ERROR_WHEN]
                   [ERROR_DETAILS] Structural data cannot be processed or insufficient valid samples [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Check data format and ensure structures are valid for feature extraction [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Advanced feature engineering requires valid crystal structure data
    - Processing time scales with dataset size and feature engineering complexity
    - Some features may not be meaningful for all material types
    - Cannot handle missing structural data gracefully in advanced mode
    [/LIMITATIONS]
    """
    import json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load polymorphs data
        with Path(polymorphs_json_path).open("r") as f:
            polymorphs = json.load(f)

        if not isinstance(polymorphs, list):
            return json.dumps(
                {"success": False, "error": "Input must be list of polymorphs"}
            )

        # Extract features and targets
        features_list = []
        targets = []
        metadata = []

        for poly in polymorphs:
            if target_property not in poly or poly[target_property] is None:
                continue

            # Basic features
            features = {
                "density": poly.get("density", 0),
                "volume": poly.get("volume", 0),
                "nsites": poly.get("nsites", 0),
                "band_gap": poly.get("band_gap", 0),
                "energy_above_hull": poly.get("energy_above_hull", 0),
            }

            # Advanced features from crystal structure
            if feature_engineering in ["advanced", "custom"]:
                try:
                    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
                    from pymatgen.core import Structure

                    structure = Structure.from_str(poly["cif"], fmt="cif")

                    # Structural features
                    features.update(
                        {
                            "lattice_a": structure.lattice.a,
                            "lattice_b": structure.lattice.b,
                            "lattice_c": structure.lattice.c,
                            "lattice_alpha": structure.lattice.alpha,
                            "lattice_beta": structure.lattice.beta,
                            "lattice_gamma": structure.lattice.gamma,
                            "lattice_volume": structure.lattice.volume,
                            "num_species": len(structure.composition.elements),
                            "packing_efficiency": structure.density
                            / structure.lattice.volume
                            * len(structure),
                        }
                    )

                    # Space group features
                    sg_analyzer = SpacegroupAnalyzer(structure)
                    features["space_group_number"] = (
                        sg_analyzer.get_space_group_number()
                    )
                    features["crystal_system"] = sg_analyzer.get_crystal_system()

                    # Composition features
                    composition = structure.composition
                    features["num_elements"] = len(composition.elements)
                    features["electronegativity_diff"] = (
                        max([e.X for e in composition.elements])
                        - min([e.X for e in composition.elements])
                        if len(composition.elements) > 1
                        else 0
                    )

                except Exception:
                    logger.info(
                        f"Warning: Could not extract advanced features for {poly.get('material_id', 'unknown')}"
                    )
                    continue

            features_list.append(features)
            targets.append(poly[target_property])
            metadata.append(
                {
                    "material_id": poly.get("material_id", "unknown"),
                    "composition": poly.get("composition", "unknown"),
                    "space_group": poly.get("space_group", "unknown"),
                }
            )

        if len(features_list) == 0:
            return json.dumps({"success": False, "error": "No valid samples found"})

        # Convert to DataFrame
        df_features = pd.DataFrame(features_list)
        df_targets = pd.Series(targets)
        df_metadata = pd.DataFrame(metadata)

        # Handle categorical features
        categorical_columns = (
            ["crystal_system"] if feature_engineering in ["advanced", "custom"] else []
        )
        for col in categorical_columns:
            if col in df_features.columns:
                df_features[col] = pd.Categorical(df_features[col]).codes

        # Handle missing values
        df_features = df_features.fillna(df_features.mean())

        # Split data
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(df_features))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_split, random_state=42
        )

        X_train, X_test = df_features.iloc[train_idx], df_features.iloc[test_idx]
        y_train, y_test = df_targets.iloc[train_idx], df_targets.iloc[test_idx]
        metadata_train, metadata_test = (
            df_metadata.iloc[train_idx],
            df_metadata.iloc[test_idx],
        )

        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), columns=X_test.columns, index=X_test.index
            )

            # Save scaler
            scaler_path = output_dir / "scaler.pkl"
            with scaler_path.open("wb") as f:
                pickle.dump(scaler, f)
            scaler_path = str(scaler_path)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
            scaler_path = None

        # Save datasets
        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"

        # Combine features and targets for saving
        train_data = X_train_scaled.copy()
        train_data[target_property] = y_train
        train_data.to_csv(train_path, index=False)

        test_data = X_test_scaled.copy()
        test_data[target_property] = y_test
        test_data.to_csv(test_path, index=False)

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        dataset_info = {
            "target_property": target_property,
            "feature_engineering": feature_engineering,
            "normalize": normalize,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(df_features.columns),
            "feature_count": len(df_features.columns),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "scaler_path": scaler_path,
            "train_metadata": metadata_train.to_dict("records"),
            "test_metadata": metadata_test.to_dict("records"),
        }

        with metadata_path.open("w") as f:
            json.dump(dataset_info, f, indent=2)

        return json.dumps(
            {
                "success": True,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "metadata_path": str(metadata_path),
                "scaler_path": scaler_path,
                "dataset_info": dataset_info,
            },
            indent=2,
        )

    except Exception as e:
        import traceback

        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@tool
def get_mp_thermo_data(material_id: str) -> str:
    """[BRIEF] Retrieve comprehensive thermodynamic data for materials from Materials Project database. [/BRIEF]

    [DETAILED] This tool accesses detailed thermodynamic information from the Materials Project database, providing
    essential data for understanding material stability, phase relationships, and thermodynamic properties. It
    retrieves formation energies, energy above hull, decomposition pathways, and stability information crucial
    for materials design and selection. This thermodynamic data enables informed decisions about material
    synthesis feasibility and provides benchmarks for computational studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need comprehensive thermodynamic data for specific materials
    - Recommended if you need to retrieve more thermodynaic infprmation of a structure
    - Avoid when you only need basic structural or electronic properties
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Connects to Materials Project thermodynamics database via API
    - Retrieves calculated formation energies and stability information
    - Provides energy above hull data for phase stability assessment
    - Reports decomposition pathways and competing phases
    - Returns comprehensive thermodynamic dataset with proper energy corrections
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure MP_API_KEY is set and material ID is valid [/PREREQUISITE]
    2. [CURRENT] Retrieve comprehensive thermodynamic data for target material [/CURRENT]
    3. [FOLLOW_UP] Use thermodynamic data for stability analysis or phase diagram studies [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `get_mp_thermo_data("mp-149")   # Silicon thermodynamic data`,
    `get_mp_thermo_data("mp-2657")  # TiO2 thermodynamic properties`,
    `get_mp_thermo_data("mp-1143")  # Al2O3 stability information`,
    [/SYNTACTICAL]

    Args:
        material_id: [BRIEF] Materials Project ID for the target material. [/BRIEF]
                    [DETAILED] The unique Materials Project identifier for the material of interest.
                    Should be in the format "mp-XXXXX" where XXXXX is the numerical ID. The material
                    must exist in the Materials Project database and have thermodynamic calculations
                    available. [/DETAILED]
                    [SYNTACTIC] Format: "mp-" followed by digits (e.g., "mp-149", "mp-2657") [/SYNTACTIC]
                    [EXAMPLES] Examples: "mp-149" (Silicon), "mp-2657" (TiO2), "mp-1143" (Al2O3) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string containing comprehensive thermodynamic data and stability information. [/BRIEF]
             [DETAILED] A JSON-formatted string containing thermodynamic properties including material ID,
             thermodynamic functional used, formation energy per atom, energy above hull, decomposition
             products, stability status, energy type, and uncorrected energies. Returns error information
             if thermodynamic data is not available. [/DETAILED]
             [EXAMPLES] Example output: '[{"material_id": "mp-149", "formation_energy_per_atom": -4.2, "energy_above_hull": 0.0, "is_stable": true, ...}]' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When Materials Project API key is not available [/ERROR_WHEN]
                   [ERROR_DETAILS] MP_API_KEY environment variable not set or invalid [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Obtain valid API key from Materials Project and set environment variable [/ERROR_RECOVERY]
        ConnectionError: [ERROR_WHEN] When unable to connect to Materials Project API [/ERROR_WHEN]
                        [ERROR_DETAILS] Network connectivity issues or API server problems [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check internet connection and try again later [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When material ID is not found or has no thermodynamic data [/ERROR_WHEN]
                 [ERROR_DETAILS] Invalid material ID or thermodynamic properties not calculated [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Verify material ID exists and has thermodynamic calculations [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to materials with calculated thermodynamic properties in Materials Project
    - Thermodynamic accuracy depends on computational methodology and corrections applied
    - May not include experimental thermodynamic data or recent calculations
    - Cannot provide thermodynamic data for custom or modified compositions
    [/LIMITATIONS]
    """
    from mp_api.client import MPRester

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Get thermodynamic data
        thermo_docs = mpr.thermo.search(
            material_ids=[material_id],
            fields=[
                "material_id",
                "thermo_type",
                "formation_energy_per_atom",
                "energy_above_hull",
                "decomposes_to",
                "is_stable",
                "energy_type",
                "uncorrected_energy_per_atom",
            ],
        )

        if not thermo_docs:
            return json.dumps(
                {"error": f"No thermodynamic data found for {material_id}"}
            )

        thermo_data = [
            {
                "material_id": doc.material_id,
                "thermo_type": str(doc.thermo_type),
                "formation_energy_per_atom": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "decomposes_to": [
                    {
                        "material_id": d.material_id,
                        "formula": getattr(d, "formula", ""),
                        "amount": d.amount,
                    }
                    for d in (doc.decomposes_to or [])
                ],
                "is_stable": doc.is_stable,
                "energy_type": doc.energy_type,
                "uncorrected_energy_per_atom": doc.uncorrected_energy_per_atom,
            }
            for doc in thermo_docs
        ]

        return json.dumps(thermo_data, indent=2)


"""
## NL training models
"""


@tool
def train_xgboost_model(
    train_data_path: str,
    test_data_path: str,
    model_save_path: str,
    target_column: str = "formation_energy_per_atom",
    hyperparameters: dict | None = None,
) -> str:
    """[BRIEF] Train XGBoost regression model for property prediction with evaluation. [/BRIEF]

    [DETAILED] This tool implements comprehensive XGBoost model training for materials property prediction,
    including hyperparameter management, model evaluation. The tool provides complete training pipeline with automatic
    evaluation metrics and save the model to the give path. The model takes as input csv file for train and test dataset [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need regression models for materials property prediction
    - Best suited for structured/tabular materials data with engineered features
    - We can vary the hyperparamters to optimize performance metrics
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads training and test data from CSV files with proper feature/target separation
    - Applies XGBoost regression with hyperparameters
    - Performs training with automatic validation and metric calculation
    - Generates comprehensive evaluation including MAE, RMSE, R2, and feature importance
    - Saves trained model and detailed results for future use and analysis
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First prepare tabular dataset using prepare_tabular_dataset [/PREREQUISITE]
    2. [CURRENT] Train XGBoost model with optimized hyperparameters [/CURRENT]
    3. [FOLLOW_UP] Use evaluate_xgboost_model for detailed analysis or model for predictions [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `train_xgboost_model("train.csv", "test.csv", "model.pkl", "formation_energy_per_atom")`,
    `train_xgboost_model("train.csv", "test.csv", "model.pkl", "band_gap", {"n_estimators": 200})`,
    `train_xgboost_model("data/train.csv", "data/test.csv", "models/xgb_model.pkl", "energy")`,
    [/SYNTACTICAL]

    Args:
        train_data_path: [BRIEF] Path to training data CSV file. [/BRIEF]
                        [DETAILED] Complete file path to the CSV file containing training data with
                        features and target column. The file should have a header row with column names
                        and be properly formatted with numerical features. This is typically output
                        from prepare_tabular_dataset tool. [/DETAILED]
                        [SYNTACTIC] Format: "Valid file path to CSV file with header" [/SYNTACTIC]
                        [EXAMPLES] Examples: "data/train.csv", "datasets/materials_train.csv", "ml_data/train_features.csv" [/EXAMPLES]
        test_data_path: [BRIEF] Path to test data CSV file. [/BRIEF]
                       [DETAILED] Complete file path to the CSV file containing test data with the same
                       structure as training data. Used for independent model evaluation and performance
                       assessment. Should have identical column structure to training data. [/DETAILED]
                       [SYNTACTIC] Format: "Valid file path to CSV file with header" [/SYNTACTIC]
                       [EXAMPLES] Examples: "data/test.csv", "datasets/materials_test.csv", "ml_data/test_features.csv" [/EXAMPLES]
        model_save_path: [BRIEF] Path to save the trained model file. [/BRIEF]
                        [DETAILED] Complete file path where the trained XGBoost model will be saved using
                        joblib serialization. The model can be loaded later for predictions or further
                        analysis. Using .pkl extension is recommended for clarity. [/DETAILED]
                        [SYNTACTIC] Format: "Valid file path with .pkl extension" [/SYNTACTIC]
                        [EXAMPLES] Examples: "models/xgb_model.pkl", "trained_models/formation_energy_model.pkl", "results/model.pkl" [/EXAMPLES]
        target_column: [BRIEF] Name of the target column for prediction. Defaults to "formation_energy_per_atom". [/BRIEF]
                      [DETAILED] The column name in the CSV files that contains the target values to predict.
                      This column will be separated from features during training. Common targets include
                      formation energy, band gap, bulk modulus, and other materials properties. [/DETAILED]
                      [SYNTACTIC] Format: "String matching column name in CSV files" [/SYNTACTIC]
                      [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "bulk_modulus", "density" [/EXAMPLES]
        hyperparameters: [BRIEF] Optional dictionary of XGBoost hyperparameters. [/BRIEF]
                        [DETAILED] Dictionary containing XGBoost hyperparameters to override default values.
                        Can include parameters like n_estimators, max_depth, learning_rate, subsample, etc.
                        If None, optimized default parameters will be used. Proper hyperparameter tuning
                        can significantly improve model performance. [/DETAILED]
                        [SYNTACTIC] Format: '{"param_name": value, ...} or None' [/SYNTACTIC]
                        [EXAMPLES] Examples: {"n_estimators": 200, "max_depth": 8}, {"learning_rate": 0.05}, None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive training results and model performance metrics. [/BRIEF]
             [DETAILED] A detailed JSON-formatted string containing training success status, model performance
             metrics (MAE, RMSE, R2), feature importance rankings, hyperparameters used, dataset information,
             and file paths for saved model and results. This enables comprehensive model evaluation and
             comparison. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "test_metrics": {"mae": 0.12, "rmse": 0.18, "r2": 0.85}, "feature_importance": {...}, "model_path": "model.pkl"}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When training or test data files don't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Invalid file paths or missing CSV files [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify file paths exist and contain properly formatted CSV data [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When target column is not found in the data [/ERROR_WHEN]
                 [ERROR_DETAILS] Specified target column doesn't exist in CSV files [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check column names in CSV files and use valid target column name [/ERROR_RECOVERY]
        ValueError: [ERROR_WHEN] When data contains invalid values or format issues [/ERROR_WHEN]
                   [ERROR_DETAILS] Non-numerical data in features or target, or insufficient data [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Ensure data is properly preprocessed and contains sufficient samples [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires tabular data format (csv) with numerical features
    - Model performance depends on feature engineering quality
    - May not capture complex non-linear relationships as well as deep learning
    - Hyperparameter tuning requires multiple runs
    [/LIMITATIONS]
    """
    try:
        # Load data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        # Separate features and targets
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Default hyperparameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

        if hyperparameters:
            default_params.update(hyperparameters)

        # Train model
        model = xgb.XGBRegressor(**default_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_metrics = {
            "mae": float(mean_absolute_error(y_train, y_pred_train)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "r2": float(r2_score(y_train, y_pred_train)),
        }

        test_metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "r2": float(r2_score(y_test, y_pred_test)),
        }

        # Feature importance - Convert float32 to standard float
        feature_importance = {
            col: float(importance)
            for col, importance in zip(
                X_train.columns, model.feature_importances_, strict=False
            )
        }

        # Save model
        joblib.dump(model, model_save_path)

        # Save predictions
        predictions_path = model_save_path.replace(".pkl", "_predictions.json")
        predictions = {
            "train_predictions": y_pred_train.tolist(),
            "test_predictions": y_pred_test.tolist(),
            "train_targets": y_train.tolist(),
            "test_targets": y_test.tolist(),
        }

        with Path(predictions_path).open("w") as f:
            json.dump(predictions, f, indent=2)

        training_results_path = model_save_path.replace(
            ".pkl", "_training_results.json"
        )
        training_results = {
            "success": True,
            "model_type": "xgboost",
            "model_path": model_save_path,
            "results_path": training_results_path,
            "predictions_path": predictions_path,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "hyperparameters": default_params,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X_train.columns),
        }
        with Path(training_results_path).open("w") as f:
            json.dump(training_results, f, indent=2)

        return json.dumps(training_results, indent=2)

    except Exception as e:
        import traceback

        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@tool
def evaluate_xgboost_model(
    model_path: str,
    test_data_path: str,
    target_column: str = "formation_energy_per_atom",
    detailed_analysis: bool = True,
) -> str:
    """[BRIEF] Evaluate trained XGBoost model with comprehensive performance metrics and detailed analysis. [/BRIEF]

    [DETAILED] This tool provides thorough evaluation of trained XGBoost models with comprehensive metrics
    and detailed analysis capabilities. It generates standard regression metrics (MAE, RMSE, R2, MAPE),
    error analysis (mean error, error standard deviation, max positive/negative errors), prediction
    ranges (min, max, standard deviation), and top feature importance rankings essential for model validation
    and deployment decisions. The tool supports both basic and detailed analysis modes, enabling quick assessments
    or in-depth model understanding for research and production applications. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need comprehensive evaluation of trained XGBoost models
    - Best suited for model validation and performance assessment workflows
    - Essential for comparing different models or hyperparameter configurations
    - Recommended for generating model performance reports and insights
    - Avoid when you only need basic metrics (use simpler evaluation functions)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads trained XGBoost model from serialized file
    - Processes test data with identical structure to training data
    - Calculates comprehensive regression metrics (MAE, RMSE, R2, MAPE)
    - Performs detailed error analysis including prediction ranges and distributions
    - Extracts and ranks feature importance for model interpretability
    - Provides statistical analysis of prediction quality and model behavior
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First train model using train_xgboost_model and prepare test data [/PREREQUISITE]
    2. [CURRENT] Apply comprehensive evaluation to assess model performance [/CURRENT]
    3. [FOLLOW_UP] Use results for model comparison, hyperparameter tuning, or deployment decisions [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `evaluate_xgboost_model("trained_model.pkl", "test_data.csv", "formation_energy_per_atom", True)`,
    `evaluate_xgboost_model("models/xgb_model.pkl", "data/test.csv", "band_gap", False)`,
    `evaluate_xgboost_model("model.pkl", "test.csv", "energy", True)`,
    [/SYNTACTICAL]

    Args:
        model_path: [BRIEF] Path to the saved XGBoost model file. [/BRIEF]
                   [DETAILED] Complete file path to the serialized XGBoost model created by train_xgboost_model
                   or similar training functions. The model should be saved using joblib or pickle and contain
                   a trained XGBoost regressor ready for evaluation. The file must be readable and contain
                   a valid model object. [/DETAILED]
                   [SYNTACTIC] Format: "Valid file path to .pkl model file" [/SYNTACTIC]
                   [EXAMPLES] Examples: "models/xgb_model.pkl", "trained_models/formation_energy_model.pkl", "model.pkl" [/EXAMPLES]
        test_data_path: [BRIEF] Path to test data CSV file with same structure as training data. [/BRIEF]
                       [DETAILED] Complete file path to CSV file containing test data with identical column
                       structure to the training data used for model creation. Must include both feature
                       columns and the target column for evaluation. The data should be preprocessed
                       consistently with the training data. [/DETAILED]
                       [SYNTACTIC] Format: "Valid file path to CSV file with header" [/SYNTACTIC]
                       [EXAMPLES] Examples: "data/test.csv", "datasets/materials_test.csv", "evaluation/test_data.csv" [/EXAMPLES]
        target_column: [BRIEF] Name of the target column for evaluation. Defaults to "formation_energy_per_atom". [/BRIEF]
                      [DETAILED] The column name in the test CSV that contains the true values for comparison
                      with model predictions. This should match the target column used during training.
                      Common targets include formation energy, band gap, and other materials properties. [/DETAILED]
                      [SYNTACTIC] Format: "String matching column name in test CSV" [/SYNTACTIC]
                      [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "bulk_modulus", "density" [/EXAMPLES]
        detailed_analysis: [BRIEF] Whether to include detailed analysis and feature importance. Defaults to True. [/BRIEF]
                          [DETAILED] Boolean flag controlling the depth of analysis performed. When True, includes
                          prediction ranges (min, max, std), error analysis (mean error, error std, max errors),
                          and top 10 feature importance rankings. When False, provides only basic metrics
                          (MAE, RMSE, R2, MAPE) for quick assessment. Detailed analysis is recommended for thorough
                          model evaluation and interpretation. [/DETAILED]
                          [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                          [EXAMPLES] Examples: True (comprehensive analysis), False (basic metrics only) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive evaluation metrics and analysis results. [/BRIEF]
             [DETAILED] A detailed JSON-formatted string containing evaluation success status, comprehensive
             performance metrics (MAE, RMSE, R2, MAPE), prediction statistics (min/max/std), error analysis
             (mean error, error std, max positive/negative errors), and top 10 feature importance rankings
             when detailed analysis is enabled. This provides complete model assessment
             for validation and comparison purposes. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "evaluation_metrics": {"mae": 0.15, "rmse": 0.22, "r2": 0.83, "mape": 3.45, "feature_importance": {...}}}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When model file or test data file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Invalid file paths or missing files [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify file paths exist and are accessible [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When target column is not found in test data [/ERROR_WHEN]
                 [ERROR_DETAILS] Specified target column doesn't exist in CSV [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check column names in test data and use valid target column [/ERROR_RECOVERY]
        ValueError: [ERROR_WHEN] When model and data are incompatible or contain invalid values [/ERROR_WHEN]
                   [ERROR_DETAILS] Feature mismatch between model and data, or non-numerical values [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Ensure test data has same features as training data and proper preprocessing [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires identical feature structure between training and test data
    - Cannot evaluate model performance on different target variables
    - Feature importance interpretation depends on training data characteristics
    - May not capture model performance on out-of-distribution data
    - MAPE metric becomes unreliable when target values are close to zero
    [/LIMITATIONS]
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    try:
        # Load model and test data
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_data_path)

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Make predictions
        y_pred = model.predict(X_test)

        # Basic metrics
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
            "test_samples": len(y_test),
        }

        if detailed_analysis:
            # Prediction ranges
            metrics["prediction_range"] = {
                "min": float(y_pred.min()),
                "max": float(y_pred.max()),
                "std": float(y_pred.std()),
            }

            # Error analysis
            errors = y_test - y_pred
            metrics["error_analysis"] = {
                "mean_error": float(errors.mean()),
                "error_std": float(errors.std()),
                "max_positive_error": float(errors.max()),
                "max_negative_error": float(errors.min()),
            }

            # Feature importance
            if hasattr(model, "feature_importances_"):
                feature_importance = dict(
                    zip(X_test.columns, model.feature_importances_, strict=False)
                )
                # Convert numpy float32 to Python float
                metrics["feature_importance"] = {
                    k: float(v)
                    for k, v in sorted(
                        feature_importance.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                }

        return json.dumps({"success": True, "evaluation_metrics": metrics}, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def perform_cross_validation(
    train_data_path: str,
    target_column: str = "formation_energy_per_atom",
    cv_folds: int = 5,
    hyperparameters: dict | None = None,
) -> str:
    """[BRIEF] Perform k-fold cross-validation to assess model stability and generalization performance. [/BRIEF]

    [DETAILED] This tool implements comprehensive k-fold cross-validation for XGBoost models to assess
    model  generalization capability, and robustness across different data splits. Cross-validation
    provides more reliable performance estimates than single train/test splits by evaluating model performance
    across multiple data partitions. This is essential for hyperparameter tuning, model comparison, and
    ensuring reliable performance estimates for materials property prediction models. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need robust estimates of model performance and stability
    - Best suited for hyperparameter tuning and model comparison workflows
    - Essential for assessing model generalization before final deployment
    - Recommended for small to medium datasets where train/test splits may be unreliable
    - Avoid for very large datasets where computational cost becomes prohibitive
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Implements k-fold cross-validation with stratified data splitting
    - Trains XGBoost models on k-1 folds and evaluates on the held-out fold
    - Calculates performance metrics (R2, MAE) across all folds
    - Provides statistical analysis including mean performance and variance
    - Uses consistent hyperparameters across all folds for fair comparison
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First prepare training data using prepare_tabular_dataset and train a model using train_xgboost_model [/PREREQUISITE]
    2. [CURRENT] Apply cross-validation to assess model stability and performance [/CURRENT]
    3. [FOLLOW_UP] Use results for hyperparameter optimization or proceed to train_xgboost_model [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `perform_cross_validation("train_data.csv", "formation_energy_per_atom", 5, None)`,
    `perform_cross_validation("data/train.csv", "band_gap", 10, {"n_estimators": 200})`,
    `perform_cross_validation("training.csv", "energy", 3, {"max_depth": 8, "learning_rate": 0.05})`,
    [/SYNTACTICAL]

    Args:
        train_data_path: [BRIEF] Path to training data CSV file. [/BRIEF]
                        [DETAILED] Complete file path to CSV file containing training data with features and
                        target column. The file should have a header row and be properly formatted for machine
                        learning. This data will be split into k folds for cross-validation, so it should
                        represent the complete training dataset. [/DETAILED]
                        [SYNTACTIC] Format: "Valid file path to CSV file with header" [/SYNTACTIC]
                        [EXAMPLES] Examples: "data/train.csv", "datasets/materials_train.csv", "ml_data/training_features.csv" [/EXAMPLES]
        target_column: [BRIEF] Name of the target column for prediction. Defaults to "formation_energy_per_atom". [/BRIEF]
                      [DETAILED] The column name in the CSV that contains the target values for prediction.
                      This column will be separated from features during cross-validation. Should match the
                      target used in subsequent training workflows. Common targets include formation energy,
                      band gap, and other materials properties. [/DETAILED]
                      [SYNTACTIC] Format: "String matching column name in CSV file" [/SYNTACTIC]
                      [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "bulk_modulus", "density" [/EXAMPLES]
        cv_folds: [BRIEF] Number of cross-validation folds. Defaults to 5. [/BRIEF]
                 [DETAILED] The number of folds to use for k-fold cross-validation. Higher values provide
                 more robust estimates but increase computational cost. Common choices are 5 or 10 folds.
                 The value should be chosen based on dataset size - smaller datasets benefit from more
                 folds while larger datasets can use fewer folds. [/DETAILED]
                 [SYNTACTIC] Format: positive integer between 2 and dataset_size [/SYNTACTIC]
                 [EXAMPLES] Examples: 5 (standard), 10 (robust), 3 (quick assessment) [/EXAMPLES]
        hyperparameters: [BRIEF] Optional XGBoost hyperparameters for cross-validation. [/BRIEF]
                        [DETAILED] Dictionary containing XGBoost hyperparameters to use across all cross-validation
                        folds. If None, optimized default parameters will be used. Consistent hyperparameters
                        across folds ensure fair comparison and reliable performance estimates. Useful for
                        testing specific hyperparameter configurations. [/DETAILED]
                        [SYNTACTIC] Format: '{"param_name": value, ...} or None' [/SYNTACTIC]
                        [EXAMPLES] Examples: {"n_estimators": 150, "max_depth": 7}, {"learning_rate": 0.05}, None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with comprehensive cross-validation results and statistical analysis. [/BRIEF]
             [DETAILED] A detailed JSON-formatted string containing cross-validation success status,
             individual fold scores, statistical summaries (mean, standard deviation), hyperparameters used,
             and performance stability assessment. This enables comprehensive evaluation of model robustness
             and generalization capability. [/DETAILED]
             [EXAMPLES] Example output: '{"success": true, "cross_validation_results": {"r2_mean": 0.85, "r2_std": 0.03, "mae_mean": 0.12, "mae_std": 0.02, ...}}' [/EXAMPLES]

    [RAISES] Exceptions:
        FileNotFoundError: [ERROR_WHEN] When training data file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Invalid file path or missing CSV file [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Verify file path exists and contains properly formatted CSV data [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When target column is not found in the data [/ERROR_WHEN]
                 [ERROR_DETAILS] Specified target column doesn't exist in CSV file [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Check column names in CSV file and use valid target column name [/ERROR_RECOVERY]
        ValueError: [ERROR_WHEN] When cv_folds is invalid or data contains invalid values [/ERROR_WHEN]
                   [ERROR_DETAILS] cv_folds less than 2, greater than dataset size, or non-numerical data [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Use valid cv_folds value and ensure data is properly preprocessed [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Computational cost scales linearly with number of folds
    - May not be suitable for very large datasets due to memory constraints
    - Assumes data is suitable for random splitting (no temporal or spatial dependencies)
    - Cannot assess performance on truly out-of-distribution data
    [/LIMITATIONS]
    """
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import KFold, cross_val_score

    try:
        # Load data
        train_df = pd.read_csv(train_data_path)
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]

        # Default hyperparameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        if hyperparameters:
            default_params.update(hyperparameters)

        # Create model
        model = xgb.XGBRegressor(**default_params)

        # Cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # R2 scores
        r2_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")

        # MAE scores (note: sklearn returns negative MAE, so we negate)
        mae_scores = -cross_val_score(
            model, X, y, cv=kfold, scoring="neg_mean_absolute_error"
        )

        results = {
            "cv_folds": cv_folds,
            "r2_scores": r2_scores.tolist(),
            "mae_scores": mae_scores.tolist(),
            "r2_mean": float(r2_scores.mean()),
            "r2_std": float(r2_scores.std()),
            "mae_mean": float(mae_scores.mean()),
            "mae_std": float(mae_scores.std()),
            "hyperparameters": default_params,
        }

        return json.dumps(
            {"success": True, "cross_validation_results": results}, indent=2
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def create_ml_tools() -> dict[str, Tool]:
    """Create all available ML tools"""
    return {
        "get_structure_from_mp_text": get_structure_from_mp_text,
        "batch_retrieve_polymorphs": batch_retrieve_polymorphs,
        "get_bulk_polymorphs_data": get_bulk_polymorphs_data,
        "get_bulk_polymorphs_data_to_file": get_bulk_polymorphs_data_to_file,
        "sort_and_get_first_from_json": sort_and_get_first_from_json,
        "select_polymorphs_with_strategy": select_polymorphs_with_strategy,
        "consolidate_polymorph_datasets": consolidate_polymorph_datasets,
        "execute_python_code": execute_python_code,
        "execute_python_script": execute_python_script,
        "filter_json_with_strategy": filter_json_with_strategy,
        "select_polymorphs_with_strategy_to_file": select_polymorphs_with_strategy_to_file,
        "prepare_tabular_dataset": prepare_tabular_dataset,
        "train_xgboost_model": train_xgboost_model,
        "evaluate_xgboost_model": evaluate_xgboost_model,
        "perform_cross_validation": perform_cross_validation,
    }
