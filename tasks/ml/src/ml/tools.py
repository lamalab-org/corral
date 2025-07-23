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


@tool
def create_slab_from_structure_text(
    structure_cif: str,
    miller_index: tuple = (1, 1, 1),
    min_slab_size: int = 12,
    min_vacuum_size: int = 5,
    primitive: bool = True,
) -> str:
    """[BRIEF] Create a surface slab from a bulk crystal structure with specified Miller indices and dimensions. [/BRIEF]

    [DETAILED] This tool generates a surface slab by cleaving a bulk crystal structure along a specified
    crystallographic plane. It creates a two-dimensional periodic surface model suitable for surface
    chemistry calculations, catalysis studies, and adsorption analysis. The tool automatically handles
    the creation of vacuum space above the surface and ensures proper termination of the crystal structure.
    This is essential for computational surface science studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to create a single slab from a bulk structure with known Miller indices
    - Best suited for straightforward surface generation without need for multiple terminations
    - Recommended when you have specific requirements for slab thickness and vacuum spacing
    - Avoid when you need to explore multiple possible surface terminations (use enumerate_slabs_text instead)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the input CIF structure to create a pymatgen Structure object
    - Uses SlabGenerator to cleave the structure along specified Miller indices
    - Creates a slab with the specified minimum thickness and vacuum spacing
    - Reorients the slab to have the surface normal along the c-axis
    - Sorts atomic positions for consistent structure representation
    - Converts the final slab structure back to CIF format
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain bulk structure using get_structure_from_mp_text or using other tools that return single struucture CIF [/PREREQUISITE]
    2. [CURRENT] Apply this tool to create slab from bulk structure [/CURRENT]
    3. [FOLLOW_UP] Use output with adsorption site tools like get_adsorption_sites_text [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [ create_slab_from_structure_text(cif_string, (1,1,1), 12, 5, True), # Create slab with (1,1,1) Miller indices
    create_slab_from_structure_text(cif_string, (1,0,0), 15, 10, False), # Create slab with (1,0,0) Miller indices
    create_slab_from_structure_text(cif_string)  # Uses defaults (1,1,1), 12, 5, True
    ]
    [/SYNTACTICAL]

    Args:
        structure_cif: [BRIEF] CIF content string of the bulk crystal structure. [/BRIEF]
                      [DETAILED] A properly formatted CIF string containing the bulk crystal structure
                      data including lattice parameters, atomic positions, and space group information.
                      This structure will be cleaved to create the surface. [/DETAILED]
                      [SYNTACTIC] Format: "Valid CIF format string with atomic coordinates and lattice parameters" [/SYNTACTIC]
                      [EXAMPLES] Examples: CIF string from get_structure_from_mp_text output [/EXAMPLES]
        miller_index: [BRIEF] Miller indices for the surface plane. Defaults to (1,1,1). [/BRIEF]
                     [DETAILED] A tuple of three integers specifying the crystallographic plane along
                     which the structure will be cleaved. These indices define the surface orientation
                     and determine the atomic arrangement at the surface. Common choices include (1,1,1),
                     (1,0,0), and (1,1,0) for different surface orientations. [/DETAILED]
                     [SYNTACTIC] Format: tuple of three integers (h, k, l) [/SYNTACTIC]
                     [EXAMPLES] Examples: (1,1,1), (1,0,0), (1,1,0) [/EXAMPLES]
        min_slab_size: [BRIEF] Minimum slab thickness in Angstroms. Defaults to 12. [/BRIEF]
                      [DETAILED] The minimum thickness of the slab in the direction perpendicular to
                      the surface plane. This parameter ensures that the slab has sufficient bulk-like
                      character in the center while exposing the desired surface. Larger values provide
                      more accurate representation of bulk properties but increase computational cost. [/DETAILED]
                      [SYNTACTIC] Format: positive integer representing thickness in Angstroms [/SYNTACTIC]
                      [EXAMPLES] Examples: 12, 15, 8[/EXAMPLES]
        min_vacuum_size: [BRIEF] Minimum vacuum spacing in Angstroms. Defaults to 5. [/BRIEF]
                        [DETAILED] The minimum vacuum space above the surface to prevent interactions
                        between periodic images in surface calculations. This parameter is crucial for
                        accurate surface energy calculations and adsorption studies. Larger values
                        reduce spurious interactions but increase computational requirements. [/DETAILED]
                        [SYNTACTIC] Format: positive integer representing vacuum thickness in Angstroms [/SYNTACTIC]
                        [EXAMPLES] Examples: 5 (minimal), 10 (standard), 15 (large) [/EXAMPLES]
        primitive: [BRIEF] Whether to create a primitive cell slab. Defaults to True. [/BRIEF]
                  [DETAILED] Controls whether to use the primitive cell or conventional cell for
                  slab generation. Primitive cells have the minimum number of atoms while maintaining
                  the essential symmetry, leading to smaller, more efficient computational models.
                  Setting to False uses the conventional cell which may be larger but more intuitive. [/DETAILED]
                  [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                  [EXAMPLES] Examples: True, False [/EXAMPLES]

    Returns:
        str: [BRIEF] CIF content string of the generated surface slab. [/BRIEF]
             [DETAILED] A CIF-formatted string containing the surface slab structure with the
             specified Miller indices, thickness, and vacuum spacing. The structure is oriented
             with the surface normal along the c-axis and includes all necessary crystallographic
             information for surface calculations. [/DETAILED]
             [EXAMPLES] Example output: CIF string with slab structure having surface atoms and vacuum region [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When CIF string is malformed or Miller indices are invalid [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format, zero Miller indices, or incompatible surface [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify CIF format and choose valid Miller indices for the crystal system [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When slab generation fails due to structural issues [/ERROR_WHEN]
                       [ERROR_DETAILS] Insufficient slab thickness or problematic surface termination [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Increase min_slab_size or try different Miller indices [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - May not handle complex surface reconstructions or relaxations
    - Does not optimize atomic positions
    - Limited to simple surface terminations without defects
    - Cannot account for surface segregation or compositional changes
    [/LIMITATIONS]
    """
    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator

    # Load the structure from CIF string
    structure = Structure.from_str(structure_cif, fmt="cif")

    # Create the slab
    slab_gen = SlabGenerator(
        structure, miller_index, min_slab_size, min_vacuum_size, primitive=primitive
    )
    slab = slab_gen.get_slab()
    slab = slab.get_orthogonal_c_slab().get_sorted_structure()

    return slab.to(fmt="cif")


@tool
def enumerate_slabs_text(
    bulk_cif: str,
    miller_index: tuple = (1, 1, 1),
    min_slab_size: float = 12,
    min_vacuum_size: float = 5,
) -> str:
    """[BRIEF] Enumerate all possible surface slab terminations from a bulk structure and return as JSON. [/BRIEF]

    [DETAILED] This tool generates all possible surface terminations for a given bulk crystal structure
    along specified Miller indices. Unlike creating a single slab, this tool explores different ways
    to terminate the surface, which is crucial for materials with complex structures or multiple
    chemically distinct layers. Each termination represents a different surface chemistry and reactivity,
    making this tool essential for comprehensive surface studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to explore all possible surface terminations for a material
    - Best suited for complex materials with multiple distinct atomic layers
    - Recommended for systematic surface studies and comparing different surface chemistries
    - Avoid when you only need a single, well-defined surface (use create_slab_from_structure_text instead)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the bulk CIF structure to create a pymatgen Structure object
    - Uses SlabGenerator to systematically create all possible surface terminations
    - Generates multiple slabs with different atomic arrangements at the surface
    - Applies structural sorting and standardization to each slab
    - Returns all slabs as a JSON dictionary with indexed keys for easy selection
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain bulk structure using get_structure_from_mp_text [/PREREQUISITE]
    2. [CURRENT] Apply this tool to enumerate all possible slab terminations [/CURRENT]
    3. [FOLLOW_UP] Use choose_slab_text to select a specific termination from the results [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [
    `enumerate_slabs_text(cif_string, (1,1,1), 12, 5)`,  # Enumerate slabs with (1,1,1) Miller indices
    `enumerate_slabs_text(cif_string, (1,0,0), 15, 10)`,  # Enumerate slabs with (1,0,0) Miller indices
    `enumerate_slabs_text(cif_string)`  # Uses default parameters (1,1,1), 12, 5
    ]
    [/SYNTACTICAL]

    Args:
        bulk_cif: [BRIEF] Bulk crystal structure in CIF string format. [/BRIEF]
                 [DETAILED] A properly formatted CIF string containing the bulk crystal structure
                 from which surface slabs will be generated. This should be a three-dimensional
                 periodic structure with well-defined atomic positions and lattice parameters.
                 The structure will be analyzed to determine all possible surface terminations. [/DETAILED]
                 [SYNTACTIC] Format: "Valid CIF format string with complete structural information" [/SYNTACTIC]
                 [EXAMPLES] Examples: CIF string from Materials Project structures [/EXAMPLES]
        miller_index: [BRIEF] Miller indices for surface orientation. Defaults to (1,1,1). [/BRIEF]
                     [DETAILED] A tuple of three integers specifying the crystallographic plane along
                     which all surface terminations will be generated. This determines the surface
                     orientation but allows for different terminations along the same plane. Different
                     Miller indices will produce different surface structures and properties. [/DETAILED]
                     [SYNTACTIC] Format: tuple of three integers (h, k, l) [/SYNTACTIC]
                     [EXAMPLES] Examples: (1,1,1), (1,0,0), (1,1,0) [/EXAMPLES]
        min_slab_size: [BRIEF] Minimum slab thickness in Angstroms. Defaults to 12. [/BRIEF]
                      [DETAILED] The minimum thickness of each slab in the direction perpendicular
                      to the surface plane. This ensures that all generated slabs have sufficient
                      bulk-like character while exposing different surface terminations. Affects
                      both the structural accuracy and computational requirements. [/DETAILED]
                      [SYNTACTIC] Format: positive float representing thickness in Angstroms [/SYNTACTIC]
                      [EXAMPLES] Examples: 10.0 (for thin slab), 12.0 (standard), 15.0 (for thick slab) [/EXAMPLES]
        min_vacuum_size: [BRIEF] Minimum vacuum layer thickness in Angstroms. Defaults to 5. [/BRIEF]
                        [DETAILED] The minimum vacuum space above each surface to prevent interactions
                        between periodic images. This parameter is applied to all generated slabs
                        and is crucial for accurate surface calculations. Larger values reduce
                        spurious interactions but increase computational cost. [/DETAILED]
                        [SYNTACTIC] Format: positive float representing vacuum thickness in Angstroms [/SYNTACTIC]
                        [EXAMPLES] Examples: 5.0, 10.0, 15.0 [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string mapping slab indices to their CIF representations. [/BRIEF]
             [DETAILED] A JSON-formatted string containing a dictionary where keys are slab
             identifiers (e.g., "slab_0", "slab_1") and values are the corresponding CIF
             strings for each surface termination. This format allows easy selection and
             comparison of different surface terminations. [/DETAILED]
             [EXAMPLES] Example output: '{"slab_0": "CIF content...", "slab_1": "CIF content...", ...}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When CIF string is malformed or parameters are invalid [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format, negative size parameters, or incompatible Miller indices [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify CIF format and ensure all parameters are positive numbers [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When slab generation fails for the given structure [/ERROR_WHEN]
                       [ERROR_DETAILS] Structure not compatible with specified Miller indices or size constraints [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Try different Miller indices or adjust size parameters [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - This tool might generate really many slabs.
    - Does not perform surface relaxation or optimization
    - May generate many similar terminations for high-symmetry structures
    - Limited to periodic slab models without defects or reconstructions
    [/LIMITATIONS]
    """
    import json

    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator

    bulk_structure = Structure.from_str(bulk_cif, fmt="cif")

    # Create a SlabGenerator from the bulk structure
    slab_gen = SlabGenerator(
        bulk_structure, miller_index, min_slab_size, min_vacuum_size
    )
    slabs = slab_gen.get_slabs()  # returns a list of Slab objects

    slabs_dict = {}
    for i, slab in enumerate(slabs):
        # We use get_orthogonal_c_slab() ensures that the slab lattice is reoriented in c axis for easier adsorption placement.
        # get_sorted_structure() variations in atom ordering that might occur due to how the slab was originally created.
        slab_clean = (
            slab.get_sorted_structure()
            # slab.get_orthogonal_c_slab().get_sorted_structure()
        )  # TODO needs to think about the material science
        slabs_dict[f"slab_{i}"] = slab_clean.to(fmt="cif")

    return json.dumps(slabs_dict, indent=2)


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
                full_code += f"input_data = json.loads('''{input_data}''')\n"

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
        if not Path.exists(script_path):
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


@tool
def save_structures_to_db(
    db_path: str,
    structures_json: str,
    structure_type: str = "slab",  # Options: "bulk", "slab", "adsorbate", "adsorbate_slab"
    table_name: str | None = None,
    additional_properties: dict | None = None,
) -> str:
    """[BRIEF] Save crystal structures to SQLite database with appropriate schema for materials informatics. [/BRIEF]

    [DETAILED] This tool provides comprehensive database storage for crystal structures with specialized
    schemas for different structure types (bulk, slab, adsorbate, adsorbate_slab). It creates appropriate
    tables with optimized column types and relationships for materials informatics workflows. This is
    essential for building persistent materials databases, enabling efficient queries, and supporting
    large-scale computational studies with proper data organization. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need persistent storage for large collections of crystal structures
    - Best suited for building materials databases with queryable metadata
    - Essential for organizing complex computational workflows with multiple structure types
    - Recommended for datasets requiring efficient access and relationship management
    - Avoid for simple file-based storage or small datasets
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Creates SQLite database with optimized schemas for different structure types
    - Parses JSON structure data and maps to appropriate database columns
    - Implements proper data types and constraints for materials properties
    - Handles bulk insertion with conflict resolution and data validation
    - Supports extensible schemas with additional property columns
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First generate structure data using polymorph retrieval or slab generation tools [/PREREQUISITE]
    2. [CURRENT] Save structures to database with appropriate schema [/CURRENT]
    3. [FOLLOW_UP] Use add_descriptor_column_to_db for feature engineering or generate_ml_dataset_format for ML [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `save_structures_to_db("materials.db", polymorphs_json, "bulk", "bulk_structures")`,
    `save_structures_to_db("surfaces.db", slabs_json, "slab", "slab_structures")`,
    `save_structures_to_db("configs.db", adsorbate_configs_json, "adsorbate_slab", "configurations")`,
    [/SYNTACTICAL]

    Args:
        db_path: [BRIEF] Path to SQLite database file. [/BRIEF]
                [DETAILED] Complete file path to the SQLite database where structures will be stored.
                The database will be created if it doesn't exist. Multiple tables can be stored
                in the same database for related structure types. Path should be writable and
                the directory should exist. [/DETAILED]
                [SYNTACTIC] Format: "Valid file path with .db extension" [/SYNTACTIC]
                [EXAMPLES] Examples: "materials.db", "data/structures.db", "databases/materials_db.db" [/EXAMPLES]

        structures_json: [BRIEF] JSON string containing structure data to save. [/BRIEF]
                        [DETAILED] A JSON-formatted string containing structure data with appropriate
                        format for the specified structure_type. The JSON should contain dictionaries
                        with structure IDs as keys and structure properties as values. This is typically
                        output from polymorph retrieval or slab generation tools. [/DETAILED]
                        [SYNTACTIC] Format: "Valid JSON string with structure data" [/SYNTACTIC]
                        [EXAMPLES] Examples: JSON from get_bulk_polymorphs_data, JSON from find_all_unique_slabs_upto_millerindex [/EXAMPLES]

        structure_type: [BRIEF] Type of structures being saved. Defaults to "slab". [/BRIEF]
                       [DETAILED] The type of crystal structures being saved, which determines the
                       database schema and column structure. "bulk" for bulk crystals, "slab" for
                       surface slabs, "adsorbate" for molecules, and "adsorbate_slab" for combined
                       surface-molecule systems. Each type has specialized properties and metadata. [/DETAILED]
                       [SYNTACTIC] Format: "bulk", "slab", "adsorbate", or "adsorbate_slab" [/SYNTACTIC]
                       [EXAMPLES] Examples: "bulk" (polymorphs), "slab" (surfaces), "adsorbate_slab" (adsorption configs) [/EXAMPLES]

        table_name: [BRIEF] Optional custom table name. [/BRIEF]
                   [DETAILED] Custom name for the database table where structures will be stored.
                   If None, the table name will be automatically determined based on structure_type
                   (e.g., "bulks", "slabs", "adsorbates", "adsorbate_slabs"). Custom names are
                   useful for organizing different datasets or studies. [/DETAILED]
                   [SYNTACTIC] Format: "Valid SQL table name or None" [/SYNTACTIC]
                   [EXAMPLES] Examples: "tio2_polymorphs", "silicon_surfaces", "co2_adsorption_configs", None [/EXAMPLES]

        additional_properties: [BRIEF] Optional dictionary of additional column definitions. [/BRIEF]
                              [DETAILED] Dictionary specifying additional columns to add to the database
                              table with their SQL data types. Useful for storing custom calculated
                              properties or metadata specific to a particular study. Keys are column
                              names and values are SQL data types. [/DETAILED]
                              [SYNTACTIC] Format: '{"column_name": "SQL_TYPE", ...} or None' [/SYNTACTIC]
                              [EXAMPLES] Examples: {"custom_descriptor": "REAL", "study_id": "TEXT"}, None [/EXAMPLES]

    Returns:
        str: [BRIEF] Status message indicating successful storage with count information. [/BRIEF]
             [DETAILED] A string message confirming successful storage of structures to the database,
             including the number of structures saved, the table name used, and the database path.
             This provides confirmation and summary information about the storage operation. [/DETAILED]
             [EXAMPLES] Example output: "Saved 25 slab structures to slab_structures in materials.db" [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When invalid structure_type is specified [/ERROR_WHEN]
                   [ERROR_DETAILS] Structure type not recognized or invalid JSON format [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Use valid structure types: "bulk", "slab", "adsorbate", "adsorbate_slab" [/ERROR_RECOVERY]
        sqlite3.Error: [ERROR_WHEN] When database operations fail [/ERROR_WHEN]
                      [ERROR_DETAILS] Database creation, table creation, or data insertion errors [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Check database path permissions and ensure valid data format [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When structures_json contains invalid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON string or incorrect data structure [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and ensure proper structure data format [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - SQLite limitations on concurrent access and database size
    - Schema is predefined and may not accommodate all possible structure properties
    - CIF strings are stored as text, which may be inefficient for very large structures
    - No automatic indexing optimization for large datasets
    - Limited support for complex relationships between structure types
    [/LIMITATIONS]
    """
    import json
    import sqlite3

    structures = json.loads(structures_json)

    # Determine table name if not provided
    if table_name is None:
        table_name = structure_type + "s"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Create schema based on structure type
        if structure_type == "bulk":
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    material_id TEXT PRIMARY KEY,
                    formula TEXT,
                    cif TEXT,
                    energy_above_hull REAL,
                    formation_energy_per_atom REAL,
                    band_gap REAL,
                    density REAL,
                    volume REAL,
                    nsites INTEGER,
                    space_group TEXT,
                    is_stable INTEGER
                )
            """)

            # Insert bulk structures
            for material_id, data in structures.items():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (material_id, formula, cif, energy_above_hull, formation_energy_per_atom,
                     band_gap, density, volume, nsites, space_group, is_stable)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        material_id,
                        data.get("formula", ""),
                        data["cif"],
                        data.get("energy_above_hull", None),
                        data.get("formation_energy_per_atom", None),
                        data.get("band_gap", None),
                        data.get("density", None),
                        data.get("volume", None),
                        data.get("nsites", None),
                        data.get("space_group", None),
                        1 if data.get("is_stable", False) else 0,
                    ),
                )

        elif structure_type == "slab":
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    slab_id TEXT PRIMARY KEY,
                    parent_material_id TEXT,
                    miller_index TEXT,
                    termination INTEGER,
                    cif TEXT,
                    area REAL,
                    num_sites INTEGER,
                    slab_thickness REAL,
                    relaxed INTEGER DEFAULT 0,
                    energy REAL,
                    surface_energy REAL
                )
            """)

            # Insert slabs
            for slab_id, data in structures.items():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (slab_id, parent_material_id, miller_index, termination, cif, area,
                     num_sites, slab_thickness, relaxed, energy, surface_energy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        slab_id,
                        data.get("parent_material_id", ""),
                        str(data.get("miller_index", [])),
                        data.get("termination", 0),
                        data["cif"],
                        data.get("area", None),
                        data.get("num_sites", None),
                        data.get("slab_thickness", None),
                        1 if data.get("relaxed", False) else 0,
                        data.get("energy", None),
                        data.get("surface_energy", None),
                    ),
                )

        elif structure_type == "adsorbate":
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    adsorbate_id TEXT PRIMARY KEY,
                    formula TEXT,
                    cif TEXT,
                    gas_phase_energy REAL,
                    num_atoms INTEGER
                )
            """)

            # Insert adsorbates
            for adsorbate_id, data in structures.items():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (adsorbate_id, formula, cif, gas_phase_energy, num_atoms)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        adsorbate_id,
                        data.get("formula", ""),
                        data["cif"],
                        data.get("gas_phase_energy", None),
                        data.get("num_atoms", None),
                    ),
                )

        elif structure_type == "adsorbate_slab":
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    config_id TEXT PRIMARY KEY,
                    slab_id TEXT,
                    adsorbate_id TEXT,
                    site_type TEXT,
                    site_index INTEGER,
                    site_coords TEXT,
                    height REAL,
                    orientation TEXT,
                    cif TEXT,
                    relaxed INTEGER DEFAULT 0,
                    energy REAL,
                    adsorption_energy REAL
                )
            """)

            # Insert adsorbate+slab configurations
            for config_id, data in structures.items():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {table_name}
                    (config_id, slab_id, adsorbate_id, site_type, site_index, site_coords,
                     height, orientation, cif, relaxed, energy, adsorption_energy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        config_id,
                        data.get("slab_id", ""),
                        data.get("adsorbate_id", ""),
                        data.get("site_type", ""),
                        data.get("site_index", None),
                        json.dumps(data.get("site_coords", [])),
                        data.get("height", None),
                        data.get("orientation", "default"),
                        data["cif"],
                        1 if data.get("relaxed", False) else 0,
                        data.get("energy", None),
                        data.get("adsorption_energy", None),
                    ),
                )
        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

        # Add additional properties if provided
        if additional_properties:
            for column_name, data_type in additional_properties.items():
                from contextlib import suppress

                with suppress(sqlite3.OperationalError):
                    cursor.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}"
                    )

    return f"Saved {len(structures)} {structure_type} structures to {table_name} in {db_path}"


########################
# Extra  tools for ablations
########################


@tool
def add_descriptor_column_to_db(
    db_path: str,
    table_name: str,
    descriptor_type: str,
    descriptor_function_code: str,
    batch_size: int = 100,
    dependencies: list | None = None,
) -> str:
    """[BRIEF] Add computed descriptor columns to a database table with batch processing for ML feature engineering. [/BRIEF]

    [DETAILED] This tool adds one or more descriptor columns to a database table by executing user-defined
    Python functions that compute features from existing data. It's designed for machine learning feature
    engineering in materials science, allowing computation of structural descriptors, coordination numbers,
    bond lengths, and other properties. The tool supports batch processing for memory efficiency and can
    handle dependencies between tables for complex feature calculations. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use for computing ML features from structural or property data
    - Essential for feature engineering in materials property prediction
    - Use when you need to add calculated properties to existing databases
    - Recommended for batch processing of large datasets
    - Avoid for simple column additions that don't require computation
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Connects to the specified SQLite database and table
    - Executes the provided Python function code to define descriptor computation
    - Processes database rows in batches for memory efficiency
    - Handles table joins if dependencies are specified
    - Creates new columns with appropriate data types
    - Tracks statistics for numerical descriptors
    - Updates the database with computed descriptor values
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Have database with structural or property data [/PREREQUISITE]
    2. [CURRENT] Apply this tool to compute and add descriptors [/CURRENT]
    3. [FOLLOW_UP] Use generate_ml_dataset_format to create ML-ready datasets [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `add_descriptor_column_to_db(db_path, "slabs", "coordination", function_code, 100)`,
    `add_descriptor_column_to_db(db_path, "table", "bond_length", code, batch_size=50)`,
    `add_descriptor_column_to_db(db_path, "table", "d_band", code, dependencies=["bulk"])`,
    [/SYNTACTICAL]

    Args:
        db_path: [BRIEF] Path to the SQLite database file. [/BRIEF]
                [DETAILED] Full path to the SQLite database file containing the table to be modified.
                The database should be accessible for both reading and writing operations. [/DETAILED]
                [SYNTACTIC] Format: "path/to/database.db" [/SYNTACTIC]
                [EXAMPLES] Examples: "materials.db", "data/surfaces.db" [/EXAMPLES]
        table_name: [BRIEF] Name of the table to modify. [/BRIEF]
                   [DETAILED] The name of the database table where descriptor columns will be added.
                   This table should contain the source data needed for descriptor computation. [/DETAILED]
                   [SYNTACTIC] Format: "table_name" [/SYNTACTIC]
                   [EXAMPLES] Examples: "slabs", "bulk_structures", "materials" [/EXAMPLES]
        descriptor_type: [BRIEF] Type of descriptor being computed. [/BRIEF]
                        [DETAILED] A descriptive name for the type of descriptor being computed, used
                        for logging and documentation purposes. This helps track what calculations
                        were performed on the data. [/DETAILED]
                        [SYNTACTIC] Format: "descriptor_type_name" [/SYNTACTIC]
                        [EXAMPLES] Examples: "coordination", "d_band", "bond_length", "surface_area" [/EXAMPLES]
        descriptor_function_code: [BRIEF] Python function code that computes descriptors from row data. [/BRIEF]
                                 [DETAILED] A string containing Python code that defines a function named
                                 'compute_descriptor' which takes a row dictionary as input and returns
                                 a dictionary mapping column names to computed values. [/DETAILED]
                                 [SYNTACTIC] Format: "Python code defining compute_descriptor(row) function" [/SYNTACTIC]
                                 [EXAMPLES] Examples: Function code computing coordination numbers, bond lengths [/EXAMPLES]
        batch_size: [BRIEF] Number of rows to process per batch. Defaults to 100. [/BRIEF]
                   [DETAILED] The number of database rows to process in each batch for memory efficiency.
                   Larger batches may be faster but use more memory, while smaller batches are more
                   memory-efficient but may be slower. [/DETAILED]
                   [SYNTACTIC] Format: positive integer [/SYNTACTIC]
                   [EXAMPLES] Examples: 50 (small), 100 (standard), 500 (large) [/EXAMPLES]
        dependencies: [BRIEF] List of other tables needed for computation. Defaults to None. [/BRIEF]
                     [DETAILED] A list of other table names that need to be joined with the main table
                     for descriptor computation. This allows access to related data from multiple tables
                     during feature calculation. [/DETAILED]
                     [SYNTACTIC] Format: list of table names ["table1", "table2"] or None [/SYNTACTIC]
                     [EXAMPLES] Examples: ["bulk_structures"], ["slabs", "adsorbates"], None [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with descriptor computation results and statistics. [/BRIEF]
             [DETAILED] A JSON-formatted string containing information about the descriptor computation,
             including the number of rows processed, columns added, and statistical information about
             the computed descriptors. [/DETAILED]
             [EXAMPLES] Example output: {"rows_processed": 1000, "columns_added": ["coord_num"], "statistics": {...}} [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When function code is invalid or doesn't define compute_descriptor [/ERROR_WHEN]
                   [ERROR_DETAILS] Syntax errors in function code or missing compute_descriptor function [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify function code syntax and ensure compute_descriptor is defined [/ERROR_RECOVERY]
        sqlite3.Error: [ERROR_WHEN] When database operations fail [/ERROR_WHEN]
                      [ERROR_DETAILS] Database connection issues, table not found, or permission errors [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Check database path and permissions [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires SQLite database format
    - Function code must be valid Python and define compute_descriptor
    - Memory usage depends on batch size and descriptor complexity
    - Limited error handling for complex descriptor calculations
    [/LIMITATIONS]
    """

    import json
    import sqlite3
    from collections import defaultdict

    import numpy as np

    # Execute the provided descriptor function code
    local_vars = {}
    exec(descriptor_function_code, {}, local_vars)
    compute_descriptor = local_vars.get("compute_descriptor")

    if not compute_descriptor:
        raise ValueError(
            "descriptor_function_code must define a 'compute_descriptor(row)' function."
        )

    # Statistics to track descriptor calculations
    stats = defaultdict(
        lambda: {
            "count": 0,
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0,
            "sum_sq": 0,
        }
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get total row count for progress reporting
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]

        # Process in batches for memory efficiency
        for offset in range(0, total_rows, batch_size):
            # Fetch a batch of rows
            cursor.execute(
                f"SELECT rowid, * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
            )
            rows = cursor.fetchall()

            batch_updates = []
            new_columns = set()

            # Process each row in the batch
            for row in rows:
                row_dict = dict(row)

                # If we need data from other tables (e.g., joining bulk + slab data)
                if dependencies:
                    for dep_table in dependencies:
                        join_column = (
                            f"{dep_table}_id"  # Assume foreign key naming convention
                        )
                        if join_column in row_dict:
                            join_id = row_dict[join_column]
                            dep_cursor = conn.cursor()
                            dep_cursor.execute(
                                f"SELECT * FROM {dep_table} WHERE {join_column.split('_')[0]}_id = ?",
                                (join_id,),
                            )
                            dep_row = dep_cursor.fetchone()
                            if dep_row:
                                # Add dependency data to row_dict with prefixed keys
                                row_dict.update(
                                    {
                                        f"{dep_table}_{k}": v
                                        for k, v in dict(dep_row).items()
                                    }
                                )

                # Calculate descriptors for this row
                try:
                    descriptor_values = compute_descriptor(row_dict)

                    # Update column tracking
                    for col_name, value in descriptor_values.items():
                        new_columns.add(col_name)

                        # Track statistics for numerical values
                        if isinstance(value, int | float) and not isinstance(
                            value, bool
                        ):
                            stats[col_name]["count"] += 1
                            stats[col_name]["min"] = min(stats[col_name]["min"], value)
                            stats[col_name]["max"] = max(stats[col_name]["max"], value)
                            stats[col_name]["sum"] += value
                            stats[col_name]["sum_sq"] += value * value

                    # Add to batch updates
                    batch_updates.append((row_dict["rowid"], descriptor_values))

                except Exception as e:
                    logger.info(
                        f"Error calculating descriptors for row {row_dict.get('rowid')}: {e}"
                    )

            # Make sure all new columns exist in the table
            for col_name in new_columns:
                try:
                    # Try to determine column type from first successful calculation
                    first_value = next(
                        (
                            values[col_name]
                            for _, values in batch_updates
                            if col_name in values
                        ),
                        None,
                    )

                    col_type = "TEXT"
                    if isinstance(first_value, int):
                        col_type = "INTEGER"
                    elif isinstance(first_value, float):
                        col_type = "REAL"
                    elif isinstance(first_value, bool):
                        col_type = "INTEGER"  # SQLite has no boolean

                    cursor.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    # Column likely already exists
                    pass

            # Apply all updates
            for rowid, values in batch_updates:
                for col_name, val in values.items():  # Renamed loop variable to 'val'
                    # Convert non-primitive types to JSON
                    if not isinstance(val, int | float | str | bool | type(None)):
                        serialized_val = json.dumps(val)

                    # Update the column
                    cursor.execute(
                        f"UPDATE {table_name} SET {col_name} = ? WHERE rowid = ?",
                        (serialized_val, rowid),
                    )

    # Calculate final statistics
    for data in stats.values():
        if data["count"] > 0:
            mean = data["sum"] / data["count"]
            variance = (data["sum_sq"] / data["count"]) - (mean * mean)
            std_dev = np.sqrt(max(0, variance))

            data["mean"] = mean
            data["std_dev"] = std_dev

    # Format results
    results = {
        "descriptor_type": descriptor_type,
        "table": table_name,
        "rows_processed": total_rows,
        "columns_added": list(new_columns),
        "statistics": dict(stats),
    }

    return json.dumps(results, indent=2)


@tool
def prepare_neural_network_dataset(
    polymorphs_json_path: str,
    output_path: str,
    target_property: str = "formation_energy_per_atom",
    sequence_features: bool = False,
    embedding_features: bool = True,
    test_split: float = 0.2,
) -> str:
    """[BRIEF] Prepare neural network-ready datasets with embeddings and sequence features for deep learning models. [/BRIEF]

    [DETAILED] This tool prepares datasets specifically designed for neural network models in materials science,
    creating feature representations suitable for deep learning architectures. It generates structural features,
    element embeddings, and optional sequence features from materials data, handling the conversion from
    crystallographic structures to tensor-ready formats. The tool supports various neural network architectures
    and provides proper train/test splits for model development. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when preparing data for neural network models (CNNs, RNNs, transformers)
    - Essential for deep learning approaches to materials property prediction
    - Use when you need embedding representations of chemical elements
    - Recommended for sequence-based modeling of atomic arrangements
    - Avoid for traditional ML models that don't require tensor inputs
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads materials data from JSON file containing structures and properties
    - Converts crystal structures to neural network features
    - Creates element embeddings using atomic properties
    - Generates sequence features from atomic positions if requested
    - Handles train/test splitting with stratification
    - Saves datasets in NPZ format optimized for neural network training
    - Includes normalization parameters and metadata for model development
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Have materials database with structures and target properties [/PREREQUISITE]
    2. [CURRENT] Apply this tool to create neural network datasets [/CURRENT]
    3. [FOLLOW_UP] Use output files for training neural network models [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `prepare_neural_network_dataset("materials.json", "nn_dataset", "formation_energy", False, True, 0.2)`,
    `prepare_neural_network_dataset("polymorphs.json", "output", sequence_features=True)`,
    `prepare_neural_network_dataset("data.json", "neural_data", test_split=0.15)`,
    [/SYNTACTICAL]

    Args:
        polymorphs_json_path: [BRIEF] Path to JSON file containing materials data. [/BRIEF]
                             [DETAILED] Full path to a JSON file containing materials structures and properties.
                             Each entry should include CIF structures and target properties for neural network
                             training. The file should be properly formatted with consistent property names. [/DETAILED]
                             [SYNTACTIC] Format: "path/to/materials.json" [/SYNTACTIC]
                             [EXAMPLES] Examples: "polymorphs.json", "data/materials_database.json" [/EXAMPLES]
        output_path: [BRIEF] Base path for saving dataset files. [/BRIEF]
                    [DETAILED] Base path for saving the neural network dataset files. The tool will create
                    train, test, and metadata files with appropriate suffixes. The path should be writable
                    and have sufficient space for the dataset files. [/DETAILED]
                    [SYNTACTIC] Format: "path/to/output_base" [/SYNTACTIC]
                    [EXAMPLES] Examples: "nn_dataset", "output/neural_data" [/EXAMPLES]
        target_property: [BRIEF] Property name to predict. Defaults to "formation_energy_per_atom". [/BRIEF]
                        [DETAILED] The name of the property in the JSON data that will be used as the target
                        for neural network training. This property should be numerical and present in most
                        or all entries in the dataset. [/DETAILED]
                        [SYNTACTIC] Format: "property_name" [/SYNTACTIC]
                        [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "bulk_modulus" [/EXAMPLES]
        sequence_features: [BRIEF] Whether to create sequence-based features. Defaults to False. [/BRIEF]
                          [DETAILED] Controls whether to generate sequence features from atomic positions,
                          suitable for RNN or transformer models. This creates flattened coordinate arrays
                          that can be used for sequence modeling of atomic arrangements. [/DETAILED]
                          [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                          [EXAMPLES] Examples: True (for sequence models), False (for standard NNs) [/EXAMPLES]
        embedding_features: [BRIEF] Whether to create element embedding features. Defaults to True. [/BRIEF]
                           [DETAILED] Controls whether to generate element embedding features using atomic
                           properties like electronegativity, atomic radius, and atomic mass. These features
                           are essential for most neural network models in materials science. [/DETAILED]
                           [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                           [EXAMPLES] Examples: True (recommended), False (structural only) [/EXAMPLES]
        test_split: [BRIEF] Fraction of data for test set. Defaults to 0.2. [/BRIEF]
                   [DETAILED] The fraction of the dataset to reserve for testing. The remaining data
                   will be used for training. A value of 0.2 means 20% test, 80% train, which is
                   standard for most machine learning applications. [/DETAILED]
                   [SYNTACTIC] Format: float between 0 and 1 [/SYNTACTIC]
                   [EXAMPLES] Examples: 0.15 (small test), 0.2 (standard), 0.25 (large test) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with dataset preparation results and file paths. [/BRIEF]
             [DETAILED] A JSON-formatted string containing information about the dataset preparation,
             including file paths for train/test data, dataset statistics, and metadata about the
             features and target property. [/DETAILED]
             [EXAMPLES] Example output: {"success": True, "train_path": "...", "test_path": "...", "dataset_info": {...}} [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When input data is invalid or target property is missing [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid JSON format, missing target property, or insufficient data [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify JSON format and ensure target property exists in data [/ERROR_RECOVERY]
        IOError: [ERROR_WHEN] When file operations fail [/ERROR_WHEN]
                [ERROR_DETAILS] Cannot read input file or write output files [/ERROR_DETAILS]
                [ERROR_RECOVERY] Check file permissions and ensure sufficient disk space [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires consistent JSON format for input data
    - Element embeddings are limited to predefined atomic properties
    - Sequence features are truncated/padded to fixed lengths
    - May not handle very large structures efficiently
    [/LIMITATIONS]
    """

    try:
        # Load polymorphs data
        with Path(polymorphs_json_path).open("r") as f:
            polymorphs = json.load(f)

        # Prepare neural network specific features
        nn_features = []
        targets = []
        metadata = []

        for poly in polymorphs:
            if target_property not in poly or poly[target_property] is None:
                continue

            try:
                from pymatgen.core import Structure

                structure = Structure.from_str(poly["cif"], fmt="cif")

                # Base features
                features = {
                    "structural": [
                        structure.density,
                        structure.volume,
                        len(structure),
                        len(structure.composition.elements),
                        structure.lattice.a,
                        structure.lattice.b,
                        structure.lattice.c,
                        structure.lattice.alpha,
                        structure.lattice.beta,
                        structure.lattice.gamma,
                    ]
                }

                # Element embeddings
                if embedding_features:
                    element_properties = []
                    for element in structure.composition.elements:
                        element_properties.extend(
                            [
                                element.atomic_radius or 0,
                                element.X,  # electronegativity
                                element.atomic_mass,
                                element.number,
                                element.row,
                                element.group,
                            ]
                        )

                    # Pad or truncate to fixed size (max 5 elements * 6 properties = 30)
                    element_properties = element_properties[:30]
                    element_properties.extend([0] * (30 - len(element_properties)))
                    features["elements"] = element_properties

                # Sequence features (atomic positions)
                if sequence_features:
                    positions = structure.frac_coords.flatten()
                    # Limit to first 150 coordinates (50 atoms * 3 coords)
                    positions = positions[:150]
                    positions = np.pad(positions, (0, max(0, 150 - len(positions))))
                    features["positions"] = positions.tolist()

                nn_features.append(features)
                targets.append(poly[target_property])
                metadata.append(
                    {
                        "material_id": poly.get("material_id", "unknown"),
                        "composition": poly.get("composition", "unknown"),
                    }
                )

            except Exception as e:
                logger.info(
                    f"Warning: Could not process {poly.get('material_id', 'unknown')}: {e}"
                )
                continue

        if len(nn_features) == 0:
            return json.dumps({"success": False, "error": "No valid samples found"})

        # Split data
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(nn_features))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_split, random_state=42
        )

        train_features = [nn_features[i] for i in train_idx]
        test_features = [nn_features[i] for i in test_idx]
        train_targets = [targets[i] for i in train_idx]
        test_targets = [targets[i] for i in test_idx]
        train_metadata = [metadata[i] for i in train_idx]
        test_metadata = [metadata[i] for i in test_idx]

        # Save as NPZ files for neural networks
        train_path = f"{output_path}_train.npz"
        test_path = f"{output_path}_test.npz"

        # Prepare arrays
        train_data = {"targets": np.array(train_targets), "metadata": train_metadata}
        test_data = {"targets": np.array(test_targets), "metadata": test_metadata}

        # Add feature arrays
        for feature_type in ["structural", "elements", "positions"]:
            if feature_type in train_features[0]:
                train_data[feature_type] = np.array(
                    [f[feature_type] for f in train_features]
                )
                test_data[feature_type] = np.array(
                    [f[feature_type] for f in test_features]
                )

        np.savez(train_path, **train_data)
        np.savez(test_path, **test_data)

        # Save dataset info
        metadata_path = f"{output_path}_metadata.json"
        dataset_info = {
            "target_property": target_property,
            "sequence_features": sequence_features,
            "embedding_features": embedding_features,
            "train_samples": len(train_features),
            "test_samples": len(test_features),
            "feature_types": list(train_features[0].keys()),
            "train_path": train_path,
            "test_path": test_path,
        }

        with Path(metadata_path).open("w") as f:
            json.dump(dataset_info, f, indent=2)

        return json.dumps(
            {
                "success": True,
                "train_path": train_path,
                "test_path": test_path,
                "metadata_path": metadata_path,
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
def prepare_graph_dataset(
    polymorphs_json_path: str,
    output_path: str,
    target_property: str = "formation_energy_per_atom",
    cutoff_radius: float = 5.0,
    test_split: float = 0.2,
) -> str:
    """[BRIEF] Prepare graph-based datasets for Graph Neural Networks (GNNs) with node and edge features. [/BRIEF]

    [DETAILED] This tool creates graph representations of crystal structures suitable for Graph Neural Networks,
    converting atomic structures into graphs where atoms are nodes and bonds are edges. It generates node features
    from atomic properties and edge features from interatomic distances, creating comprehensive graph datasets
    for GNN-based materials property prediction. The tool handles the complexity of converting 3D crystal
    structures to graph format while preserving essential structural information. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when preparing data for Graph Neural Network models
    - Essential for GNN-based materials property prediction
    - Use when you need to capture local atomic environments and bonding
    - Recommended for problems where atomic connectivity is important
    - Avoid for models that don't require graph representations
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads materials data from JSON file containing structures and properties
    - Converts each crystal structure to a graph representation
    - Creates node features from atomic properties (radius, electronegativity, etc.)
    - Generates edges based on interatomic distances within cutoff radius
    - Computes edge features from distances and geometric relationships
    - Saves graph datasets in JSON format suitable for GNN frameworks
    - Includes metadata about graph statistics and dataset properties
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Have materials database with crystal structures and properties [/PREREQUISITE]
    2. [CURRENT] Apply this tool to create graph datasets [/CURRENT]
    3. [FOLLOW_UP] Use output with GNN frameworks like PyTorch Geometric [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `prepare_graph_dataset("materials.json", "graph_data", "formation_energy", 5.0, 0.2)`,
    `prepare_graph_dataset("polymorphs.json", "gnn_dataset", cutoff_radius=4.0)`,
    `prepare_graph_dataset("data.json", "graphs", test_split=0.15)`,
    [/SYNTACTICAL]

    Args:
        polymorphs_json_path: [BRIEF] Path to JSON file containing materials data. [/BRIEF]
                             [DETAILED] Full path to a JSON file containing materials structures and properties.
                             Each entry should include CIF structures and target properties for GNN training.
                             The structures should be well-formed crystal structures with atomic coordinates. [/DETAILED]
                             [SYNTACTIC] Format: "path/to/materials.json" [/SYNTACTIC]
                             [EXAMPLES] Examples: "polymorphs.json", "data/crystal_database.json" [/EXAMPLES]
        output_path: [BRIEF] Base path for saving graph dataset files. [/BRIEF]
                    [DETAILED] Base path for saving the graph dataset files. The tool will create train,
                    test, and metadata files with appropriate suffixes. The files will be in JSON format
                    suitable for GNN frameworks. [/DETAILED]
                    [SYNTACTIC] Format: "path/to/output_base" [/SYNTACTIC]
                    [EXAMPLES] Examples: "graph_dataset", "output/gnn_data" [/EXAMPLES]
        target_property: [BRIEF] Property name to predict. Defaults to "formation_energy_per_atom". [/BRIEF]
                        [DETAILED] The name of the property in the JSON data that will be used as the target
                        for GNN training. This property should be numerical and represent a material property
                        that can be predicted from structural information. [/DETAILED]
                        [SYNTACTIC] Format: "property_name" [/SYNTACTIC]
                        [EXAMPLES] Examples: "formation_energy_per_atom", "band_gap", "elastic_modulus" [/EXAMPLES]
        cutoff_radius: [BRIEF] Cutoff radius for graph edges in Angstroms. Defaults to 5.0. [/BRIEF]
                      [DETAILED] The maximum distance between atoms to create edges in the graph representation.
                      This parameter controls the connectivity of the graph and affects the local environment
                      captured by the GNN. Typical values are 3-8 Å depending on the material system. [/DETAILED]
                      [SYNTACTIC] Format: positive float representing distance in Angstroms [/SYNTACTIC]
                      [EXAMPLES] Examples: 3.0 (short-range), 5.0 (standard), 8.0 (long-range) [/EXAMPLES]
        test_split: [BRIEF] Fraction of data for test set. Defaults to 0.2. [/BRIEF]
                   [DETAILED] The fraction of the dataset to reserve for testing. The remaining data
                   will be used for training. This split is important for evaluating GNN model performance
                   on unseen graph structures. [/DETAILED]
                   [SYNTACTIC] Format: float between 0 and 1 [/SYNTACTIC]
                   [EXAMPLES] Examples: 0.15 (small test), 0.2 (standard), 0.25 (large test) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with graph dataset preparation results and file paths. [/BRIEF]
             [DETAILED] A JSON-formatted string containing information about the graph dataset preparation,
             including file paths for train/test data, graph statistics (average nodes/edges per graph),
             and metadata about the dataset and target property. [/DETAILED]
             [EXAMPLES] Example output: {"success": True, "train_path": "...", "avg_nodes_per_graph": 42.3, ...} [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When input data is invalid or graph creation fails [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid JSON format, missing target property, or structure conversion errors [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify JSON format and ensure structures are valid crystal structures [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When crystal structures cannot be converted to graphs [/ERROR_WHEN]
                       [ERROR_DETAILS] Malformed crystal structures or inappropriate cutoff radius [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Check structure quality and adjust cutoff radius if needed [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Graph size varies significantly with structure size and cutoff radius
    - May create very large graphs for complex structures
    - Edge features are limited to distance-based properties
    - Does not account for periodic boundary conditions in edge creation
    [/LIMITATIONS]
    """
    try:
        # Load polymorphs data
        with Path(polymorphs_json_path).open("r") as f:
            polymorphs = json.load(f)

        graphs = []
        targets = []
        metadata = []

        for poly in polymorphs:
            if target_property not in poly or poly[target_property] is None:
                continue

            try:
                from pymatgen.core import Structure

                structure = Structure.from_str(poly["cif"], fmt="cif")

                # Create graph representation
                # Nodes: atoms with features
                # Edges: bonds within cutoff radius

                node_features = []
                edge_indices = []
                edge_features = []

                # Node features (atomic properties)
                for _i, site in enumerate(structure.sites):
                    element = site.specie
                    node_features.append(
                        [
                            element.atomic_radius or 1.0,
                            element.X,  # electronegativity
                            element.atomic_mass,
                            element.number,
                            float(element.row),
                            float(element.group),
                            site.coords[0],
                            site.coords[1],
                            site.coords[2],  # coordinates
                        ]
                    )

                # Edge features (distances and angles)
                for i, _site_i in enumerate(structure.sites):
                    for j, _site_j in enumerate(structure.sites):
                        if i != j:
                            distance = structure.get_distance(i, j)
                            if distance <= cutoff_radius:
                                edge_indices.append([i, j])
                                edge_features.append(
                                    [distance, 1.0 / distance]
                                )  # distance and inverse distance

                graph_data = {
                    "node_features": node_features,
                    "edge_indices": edge_indices,
                    "edge_features": edge_features,
                    "num_nodes": len(node_features),
                    "num_edges": len(edge_indices),
                }

                graphs.append(graph_data)
                targets.append(poly[target_property])
                metadata.append(
                    {
                        "material_id": poly.get("material_id", "unknown"),
                        "composition": poly.get("composition", "unknown"),
                        "num_atoms": len(structure),
                    }
                )

            except Exception as e:
                logger.info(
                    f"Warning: Could not create graph for {poly.get('material_id', 'unknown')}: {e}"
                )
                continue

        if len(graphs) == 0:
            return json.dumps({"success": False, "error": "No valid graphs created"})

        # Split data
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(graphs))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_split, random_state=42
        )

        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        train_targets = [targets[i] for i in train_idx]
        test_targets = [targets[i] for i in test_idx]
        train_metadata = [metadata[i] for i in train_idx]
        test_metadata = [metadata[i] for i in test_idx]

        # Save graph datasets
        train_path = f"{output_path}_train_graphs.json"
        test_path = f"{output_path}_test_graphs.json"

        train_data = {
            "graphs": train_graphs,
            "targets": train_targets,
            "metadata": train_metadata,
        }

        test_data = {
            "graphs": test_graphs,
            "targets": test_targets,
            "metadata": test_metadata,
        }

        with Path(train_path).open("w") as f:
            json.dump(train_data, f, indent=2)

        with Path(test_path).open("w") as f:
            json.dump(test_data, f, indent=2)

        # Save dataset info
        metadata_path = f"{output_path}_metadata.json"
        dataset_info = {
            "target_property": target_property,
            "cutoff_radius": cutoff_radius,
            "train_samples": len(train_graphs),
            "test_samples": len(test_graphs),
            "avg_nodes_per_graph": np.mean([g["num_nodes"] for g in graphs]),
            "avg_edges_per_graph": np.mean([g["num_edges"] for g in graphs]),
            "train_path": train_path,
            "test_path": test_path,
        }

        with Path(metadata_path).open("w") as f:
            json.dump(dataset_info, f, indent=2)

        return json.dumps(
            {
                "success": True,
                "train_path": train_path,
                "test_path": test_path,
                "metadata_path": metadata_path,
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
def generate_ml_dataset_format(
    db_path: str,
    table_name: str,
    feature_columns: list,
    target_column: str,
    output_format: str = "csv",
    output_path: str = "ml_dataset",
    validation_split: float = 0.2,
    normalize_features: bool = True,
    include_metadata: bool = True,
) -> str:
    """[BRIEF] Generate ML-ready datasets from database tables with proper formatting, splits, and normalization. [/BRIEF]

    [DETAILED] This tool extracts data from database tables and converts it to machine learning-ready formats
    with proper train/validation splits, feature normalization, and metadata tracking. It supports multiple
    output formats (CSV, JSON, NPZ) and handles missing data, feature scaling, and dataset documentation.
    This is the final step in the ML pipeline, converting processed descriptors and features into datasets
    ready for training traditional machine learning models. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after computing descriptors and features in database tables
    - Essential for creating final ML datasets from processed data
    - Use when you need properly formatted datasets for scikit-learn or similar libraries
    - Recommended for traditional ML models (Random Forest, SVM, etc.)
    - Avoid for deep learning models that require specialized preprocessing
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Connects to the specified database and extracts data from the table
    - Validates feature and target columns for completeness
    - Handles missing data by excluding incomplete rows
    - Applies feature normalization using z-score standardization
    - Splits data into train/validation sets with random sampling
    - Exports datasets in the specified format with proper structure
    - Generates metadata files with normalization parameters and dataset information
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Have database with computed descriptors using add_descriptor_column_to_db [/PREREQUISITE]
    2. [CURRENT] Apply this tool to create ML-ready datasets [/CURRENT]
    3. [FOLLOW_UP] Use output files for training ML models with scikit-learn or similar [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `generate_ml_dataset_format("data.db", "materials", ["coord", "area"], "energy", "csv", "dataset")`,
    `generate_ml_dataset_format("db.sqlite", "slabs", features, "stability", "json", "ml_data")`,
    `generate_ml_dataset_format("materials.db", "table", cols, "target", output_format="npz")`,
    [/SYNTACTICAL]

    Args:
        db_path: [BRIEF] Path to the SQLite database file. [/BRIEF]
                [DETAILED] Full path to the SQLite database file containing the processed data with
                computed descriptors and target properties. The database should be accessible for
                reading and contain the specified table. [/DETAILED]
                [SYNTACTIC] Format: "path/to/database.db" [/SYNTACTIC]
                [EXAMPLES] Examples: "materials.db", "data/processed_data.sqlite" [/EXAMPLES]
        table_name: [BRIEF] Name of the table containing the data. [/BRIEF]
                   [DETAILED] The name of the database table containing the features and target values
                   for ML dataset generation. This table should have computed descriptors and complete
                   data for the specified columns. [/DETAILED]
                   [SYNTACTIC] Format: "table_name" [/SYNTACTIC]
                   [EXAMPLES] Examples: "materials", "slabs", "processed_data" [/EXAMPLES]
        feature_columns: [BRIEF] List of column names to use as features. [/BRIEF]
                        [DETAILED] A list of column names from the database table that will be used as
                        input features for machine learning. These should be numerical columns containing
                        computed descriptors or material properties. [/DETAILED]
                        [SYNTACTIC] Format: list of column names ["col1", "col2", "col3"] [/SYNTACTIC]
                        [EXAMPLES] Examples: ["coordination_number", "surface_area"], ["d_band_center", "work_function"] [/EXAMPLES]
        target_column: [BRIEF] Column name for the prediction target. [/BRIEF]
                      [DETAILED] The name of the column containing the target values for machine learning.
                      This should be a numerical column representing the property you want to predict
                      using the feature columns. [/DETAILED]
                      [SYNTACTIC] Format: "column_name" [/SYNTACTIC]
                      [EXAMPLES] Examples: "formation_energy", "band_gap", "adsorption_energy" [/EXAMPLES]
        output_format: [BRIEF] Format for the dataset output. Defaults to "csv". [/BRIEF]
                      [DETAILED] The format for saving the ML dataset. Options include "csv" for tabular data,
                      "json" for structured data, and "npz" for NumPy arrays. CSV is most compatible with
                      standard ML libraries. [/DETAILED]
                      [SYNTACTIC] Format: string from ["csv", "json", "npz"] [/SYNTACTIC]
                      [EXAMPLES] Examples: "csv" (most common), "json" (structured), "npz" (NumPy) [/EXAMPLES]
        output_path: [BRIEF] Base path for output files. Defaults to "ml_dataset". [/BRIEF]
                    [DETAILED] Base path for saving the ML dataset files. The tool will create separate
                    files for train/validation splits with appropriate suffixes. Should be a writable
                    location with sufficient space. [/DETAILED]
                    [SYNTACTIC] Format: "path/to/output_base" [/SYNTACTIC]
                    [EXAMPLES] Examples: "ml_dataset", "output/materials_data" [/EXAMPLES]
        validation_split: [BRIEF] Fraction of data for validation set. Defaults to 0.2. [/BRIEF]
                         [DETAILED] The fraction of the dataset to reserve for validation. The remaining
                         data will be used for training. This split is crucial for proper model evaluation
                         and hyperparameter tuning. [/DETAILED]
                         [SYNTACTIC] Format: float between 0 and 1 [/SYNTACTIC]
                         [EXAMPLES] Examples: 0.15 (small validation), 0.2 (standard), 0.25 (large validation) [/EXAMPLES]
        normalize_features: [BRIEF] Whether to normalize features. Defaults to True. [/BRIEF]
                           [DETAILED] Controls whether to apply z-score normalization to feature columns.
                           Normalization is generally recommended for ML models to ensure features are
                           on similar scales and improve training stability. [/DETAILED]
                           [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                           [EXAMPLES] Examples: True (recommended), False (raw features) [/EXAMPLES]
        include_metadata: [BRIEF] Whether to include metadata files. Defaults to True. [/BRIEF]
                         [DETAILED] Controls whether to generate metadata files containing information
                         about the dataset, normalization parameters, and data splits. These files are
                         important for reproducibility and model deployment. [/DETAILED]
                         [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                         [EXAMPLES] Examples: True (recommended), False (data only) [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string with dataset generation results and file paths. [/BRIEF]
             [DETAILED] A JSON-formatted string containing information about the generated ML dataset,
             including file paths for train/validation data, dataset statistics, and metadata about
             the features and normalization applied. [/DETAILED]
             [EXAMPLES] Example output: {"format": "csv", "train_path": "...", "validation_path": "...", "train_samples": 800} [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When specified columns don't exist or contain invalid data [/ERROR_WHEN]
                   [ERROR_DETAILS] Missing columns, non-numerical data, or insufficient valid rows [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify column names and ensure data is numerical and complete [/ERROR_RECOVERY]
        sqlite3.Error: [ERROR_WHEN] When database operations fail [/ERROR_WHEN]
                      [ERROR_DETAILS] Database connection issues, table not found, or query errors [/ERROR_DETAILS]
                      [ERROR_RECOVERY] Check database path and table names [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires numerical data for all feature and target columns
    - Excludes rows with any missing values
    - Normalization assumes normal distribution of features
    - Limited to single-table datasets without complex joins
    [/LIMITATIONS]
    """
    import json
    import sqlite3

    import numpy as np

    # Connect to database
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get column names from the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        all_columns = [row["name"] for row in cursor.fetchall()]

        # Make sure specified columns exist
        valid_feature_columns = [col for col in feature_columns if col in all_columns]

        if not valid_feature_columns:
            return json.dumps({"error": "No valid feature columns found"})

        if target_column not in all_columns:
            return json.dumps({"error": f"Target column '{target_column}' not found"})

        # Fetch all data
        columns_to_fetch = [*valid_feature_columns, target_column]
        cursor.execute(f"SELECT {', '.join(columns_to_fetch)} FROM {table_name}")
        rows = cursor.fetchall()

        if not rows:
            return json.dumps({"error": "No data found in table"})

        # Convert to numpy arrays
        feature_data = []
        target_data = []

        for row in rows:
            # Only include rows with valid target values
            if row[target_column] is not None:
                try:
                    target_val = float(row[target_column])

                    # Extract feature values
                    feature_row = []
                    valid_row = True

                    for col in valid_feature_columns:
                        if row[col] is not None:
                            try:
                                feature_row.append(float(row[col]))
                            except (ValueError, TypeError):
                                valid_row = False
                                break
                        else:
                            valid_row = False
                            break

                    # Only add complete rows
                    if valid_row and len(feature_row) == len(valid_feature_columns):
                        feature_data.append(feature_row)
                        target_data.append(target_val)

                except (ValueError, TypeError):
                    # Skip rows with invalid target values
                    continue

        if not feature_data:
            return json.dumps({"error": "No valid data rows found"})

        # Convert to numpy arrays
        X = np.array(feature_data)
        y = np.array(target_data)

        # Calculate normalization parameters if needed
        normalization_params = {}
        if normalize_features:
            normalization_params = {}
            for i, col in enumerate(valid_feature_columns):
                col_data = X[:, i]
                col_mean = np.mean(col_data)
                col_std = np.std(col_data)

                # Avoid division by zero
                if col_std == 0:
                    col_std = 1.0

                normalization_params[col] = {
                    "mean": float(col_mean),
                    "std": float(col_std),
                }

                # Normalize the data
                X[:, i] = (col_data - col_mean) / col_std

        # Split into train/validation sets
        n_samples = len(X)
        rng = np.random.default_rng()
        indices = rng.permutation(n_samples)
        n_validation = int(validation_split * n_samples)

        validation_idx = indices[:n_validation]
        train_idx = indices[n_validation:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[validation_idx], y[validation_idx]

        # Create output directory if it doesn't exist
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save in the requested format
        metadata = {
            "features": valid_feature_columns,
            "target": target_column,
            "normalization": normalization_params if normalize_features else None,
            "dataset_size": n_samples,
            "train_size": len(X_train),
            "validation_size": len(X_val),
            "validation_split": validation_split,
        }

        if output_format.lower() == "csv":
            import csv

            # Save training data
            train_path = f"{output_path}_train.csv"
            from pathlib import Path

            with Path(train_path).open("w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([*valid_feature_columns, target_column])
                # Write data
                for i in range(len(X_train)):
                    writer.writerow([*list(X_train[i]), y_train[i]])

            # Save validation data
            val_path = f"{output_path}_validation.csv"
            with Path(val_path).open("w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([*valid_feature_columns, target_column])
                # Write data
                for i in range(len(X_val)):
                    writer.writerow([*list(X_val[i]), y_val[i]])

            # Save metadata if requested
            if include_metadata:
                meta_path = f"{output_path}_metadata.json"
                with Path(meta_path).open("w") as f:
                    json.dump(metadata, f, indent=2)

            result = {
                "format": "csv",
                "train_path": train_path,
                "validation_path": val_path,
                "metadata_path": meta_path if include_metadata else None,
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
            }

        elif output_format.lower() == "json":
            # Save training data
            train_path = f"{output_path}_train.json"
            train_data = {
                "features": valid_feature_columns,
                "target": target_column,
                "data": [
                    {
                        "features": {
                            col: float(X_train[i, j])
                            for j, col in enumerate(valid_feature_columns)
                        },
                        "target": float(y_train[i]),
                    }
                    for i in range(len(X_train))
                ],
            }
            with Path(train_path).open("w") as f:
                json.dump(train_data, f, indent=2)

            # Save validation data
            val_path = f"{output_path}_validation.json"
            val_data = {
                "features": valid_feature_columns,
                "target": target_column,
                "data": [
                    {
                        "features": {
                            col: float(X_val[i, j])
                            for j, col in enumerate(valid_feature_columns)
                        },
                        "target": float(y_val[i]),
                    }
                    for i in range(len(X_val))
                ],
            }
            with Path(val_path).open("w") as f:
                json.dump(val_data, f, indent=2)

            # Save metadata if requested
            if include_metadata:
                meta_path = f"{output_path}_metadata.json"
                with Path(meta_path).open("w") as f:
                    json.dump(metadata, f, indent=2)

            result = {
                "format": "json",
                "train_path": train_path,
                "validation_path": val_path,
                "metadata_path": meta_path if include_metadata else None,
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
            }

        elif output_format.lower() == "npz":
            # Save as numpy arrays
            np_path = f"{output_path}.npz"
            np.savez(
                np_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=valid_feature_columns,
                target_name=target_column,
            )

            # Save metadata if requested
            if include_metadata:
                meta_path = f"{output_path}_metadata.json"
                with Path(meta_path).open("w") as f:
                    json.dump(metadata, f, indent=2)

            result = {
                "format": "npz",
                "npz_path": np_path,
                "metadata_path": meta_path if include_metadata else None,
                "train_samples": len(X_train),
                "validation_samples": len(X_val),
                "train_key": "X_train",
                "train_target_key": "y_train",
                "validation_key": "X_val",
                "validation_target_key": "y_val",
            }

        else:
            return json.dumps({"error": f"Unsupported output format: {output_format}"})

        return json.dumps(result, indent=2)


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
        "prepare_neural_network_dataset": prepare_neural_network_dataset,
        "train_xgboost_model": train_xgboost_model,
        "evaluate_xgboost_model": evaluate_xgboost_model,
        "perform_cross_validation": perform_cross_validation,
    }
