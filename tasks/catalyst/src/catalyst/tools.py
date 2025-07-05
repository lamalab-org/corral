import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np  # Import numpy
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tool_utils import (
    ensure_directory_exists,
    find_surface_atoms_with_voronoi,
    generate_output_capture_code,
    load_structure,
    parse_execution_output,
    safe_convert_timeout,
    set_fixed_atom_constraints,
    standardize_bulk,
    tile_atoms,
)

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
    - get_structure_from_mp_text("mp-149")  # Silicon structure
    - get_structure_from_mp_text("mp-20066")  # CO2 structure
    - get_structure_from_mp_text("mp-2")  # Other material
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
    - create_slab_from_structure_text(cif_string, (1,1,1), 12, 5, True)
    - create_slab_from_structure_text(cif_string, (1,0,0), 15, 10, False)
    - create_slab_from_structure_text(cif_string)  # Uses defaults
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
    - enumerate_slabs_text(cif_string, (1,1,1), 12, 5)
    - enumerate_slabs_text(cif_string, (1,0,0), 15, 10)
    - enumerate_slabs_text(cif_string)  # Uses default parameters
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


@tool
def choose_slab_text(slabs_json: str, index: int = 0) -> str:
    """[BRIEF] Select a specific slab from a JSON dictionary of enumerated slabs by index. [/BRIEF]

    [DETAILED] This tool selects one surface slab from a collection of enumerated slabs based on
    its index number. It's designed to work with the output from enumerate_slabs_text, allowing
    users to choose a specific surface termination for further analysis. This selection process
    is crucial for focusing on the most relevant or interesting surface termination for catalysis
    or adsorption studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after enumerate_slabs_text to select a specific surface termination
    - Best suited for systematic exploration of different surface terminations
    - Could be useful for workflows that require a single slab for adsorption or catalysis studies
    - Recommended when you need to compare results from different surface terminations
    - One can randomly pick index to select a slab from the enumerated list if they want to randomly pick a slab
    - Avoid when you only need one slab (use create_slab_from_structure_text directly)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the JSON string containing multiple slab structures
    - Locates the slab with the specified index key (e.g., "slab_0", "slab_1")
    - Extracts the CIF string for the selected slab
    - Returns the CIF content ready for use in subsequent tools
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First run enumerate_slabs_text to generate multiple slab terminations [/PREREQUISITE]
    2. [CURRENT] Apply this tool to select a specific slab by index. Can be coupled with io tools or python execution tools to figure out which index to use depending on the task, for example, filter based on miller index[/CURRENT]
    3. [FOLLOW_UP] Use the selected slab with adsorption tools like get_adsorption_sites_text [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - choose_slab_text(slabs_json, 0)  # Select first slab
    - choose_slab_text(slabs_json, 1)  # Select second slab
    - choose_slab_text(slabs_json)     # Select first slab (default)
    [/SYNTACTICAL]

    Args:
        slabs_json: [BRIEF] JSON string mapping slab keys to CIF strings. [/BRIEF]
                   [DETAILED] A JSON-formatted string containing a dictionary where keys are
                   slab identifiers (e.g., "slab_0", "slab_1") and values are the corresponding
                   CIF strings. This should be the output from enumerate_slabs_text tool. The
                   JSON structure must be valid and contain at least one slab entry. [/DETAILED]
                   [SYNTACTIC] Format: 'Valid JSON string with "slab_X" keys and CIF string values' [/SYNTACTIC]
                   [EXAMPLES] Examples: '{"slab_0": "CIF content...", "slab_1": "CIF content..."}' [/EXAMPLES]

        index: [BRIEF] Index of the slab to select. Defaults to 0. [/BRIEF]
              [DETAILED] The numerical index of the slab to select from the JSON dictionary.
              This corresponds to the enumeration order from enumerate_slabs_text, where
              index 0 is the first slab, index 1 is the second, and so on. The tool will
              look for a key named "slab_{index}" in the JSON dictionary. [/DETAILED]
              [SYNTACTIC] Format: non-negative integer [/SYNTACTIC]
              [EXAMPLES] Examples: 0 (first slab), 1 (second slab), 2 (third slab) [/EXAMPLES]

    Returns:
        str: [BRIEF] CIF string for the selected slab. [/BRIEF]
             [DETAILED] A properly formatted CIF string containing the structure data for
             the selected slab. This includes atomic positions, lattice parameters, and
             all necessary crystallographic information. The CIF can be used directly
             with other structure analysis tools. [/DETAILED]
             [EXAMPLES] Example output: CIF string with selected slab structure [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When the specified slab index is not found in the JSON [/ERROR_WHEN]
                   [ERROR_DETAILS] The key "slab_{index}" does not exist in the JSON dictionary [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Check available slab indices in the JSON or use a valid index [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When the slabs_json string is not valid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON string or incorrect format [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and ensure it's output from enumerate_slabs_text [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Cannot validate the quality or stability of the selected slab
    - Does not provide information about surface termination characteristics
    - Limited to slabs generated by enumerate_slabs_text tool
    - Cannot modify or optimize the selected slab structure
    [/LIMITATIONS]
    """
    import json

    slabs = json.loads(slabs_json)
    key = f"slab_{index}"
    if key not in slabs:
        raise ValueError(f"Slab index {index} not found.")
    return slabs[key]


@tool
def get_adsorption_sites_text(slab_cif: str) -> str:
    """[BRIEF] Identify and classify all possible adsorption sites on a surface slab. [/BRIEF]

    [DETAILED] This tool analyzes a surface slab structure to identify and classify potential
    adsorption sites where molecules can bind. It uses geometric and chemical analysis to
    determine different types of binding sites such as top sites (above surface atoms),
    bridge sites (between two atoms), and hollow sites (in multi-atom depressions). This
    analysis is fundamental for understanding surface reactivity and designing catalysts. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to identify all possible adsorption sites on a surface
    - Best suited for systematic studies of surface reactivity and catalysis
    - Essential for understanding how molecules interact with surfaces
    - Recommended before placing adsorbates to understand binding options
    - Avoid when you already know the specific binding site coordinates
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the slab CIF structure to identify surface atoms
    - Uses AdsorbateSiteFinder to geometrically analyze the surface topology
    - Classifies sites based on coordination environment (top, bridge, hollow)
    - Calculates fractional coordinates for each potential binding site
    - Returns sites organized by type in a JSON format for easy selection
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain a slab structure using choose_slab_text or create_slab_from_structure_text [/PREREQUISITE]
    2. [CURRENT] Apply this tool to identify all adsorption sites on the surface [/CURRENT]
    3. [FOLLOW_UP] Use choose_adsorption_site_text to select a specific site for adsorbate placement [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - get_adsorption_sites_text(slab_cif_string)
    - get_adsorption_sites_text(output_from_choose_slab_text)
    [/SYNTACTICAL]

    Args:
        slab_cif: [BRIEF] CIF string of the surface slab structure. [/BRIEF]
                 [DETAILED] A properly formatted CIF string containing the surface slab structure
                 with atomic positions, lattice parameters, and surface geometry. This should be
                 a two-dimensional periodic structure with a well-defined surface and vacuum
                 region. The structure is analyzed to identify potential adsorption sites. [/DETAILED]
                 [SYNTACTIC] Format: "Valid CIF format string with slab structure" [/SYNTACTIC]
                 [EXAMPLES] Examples: CIF string from choose_slab_text or create_slab_from_structure_text [/EXAMPLES]

    Returns:
        str: [BRIEF] JSON string containing classified adsorption sites with fractional coordinates. [/BRIEF]
             [DETAILED] A JSON-formatted string containing a dictionary where keys are site types
             (e.g., "top", "bridge", "hollow") and values are lists of fractional coordinates
             for each site of that type. Each coordinate is a list of three numbers [x, y, z]
             representing the fractional position within the unit cell. [/DETAILED]
             [EXAMPLES] Example output: '{"top": [[0.0, 0.0, 0.9], [0.5, 0.5, 0.9]], "bridge": [[0.25, 0.25, 0.85]]}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When the CIF string is malformed or doesn't represent a valid slab [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format, missing surface atoms, or improper slab structure [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify CIF format and ensure it represents a proper surface slab [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When the slab structure cannot be analyzed for adsorption sites [/ERROR_WHEN]
                       [ERROR_DETAILS] Insufficient surface area, unclear surface definition, or geometric issues [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Check slab structure quality and surface termination [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Does not account for surface relaxation or reconstruction effects
    - Cannot predict relative binding strengths or preferences
    - Limited to geometric analysis without chemical bonding considerations
    - May not identify all possible sites for large or complex molecules
    [/LIMITATIONS]
    """
    import json

    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Structure

    finder = AdsorbateSiteFinder(Structure.from_str(slab_cif, fmt="cif"))
    sites = (
        finder.find_adsorption_sites()
    )  # returns a dict, e.g. {"top": [site1, ...], "bridge": [...], ...}

    # Convert sites to a serializable format (list of fractional coordinates)
    serializable_sites = {
        key: [
            list(site.frac_coords) if hasattr(site, "frac_coords") else list(site)
            for site in site_list
        ]
        for key, site_list in sites.items()
    }

    return json.dumps(serializable_sites, indent=2)


@tool
def choose_adsorption_site_text(
    adsorption_sites_json: str, site_type: str, index: int = 0
) -> list[float]:
    """[BRIEF] Select a specific adsorption site from classified sites by type and index. [/BRIEF]

    [DETAILED] This tool selects one specific adsorption site from a collection of classified
    sites based on the site type (top, bridge, hollow) and index within that type.
    This selection is crucial for systematic studies of different
    binding environments and their effects on adsorption energetics. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after identifying all possible adsorption site to select a specific binding site
    - Best suited for systematic comparison of different site types
    - Essential for placing adsorbates at specific coordination environments
    - Recommended when studying site-specific reactivity or selectivity
    - Avoid when you need to place adsorbates at multiple sites simultaneously
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the JSON string containing classified adsorption sites
    - Locates the specified site type in the dictionary
    - Selects the site at the specified index within that type
    - Returns the fractional coordinates as a list of three floats
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First run get_adsorption_sites_text to identify available sites [/PREREQUISITE]
    2. [CURRENT] Apply this tool to select a specific site by type and index [/CURRENT]
    3. [FOLLOW_UP] Use the coordinates with add_adsorbate_to_slab_text for molecule placement [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - choose_adsorption_site_text(sites_json, "top", 0)    # First top site
    - choose_adsorption_site_text(sites_json, "bridge", 1) # Second bridge site
    - choose_adsorption_site_text(sites_json, "hollow", 0) # First hollow site
    [/SYNTACTICAL]

    Args:
        adsorption_sites_json: [BRIEF] JSON string mapping site types to lists of fractional coordinates. [/BRIEF]
                              [DETAILED] A JSON-formatted string containing a dictionary where keys are
                              site types (e.g., "top", "bridge", "hollow") and values are lists of
                              fractional coordinates. This should be the output from get_adsorption_sites_text.
                              Each coordinate is a list of three numbers representing position within the unit cell. [/DETAILED]
                              [SYNTACTIC] Format: 'Valid JSON string with site type keys and coordinate list values' [/SYNTACTIC]
                              [EXAMPLES] Examples: '{"top": [[0.0, 0.0, 0.9]], "bridge": [[0.25, 0.25, 0.85]]}' [/EXAMPLES]

        site_type: [BRIEF] Type of adsorption site to select. [/BRIEF]
                  [DETAILED] The type of binding site to select from the available options. Common
                  types include "top" (above surface atoms), "bridge" (between two atoms), and
                  "hollow" (in multi-atom depressions). The type must exist in the JSON dictionary
                  and determines the coordination environment of the selected site. [/DETAILED]
                  [SYNTACTIC] Format: string matching available site types [/SYNTACTIC]
                  [EXAMPLES] Examples: "top" (on-top), "bridge" (between atoms), "hollow" (in depression) [/EXAMPLES]

        index: [BRIEF] Index of the site within the specified type. Defaults to 0. [/BRIEF]
              [DETAILED] The numerical index of the site to select from the list of sites
              of the specified type. Index 0 selects the first site, index 1 the second,
              and so on. The index must be within the range of available sites for the
              specified type. [/DETAILED]
              [SYNTACTIC] Format: non-negative integer [/SYNTACTIC]
              [EXAMPLES] Examples: 0 (first site), 1 (second site), 2 (third site) [/EXAMPLES]

    Returns:
        list[float]: [BRIEF] Fractional coordinates of the selected adsorption site. [/BRIEF]
                    [DETAILED] A list of three floating-point numbers representing the fractional
                    coordinates [x, y, z] of the selected adsorption site within the unit cell.
                    These coordinates can be used directly for adsorbate placement and represent
                    the optimal binding position for the specified site type. [/DETAILED]
                    [EXAMPLES] Example output: [0.0, 0.0, 0.9] or [0.25, 0.25, 0.85] [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When the specified site type is not found in the JSON [/ERROR_WHEN]
                   [ERROR_DETAILS] The site_type key does not exist in the JSON dictionary [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Check available site types in the JSON or use a valid type [/ERROR_RECOVERY]
        IndexError: [ERROR_WHEN] When the specified index is out of range for the site type [/ERROR_WHEN]
                   [ERROR_DETAILS] The index is greater than or equal to the number of sites of that type [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Check the number of available sites for the specified type [/ERROR_RECOVERY]
        JSONDecodeError: [ERROR_WHEN] When the adsorption_sites_json string is not valid JSON [/ERROR_WHEN]
                        [ERROR_DETAILS] Malformed JSON string or incorrect format [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Verify JSON format and ensure it's output from get_adsorption_sites_text [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Cannot evaluate the relative quality or stability of different sites
    - Limited to sites identified by get_adsorption_sites_text
    - Cannot modify or optimize the selected site coordinates
    [/LIMITATIONS]
    """
    import json

    sites = json.loads(adsorption_sites_json)
    if site_type not in sites:
        raise ValueError(f"Site type {site_type} not found.")
    if index >= len(sites[site_type]):
        raise ValueError(f"Site index {index} not found.")
    return sites[site_type][index]


@tool
def add_adsorbate_to_slab_text(
    slab_cif: str,
    adsorbate_cif: str,
    height: float = 2.0,
    site: list[float] | None = None,
) -> str:
    """[BRIEF] Place an adsorbate molecule on a surface slab at a specified adsorption site. [/BRIEF]

    [DETAILED] This tool combines a surface slab with an adsorbate molecule by placing the
    adsorbate at a specific binding site on the surface. It handles the geometric placement
    of the molecule at the correct height above the surface and ensures proper structural
    integration. This is essential for creating realistic surface-adsorbate systems for
    computational studies of catalysis, adsorption energetics, and surface reactivity. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to create a surface-adsorbate system
    - Recommended for systematic studies of different binding sites or orientations
    - Avoid when you need complex multi-adsorbate systems or surface reconstructions
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the slab CIF structure to identify the surface geometry
    - Loads the adsorbate as a molecular structure (handles both XYZ and CIF formats)
    - Uses AdsorbateSiteFinder to place the adsorbate at the specified site
    - Adjusts the vertical position according to the specified height parameter
    - Combines the structures into a single CIF-formatted output
    - Automatically selects a top site if no specific site is provided
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain slab from choose_slab_text and adsorbate from get_structure_from_mp_text [/PREREQUISITE]
    2. [CURRENT] Apply this tool to place the adsorbate on the surface [/CURRENT]
    3. [FOLLOW_UP] Use the combined structure for further analysis or optimization [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - add_adsorbate_to_slab_text(slab_cif, co2_cif, 2.0, [0.0, 0.0, 0.9])
    - add_adsorbate_to_slab_text(slab_cif, molecule_cif, 1.5)  # Auto-select top site
    - add_adsorbate_to_slab_text(slab_cif, adsorbate_cif)      # Default height and site
    [/SYNTACTICAL]

    Args:
        slab_cif: [BRIEF] CIF string of the surface slab structure. [/BRIEF]
                 [DETAILED] A properly formatted CIF string containing the surface slab structure
                 on which the adsorbate will be placed. This should be a two-dimensional periodic
                 structure with a well-defined surface and vacuum region. The slab provides the
                 substrate for molecular adsorption. [/DETAILED]
                 [SYNTACTIC] Format: "Valid CIF format string with slab structure" [/SYNTACTIC]
                 [EXAMPLES] Examples: CIF string from choose_slab_text output [/EXAMPLES]

        adsorbate_cif: [BRIEF] CIF string of the adsorbate molecule structure. [/BRIEF]
                      [DETAILED] A CIF or XYZ formatted string containing the molecular structure
                      of the adsorbate to be placed on the surface. This can be a small molecule
                      like CO2, H2O, or more complex organic molecules. The tool will attempt to
                      parse both CIF and XYZ formats automatically. [/DETAILED]
                      [SYNTACTIC] Format: "Valid CIF or XYZ format string with molecular structure" [/SYNTACTIC]
                      [EXAMPLES] Examples: CIF string from get_structure_from_mp_text for molecules [/EXAMPLES]

        height: [BRIEF] Height in Angstroms above the surface for adsorbate placement. Defaults to 2.0. [/BRIEF]
               [DETAILED] The vertical distance above the surface at which the adsorbate will be
               placed. This parameter controls the initial separation between the adsorbate and
               the surface atoms. Typical values range from 1.5 to 3.0 Å depending on the
               molecular size and expected binding interaction. [/DETAILED]
               [SYNTACTIC] Format: positive float representing distance in Angstroms [/SYNTACTIC]
               [EXAMPLES] Examples: 1.5 (close to slab), 2.0, 2.5 (distant from molecule) [/EXAMPLES]

        site: [BRIEF] Optional fractional coordinates for adsorbate placement. [/BRIEF]
             [DETAILED] A list of three floating-point numbers representing the fractional
             coordinates [x, y, z] where the adsorbate should be placed. If not provided,
             the tool will automatically select the first available top site. These coordinates
             should typically come from choose_adsorption_site_text output. [/DETAILED]
             [SYNTACTIC] Format: list of three floats [x, y, z] or None [/SYNTACTIC]
             [EXAMPLES] Examples: [0.0, 0.0, 0.9], [0.5, 0.5, 0.9], None (auto-select) [/EXAMPLES]

    Returns:
        str: [BRIEF] CIF string of the combined surface-adsorbate structure. [/BRIEF]
             [DETAILED] A properly formatted CIF string containing the combined structure
             with the adsorbate placed on the surface at the specified position and height.
             This structure includes both the original slab atoms and the adsorbate atoms,
             properly integrated into a single periodic structure. [/DETAILED]
             [EXAMPLES] Example output: CIF string with both slab and adsorbate atoms [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When CIF strings are malformed or adsorbate cannot be parsed [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format, unsupported molecule format, or structural issues [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify CIF formats and ensure adsorbate is a valid molecular structure [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When adsorbate placement fails due to geometric constraints [/ERROR_WHEN]
                       [ERROR_DETAILS] Site coordinates outside unit cell, insufficient surface area, or placement conflicts [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Check site coordinates are within [0,1] range and surface has adequate space [/ERROR_RECOVERY]
        RuntimeError: [ERROR_WHEN] When no adsorption sites are found on the surface [/ERROR_WHEN]
                     [ERROR_DETAILS] Surface structure lacks identifiable binding sites [/ERROR_DETAILS]
                     [ERROR_RECOVERY] Verify slab structure has proper surface termination and geometry [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Does not optimize adsorbate orientation or conformation
    - Cannot handle multiple adsorbates or complex binding modes
    - Limited to simple geometric placement without chemical bonding
    - Does not account for surface relaxation or reconstruction upon adsorption
    [/LIMITATIONS]
    """
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Molecule, Structure

    # Load slab
    slab = Structure.from_str(slab_cif, fmt="cif")

    # Load adsorbate as a Molecule
    try:
        # First try loading directly as molecule
        adsorbate = Molecule.from_str(adsorbate_cif, fmt="xyz")
    except ValueError:
        try:
            # If that fails, try as structure and convert to molecule
            struct = Structure.from_str(adsorbate_cif, fmt="cif")
            adsorbate = Molecule(
                species=struct.species,
                coords=struct.cart_coords.tolist(),
                charge=0,
            )
        except Exception as e:
            raise ValueError(f"Could not parse adsorbate CIF: {e!r}") from e

    finder = AdsorbateSiteFinder(slab)
    if site is None:
        # try to get the first top site
        sites = finder.find_adsorption_sites()
        if sites.get("top"):
            site = list(sites["top"][0].frac_coords)
        else:
            raise ValueError("No top adsorption site found and no site provided.")

    combined_struct = finder.add_adsorbate(adsorbate, site, height)
    return combined_struct.to(fmt="cif")


@tool
def generate_reconstructed_slab(
    bulk_cif: str,
    miller_index: tuple[int, int, int],
    min_slab_size: float,
    min_vacuum_size: float,
    reconstruction_instructions: str,
    return_all_variants: bool = False,
) -> str:
    """[BRIEF] Generate surface slabs with complex reconstructions from bulk structures using detailed instructions. [/BRIEF]

    [DETAILED] This tool creates reconstructed surface slabs that go beyond simple
    terminations to include complex surface arrangements, atomic rearrangements, and
    compositional changes. Surface reconstructions are crucial for understanding real
    surface behavior as many materials undergo significant structural changes when cleaved
    to create surfaces. This tool handles sophisticated reconstruction patterns including
    atomic additions, removals, and rearrangements based on experimental observations. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Best suited for materials known to undergo significant surface rearrangements
    - If the slab has no adsorption site, reconstruction may introduce suitable sites.
    - Essential for accurate modeling of catalytic surfaces with complex structures
    - Recommended for systematic studies of reconstruction effects on surface properties
    - Avoid for simple surface terminations (use enumerate_slabs_text instead)
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the bulk structure and validates Miller indices for the crystal system
    - Interprets complex reconstruction instructions in JSON format
    - Uses ReconstructionGenerator to apply transformation matrices and structural changes
    - Implements atomic additions, removals, and rearrangements as specified
    - Generates either a single reconstruction or all possible variants
    - Validates and optimizes the resulting surface structures
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain bulk structure using get_structure_from_mp_text [/PREREQUISITE]
    2. [CURRENT] Apply this tool with detailed reconstruction instructions [/CURRENT]
    3. [FOLLOW_UP] Use the reconstructed surface for adsorption studies or analysis [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - generate_reconstructed_slab(bulk_cif, (1,1,1), 12, 5, instructions_json, False)
    - generate_reconstructed_slab(bulk_cif, (1,0,0), 15, 10, instructions_json, True)
    [/SYNTACTICAL]

    Args:
        bulk_cif: [BRIEF] CIF string of the bulk crystal structure. [/BRIEF]
                 [DETAILED] A properly formatted CIF string containing the bulk crystal structure
                 that will be used as the starting point for reconstruction. This should be a
                 three-dimensional periodic structure with well-defined symmetry and atomic
                 positions. The bulk structure provides the template for surface generation. [/DETAILED]
                 [SYNTACTIC] Format: "Valid CIF format string with bulk crystal structure" [/SYNTACTIC]
                 [EXAMPLES] Examples: CIF string from Materials Project database [/EXAMPLES]

        miller_index: [BRIEF] Miller indices for the surface orientation. [/BRIEF]
                     [DETAILED] A tuple of three integers specifying the crystallographic plane
                     along which the reconstruction will be performed. These indices must be
                     compatible with the crystal system and determine the base surface geometry
                     before reconstruction modifications are applied. [/DETAILED]
                     [SYNTACTIC] Format: tuple of three integers (h, k, l) [/SYNTACTIC]
                     [EXAMPLES] Examples: (1,1,1) (close-packed), (1,0,0) (square), (1,1,0) (rectangular) [/EXAMPLES]

        min_slab_size: [BRIEF] Minimum slab thickness in Angstroms. [/BRIEF]
                      [DETAILED] The minimum thickness of the slab before reconstruction modifications
                      are applied. This ensures adequate bulk-like behavior in the center of the
                      slab while providing sufficient surface area for reconstruction. Larger
                      values improve accuracy but increase computational cost. [/DETAILED]
                      [SYNTACTIC] Format: positive float representing thickness in Angstroms [/SYNTACTIC]
                      [EXAMPLES] Examples: 12.0 (standard), 15.0 (thick), 10.0 (thin) [/EXAMPLES]

        min_vacuum_size: [BRIEF] Minimum vacuum layer thickness in Angstroms. [/BRIEF]
                        [DETAILED] The minimum vacuum space above the reconstructed surface to
                        prevent interactions between periodic images. This parameter is crucial
                        for accurate surface calculations and should be larger for reconstructions
                        with significant surface protrusions or modifications. [/DETAILED]
                        [SYNTACTIC] Format: positive float representing vacuum thickness in Angstroms [/SYNTACTIC]
                        [EXAMPLES] Examples: 10.0 (standard), 15.0 (large), 5.0 (minimal) [/EXAMPLES]

        reconstruction_instructions: [BRIEF] JSON string containing detailed reconstruction parameters. [/BRIEF]
                                   [DETAILED] A comprehensive JSON string specifying all aspects of
                                   the reconstruction including transformation matrices, atomic
                                   additions/removals, and structural parameters. Must include
                                   required fields like name, transformation_matrix, and modification
                                   instructions. See the tool's source code for detailed format. [/DETAILED]
                                   [SYNTACTIC] Format: "Valid JSON string with reconstruction parameters" [/SYNTACTIC]
                                   [EXAMPLES] Examples: JSON with transformation matrix and atomic modifications [/EXAMPLES]

        return_all_variants: [BRIEF] Whether to return all reconstruction variants. Defaults to False. [/BRIEF]
                           [DETAILED] Controls whether to return a single CIF string (False) or a
                           comprehensive JSON with all possible reconstruction variants and metadata
                           (True). When True, provides detailed information about each variant
                           including structural parameters and characteristics. [/DETAILED]
                           [SYNTACTIC] Format: boolean value (True/False) [/SYNTACTIC]
                           [EXAMPLES] Examples: False (single CIF), True (all variants with metadata) [/EXAMPLES]

    Returns:
        str: [BRIEF] CIF string of reconstructed slab or JSON with all variants depending on return_all_variants. [/BRIEF]
             [DETAILED] Either a single CIF-formatted string containing the reconstructed surface
             structure (if return_all_variants=False) or a comprehensive JSON string with all
             variants, metadata, and structural information (if return_all_variants=True). The
             JSON format includes detailed characterization of each variant. [/DETAILED]
             [EXAMPLES] Example output: CIF string with reconstructed surface or JSON with multiple variants [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When reconstruction parameters are invalid or incompatible [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid JSON format, missing required fields, or incompatible parameters [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify JSON format and ensure all required fields are present [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When reconstruction fails due to structural incompatibilities [/ERROR_WHEN]
                       [ERROR_DETAILS] Invalid transformation matrix, incompatible surface orientation, or atomic placement issues [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Check transformation matrix and atomic modification parameters [/ERROR_RECOVERY]
        ReconstructionError: [ERROR_WHEN] When the reconstruction process fails to generate valid structures [/ERROR_WHEN]
                            [ERROR_DETAILS] Complex reconstruction instructions cannot be implemented [/ERROR_DETAILS]
                            [ERROR_RECOVERY] Simplify reconstruction instructions or try different parameters [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Cannot predict the thermodynamic stability of reconstructions
    - Limited to predefined reconstruction patterns and transformations
    - Does not account for temperature or environmental effects on reconstruction
    - May not capture all possible reconstruction variants for complex systems
    - Requires detailed prior knowledge of reconstruction parameters
    [/LIMITATIONS]
    """
    import json
    from copy import deepcopy

    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.core.surface import ReconstructionGenerator
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    # Parse bulk structure
    try:
        bulk_structure = Structure.from_str(bulk_cif, fmt="cif")
    except Exception as e:
        raise ValueError(f"Invalid CIF format: {e!s}") from e

    # Validate Miller indices are valid for the structure
    try:
        sg = SpacegroupAnalyzer(bulk_structure)
        # This will raise an exception if the Miller indices are invalid
        sg.get_conventional_standard_structure(international_monoclinic=True)
    except Exception as err:
        raise ValueError(
            f"Miller indices {miller_index} are invalid for the given structure"
        ) from err

    # Parse reconstruction instructions
    try:
        instructions = json.loads(reconstruction_instructions)
    except Exception as e:
        raise ValueError(
            f"Invalid JSON format in reconstruction_instructions: {e!s}"
        ) from e

    # Validate reconstruction instructions
    required_fields = {
        "name",
        "transformation_matrix",
        "SlabGenerator_parameters",
        "points_to_remove",
        "points_to_add",
    }
    missing_fields = required_fields - set(instructions.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields in instructions: {missing_fields}")

    # Verify transformation matrix dimensions
    trans_matrix = instructions["transformation_matrix"]
    if (
        not isinstance(trans_matrix, list)
        or len(trans_matrix) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in trans_matrix)
    ):
        raise ValueError("Transformation matrix must be a 3x3 array")

    # Convert to numpy array for easier handling
    trans_matrix = np.array(trans_matrix)

    # Initialize reconstruction generator
    recon_gen = ReconstructionGenerator(
        initial_structure=bulk_structure,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        reconstruction_name=instructions["name"],
    )

    slabgen_params = {}

    if "SlabGenerator_parameters" in instructions:
        slabgen_params.update(instructions["SlabGenerator_parameters"])

    slabgen_params["miller_index"] = miller_index
    slabgen_params["min_slab_size"] = min_slab_size
    slabgen_params["min_vacuum_size"] = min_vacuum_size

    # Set parameters on the reconstruction generator
    recon_gen.slabgen_params = slabgen_params
    recon_gen.trans_matrix = trans_matrix
    recon_gen.reconstruction_json = deepcopy(instructions)

    # Generate reconstructed slabs
    try:
        recon_slabs = recon_gen.build_slabs()
    except Exception as e:
        raise ValueError(f"Reconstruction failed: {e!s}") from e

    if not recon_slabs:
        raise ValueError("Reconstruction failed - no slabs were generated")

    # Return single CIF if requested
    if not return_all_variants:
        return recon_slabs[0].to(fmt="cif")

    output = {
        "variants": [],
        "reconstruction_metadata": {
            "name": instructions["name"],
            "description": instructions.get("description", ""),
            "reference": instructions.get("reference", ""),
            "miller_index": list(miller_index),
            "woods_notation": instructions.get("Woods_notation", ""),
            "spacegroup": instructions.get("spacegroup", {}),
            "transformation_matrix": instructions["transformation_matrix"],
            "total_variants": len(recon_slabs),
            "slab_parameters": {
                "min_slab_size": min_slab_size,
                "min_vacuum_size": min_vacuum_size,
                "SlabGenerator_parameters": {
                    k: v
                    for k, v in slabgen_params.items()
                    if k not in ["miller_index", "min_slab_size", "min_vacuum_size"]
                },
            },
        },
    }

    for idx, slab in enumerate(recon_slabs):
        # Get spacegroup info
        try:
            spacegroup_info = slab.get_space_group_info()
        except Exception:
            spacegroup_info = ("Unknown", None)  # Default if unable to determine

        lattice_params = slab.lattice.parameters
        variant_info = {
            "index": idx,
            "cif": slab.to(fmt="cif"),
            "spacegroup": str(spacegroup_info),
            "formula": slab.composition.reduced_formula,
            "lattice_parameters": {
                "a": lattice_params[0],
                "b": lattice_params[1],
                "c": lattice_params[2],
                "alpha": lattice_params[3],
                "beta": lattice_params[4],
                "gamma": lattice_params[5],
            },
            "num_atoms": len(slab),
            "surface_area": float(slab.lattice.volume / lattice_params[2]),
        }

        if "variant_info" in instructions:
            matching_variants = [
                v for v in instructions.get("variant_info", []) if v.get("index") == idx
            ]
            if matching_variants:
                variant_info.update(
                    {k: v for k, v in matching_variants[0].items() if k != "index"}
                )

        output["variants"].append(variant_info)

    return json.dumps(output, indent=2)


####################################
### Tools relevant for ml training
####################################


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
    - get_bulk_polymorphs_data("TiO2")  # Titanium dioxide polymorphs
    - get_bulk_polymorphs_data("SiO2")  # Silicon dioxide polymorphs
    - get_bulk_polymorphs_data("Al2O3") # Aluminum oxide polymorphs
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
    """
    Query the Materials Project database to find polymorphs for a given composition. This function saves the data to a JSON file to the give path.
    The data includes MP IDs, structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities,
    volumes, number of sites, symmetry, and stability. The results are sorted by energy above hull.

    Args:
        composition: Chemical composition (e.g., 'TiO2')
        save_path: Path to save the JSON file (optional)
        api_key: Materials Project API key (optional if set in environment)

    Returns:
        Path to the saved JSON file containing polymorph data.
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


@tool
def batch_retrieve_polymorphs(
    compositions: list[str],
    max_energy_above_hull: float = 0.5,
    max_per_composition: int = 10,
    save_directory: str = "polymorph_data",
) -> str:
    """
    Retrieve polymorphs for multiple compositions in batch.

    Args:
        compositions: List of chemical compositions
        max_energy_above_hull: Maximum energy above hull to include
        max_per_composition: Maximum polymorphs per composition
        save_directory: Directory to save individual composition files

    Returns:
        JSON string with batch retrieval results
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
    """
    From a JSON string, sort the data based on a given key and return the first element of the specified key.

    Args:
        polymorph_data_json: JSON string containing the data to be sorted.
        sort_key: Key to sort the data by.
        return_key: Key of the first element to return after sorting.

    Returns:
        Value of the specified return_key from the first element after sorting.
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
    """
    Select polymorphs based on a strategy (diverse_energy, most_stable, or diverse_structure).
    diverse_energy selects polymorphs with diverse energies,
    most_stable selects the most stable ones, and diverse_structure selects polymorphs with different space groups.

    Args:
        polymorphs_data: JSON string or file path with polymorph data
        selection_strategy: Strategy for selection ("diverse_energy", "most_stable", "diverse_structure")
        max_polymorphs: Maximum number of polymorphs to select
        energy_threshold: Maximum energy above hull (eV/atom)
        is_path: If True, polymorphs_data is treated as a file path

    Returns:
        JSON string with selected polymorphs
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
    """
    Consolidate multiple polymorph files into a single dataset and save it to a JSON file.

    Args:
        composition_files: Dictionary mapping compositions to file paths
        output_path: Path for consolidated dataset

    Returns:
        JSON string with consolidation results
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
        stats["average_per_composition"] = (
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
def execute_python_code(
    python_code: str,
    input_data: str | None = None,
    save_output_to: str | None = None,
    timeout: int = 300,
) -> str:
    """Executes Python code in a sandboxed environment.

    This function runs a given Python code string as a subprocess, providing
    a secure way to execute dynamic code. It supports injecting input data,
    capturing standard output and errors, and saving structured results to a file.

    Args:
        python_code: A string containing the Python code to be executed.
                    For best results, assign your main output to a variable named
                    'result' or 'output'. The tool will also attempt to capture
                    other user-defined variables as fallback.
        input_data: An optional JSON string. If provided, it will be loaded
                    into a Python variable named `input_data` within the
                    executed script, allowing the script to process external data.
                    Defaults to None.
        save_output_to: An optional file path (string) where the captured
                        execution result will be saved as a JSON file. Defaults to None.
        timeout: The maximum time in seconds the subprocess is allowed to run.
                 If the execution exceeds this limit, a `TimeoutExpired` error
                 will be returned. Defaults to 300 seconds.

    Returns:
        A JSON string detailing the execution outcome. This includes:
        - `success` (bool): True if the process completed without error and
                            returned a 0 exit code, False otherwise.
        - `stdout` (str): The standard output from the executed Python script,
                          excluding the `EXECUTION_RESULT` marker.
        - `stderr` (str): Any error messages or warnings printed to standard error.
        - `return_code` (int): The exit code of the subprocess. A value of 0
                               typically indicates success.
        - `execution_result` (dict): A dictionary containing variables captured
                                    from the executed script. If no suitable variables
                                    are found, this will be an empty dictionary.
        - `saved_to` (str or None): The path where the `execution_result` was
                                    saved, if `save_output_to` was provided
                                    and execution was successful.
        - `error` (str, optional): A descriptive error message if execution failed
                                   or timed out.
        - `traceback` (str, optional): The Python traceback in case of an exception.
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
    """
    Execute a Python script file. The function takes the path to the script,
    optional command-line arguments, and a timeout. It captures the output,
    standard error, and return code of the execution. The results are returned
    as a JSON string.

    Args:
        script_path: Path to the Python script file
        args: Optional list of command-line arguments
        timeout: Timeout in seconds (default 600)
        working_dir: Working directory for execution

    Returns:
        JSON string with execution results
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
def filter_json_with_strategy(
    input_json_path: str,
    output_json_path: str,
    custom_code: str | None = None,
) -> str:
    """
        Filters JSON data from an input file and saves the results to an output file
        using a custom Python filtering logic.

        This function reads a JSON file, applies a custom Python script to filter its
        contents, and then writes the filtered data to a new JSON file. The custom
        filtering code is executed in an isolated environment where the input JSON
        data is available as a variable named 'data'. The filtering logic should
        produce a result in a variable named 'filtered_data'.

        Args:
            input_json_path: The file path to the input JSON data.
            output_json_path: The file path where the filtered JSON data will be saved.
            custom_code: A string containing Python code that defines the filtering logic.
                         This code should expect the input data in a variable named
                         'data' and store its filtered result in a variable named
                         'filtered_data'.

        Returns:
            A JSON string indicating the status of the filtering operation.
            If successful, it includes 'success' (True), 'original_count',
            'filtered_count', 'output_path', and 'reduction_percentage'.
            If unsuccessful, it includes 'success' (False), 'error', and 'details'
            (or 'traceback' for exceptions during file operations).

        Raises:
            FileNotFoundError: If `input_json_path` does not exist.
            json.JSONDecodeError: If the input file is not a valid JSON.
            Exception: For any other errors during file operations or code execution.

        Example:
            # Assuming input json file contains a list of dictionaries like:
            # [
            #   {"material_id": "mp-1143", "band_gap": 5.85},
            #   {"material_id": "mp-752826", "band_gap": 4.17}
            # ]

            input_file = "input.json"
            output_file = "output_filtered.json"

            # Example 1: Filter materials with a band_gap greater than 5.0
            custom_filter_code = \"\"\"
    filtered_data = [item for item in data if item.get("band_gap", 0) > 5.0]
    \"\"\"
            result = filter_json_with_strategy(input_file, output_file, custom_filter_code)
            print(result)
            # Expected output (simplified):
            # {
            #   "success": true,
            #   "original_count": 2,
            #   "filtered_count": 1,
            #   "output_path": "output_filtered.json",
            #   "reduction_percentage": 50.0
            # }
    """
    try:
        with Path(input_json_path).open("r") as f:
            data = json.load(f)

        # The custom_code expects 'data' to be available and should produce 'filtered_data'
        filter_code = f"""
import json

# Input data is available as 'data'
data = {json.dumps(data)}

# Custom filtering logic
{custom_code}

# Result should be stored in 'filtered_data'
output = filtered_data
"""

        exec_result = execute_python_code_given_code(filter_code)
        exec_data = json.loads(exec_result)

        if exec_data["success"] and exec_data["execution_result"]:
            filtered_data = exec_data["execution_result"].get("output", [])
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": "Custom filtering code failed",
                    "details": exec_data,
                },
                indent=2,
            )

        # Save filtered data
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


@tool
def select_polymorphs_with_strategy_to_file(
    polymorphs_data: str,
    save_path: str,
    selection_strategy: str = "diverse_energy",
    max_polymorphs: int = 5,
    energy_threshold: float = 0.5,
    is_path: bool = False,
) -> str:
    """
    Select polymorphs based on a strategy (diverse_energy, most_stable, or diverse_structure). and save results to a file.
    diverse_energy selects polymorphs with diverse energies,
    most_stable selects the most stable ones, and diverse_structure selects polymorphs with different space groups.

    Args:
        polymorphs_data: JSON string or file path with polymorph data
        save_path: Path where to save the selected polymorphs JSON
        selection_strategy: Strategy for selection ("diverse_energy", "most_stable", "diverse_structure")
        max_polymorphs: Maximum number of polymorphs to select
        energy_threshold: Maximum energy above hull (eV/atom)
        is_path: If True, polymorphs_data is treated as a file path

    Returns:
        Path to the saved JSON file
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
def process_slab_ocdata_style(
    slab_cif: str,
    bulk_cif: str,
    min_xy_size: float = 8.0,
    apply_constraints: bool = True,
) -> str:
    """
    Applies ocdata-style processing to a raw slab CIF string:
    1. Tags surface atoms based on height and coordination relative to the bulk.
    2. Tiles the slab to meet a minimum lateral (XY) size.
    3. (Optional) Applies constraints to fix bulk-like atoms (tag=0).

    Requires the original bulk structure for accurate surface atom tagging.

    Args:
        slab_cif: CIF string of the raw slab structure (typically from pymatgen generation).
        bulk_cif: CIF string of the original bulk structure used for coordination reference.
        min_xy_size: Minimum lateral size (Å) the slab should span after tiling.
        apply_constraints: If True, applies FixAtoms constraints to non-surface atoms (tag=0).

    Returns:
        str: CIF string of the processed (tagged, tiled, constrained) slab.
    """
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        # Load structures
        slab_struct_pmg = load_structure(slab_cif)
        slab_atoms_ase = AseAtomsAdaptor.get_atoms(slab_struct_pmg)

        bulk_struct_pmg = load_structure(bulk_cif)
        # Standardize bulk *before* getting ASE atoms for consistent coordination check
        standardized_bulk_pmg = standardize_bulk(bulk_struct_pmg)
        standardized_bulk_ase = AseAtomsAdaptor.get_atoms(standardized_bulk_pmg)

        # 1. Tag Surface Atoms
        tags = find_surface_atoms_with_voronoi(standardized_bulk_ase, slab_atoms_ase)
        slab_atoms_ase.set_tags(tags)

        # 2. Tile the Tagged Slab
        tiled_atoms_ase = tile_atoms(slab_atoms_ase, min_xy_size)

        # 3. Apply Constraints (Optional)
        final_atoms_ase = tiled_atoms_ase
        if apply_constraints:
            final_atoms_ase = set_fixed_atom_constraints(tiled_atoms_ase)

        # 4. Convert back to CIF
        final_struct_pmg = Structure.from_ase_atoms(final_atoms_ase)
        return final_struct_pmg.to(fmt="cif")

    except Exception as e:
        return f"ERROR: Slab processing failed - {e}"


@tool
def get_symmetrically_distinct_miller_indices_from_bulk(
    bulk_structure_path_or_string: str, from_path: bool = False, max_miller: int = 2
) -> list:
    """
    Get symmetrically distinct Miller indices for a bulk structure.

    Args:
        bulk_structure_path_or_string: Path to CIF file or CIF string of the bulk structure
        from_path: Boolean indicating if the input is a file path
        max_miller: Maximum Miller index to consider (1, 2, or 3)

    Returns:
        List of symmetrically distinct Miller indices
    """
    # Load the bulk structure from a CIF file or string

    from pymatgen.core.surface import get_symmetrically_distinct_miller_indices

    bulk_structure = load_structure(bulk_structure_path_or_string, from_path)

    return get_symmetrically_distinct_miller_indices(bulk_structure, max_miller)


@tool
def enumerate_all_possible_miller_indices(max_miller: int = 2) -> list:
    """
    Generate all possible Miller indices up to a given maximum.

    Args:
        max_miller: Maximum Miller index to consider (1, 2, or 3)

    Returns:
        List of tuples representing all possible Miller indices
    """
    mill_list = []
    for i in range(max_miller + 1):
        for j in range(max_miller + 1):
            for k in range(max_miller + 1):
                if i == 0 and j == 0 and k == 0:
                    continue  # Skip (0,0,0)
                mill_list.append((i, j, k))
    return mill_list


@tool
def find_all_unique_slabs_upto_millerindex(
    bulk_structure_path_or_string: str,
    from_path: bool = False,
    max_index: int = 2,
    min_slab_size: float = 8,
    min_vacuum_size: float = 15,
    center_slab: bool = True,
    max_normal_search: int = 10,
) -> str:
    """
    Generates all unique slabs for a given bulk structure up to specified Miller indices.

    Args:
        bulk_structure_path_or_string: Path to CIF file or CIF string of the bulk structure
        from_path: Boolean indicating if the input is a file path
        max_index: Maximum Miller index to consider (1, 2, or 3)
        min_slab_size: Minimum slab thickness in Angstroms
        min_vacuum_size: Minimum vacuum size in Angstroms
        center_slab: If True, centers the slab in the vacuum region
        max_normal_search: Maximum number of normals to search for slab generation

    Returns:
        JSON dictionary with slab IDs as keys and their properties as values.
             Each value contains Miller index, termination, CIF string, area, number of sites, and slab thickness.
    """
    from pymatgen.core.surface import generate_all_slabs

    bulk_structure = load_structure(bulk_structure_path_or_string, from_path)

    slabs = generate_all_slabs(
        bulk_structure,
        max_index=max_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
        max_normal_search=max_normal_search,
    )
    slabs_dict = {}
    for i, slab in enumerate(slabs):
        slab_id = (
            f"{slab.miller_index[0]}{slab.miller_index[1]}{slab.miller_index[2]}_{i}"
        )
        slabs_dict[slab_id] = {
            "miller_index": slab.miller_index,
            "termination": i,
            "cif": slab.to(fmt="cif"),
            "area": slab.surface_area,
            "num_sites": len(slab),
            "slab_thickness": slab.thickness,
        }

    return json.dumps(slabs_dict, indent=2)


@tool
def find_all_unique_slabs_upto_millerindex_to_file(
    bulk_structure_path_or_string: str,
    out_put_path: str,
    from_path: bool = False,
    max_index: int = 2,
    min_slab_size: float = 8,
    min_vacuum_size: float = 15,
    center_slab: bool = True,
    max_normal_search: int = 10,
) -> str:
    """
    Generates all unique slabs for a given bulk structure up to specified Miller indices
    and saves the results to a JSON file.

    Args:
        bulk_structure_path_or_string: Path to CIF file or CIF string of the bulk structure
        out_put_path: Path where to save the JSON output
        from_path: Boolean indicating if the input is a file path
        max_index: Maximum Miller index to consider (1, 2, or 3)
        min_slab_size: Minimum slab thickness in Angstroms
        min_vacuum_size: Minimum vacuum size in Angstroms
        center_slab: If True, centers the slab in the vacuum region
        max_normal_search: Maximum number of normals to search for slab generation

    Returns:
        str: Message indicating where the slabs data has been written.
    """

    from pymatgen.core.surface import generate_all_slabs

    bulk_structure = load_structure(bulk_structure_path_or_string, from_path)

    slabs = generate_all_slabs(
        bulk_structure,
        max_index=max_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        center_slab=center_slab,
        max_normal_search=max_normal_search,
    )
    slabs_dict = {}
    for i, slab in enumerate(slabs):
        slab_id = (
            f"{slab.miller_index[0]}{slab.miller_index[1]}{slab.miller_index[2]}_{i}"
        )
        slabs_dict[slab_id] = {
            "miller_index": slab.miller_index,
            "termination": i,
            "cif": slab.to(fmt="cif"),
            "area": slab.surface_area,
            "num_sites": len(slab),
            "slab_thickness": slab.thickness,
        }

    with Path(out_put_path).open("w") as f:
        json.dump(slabs_dict, f, indent=2)
    return f"Slabs data written to {out_put_path}"


@tool
def enumerate_slabs_for_list_of_miller_index(
    bulk_structure_path_or_string: str,
    from_path: bool = False,
    miller_index_list: list[tuple] | None = None,
    min_slab_size: float = 12,
    min_vacuum_size: float = 5,
) -> str:
    """
    Generates slabs for a given bulk structure and specified Miller indices.

    Args:
        bulk_structure_path_or_string: Path to CIF file or CIF string of the bulk structure
        from_path: Boolean indicating if the input is a file path
        miller_index_list: List of Miller indices to generate slabs for (e.g., [(1, 1, 1), (2, 0, 0)])
        min_slab_size: Minimum slab thickness in Angstroms
        min_vacuum_size: Minimum vacuum size in Angstroms

    Returns:
        str: JSON dictionary: {"slab_0": "<cif_string>", "slab_1": "<cif_string>", ...}
    """
    import json

    from pymatgen.core import Structure
    from pymatgen.core.surface import SlabGenerator

    if miller_index_list is None:
        raise ValueError("Miller indexes should be defined")

    if from_path:
        bulk_structure = Structure.from_file(bulk_structure_path_or_string)
    else:
        bulk_structure = Structure.from_str(bulk_structure_path_or_string, fmt="cif")

    slabs_dict = {}
    for millers in miller_index_list:
        slab_gen = SlabGenerator(
            bulk_structure, millers, min_slab_size, min_vacuum_size
        )
        slabs = slab_gen.get_slabs()  # returns a list of Slab objects
        for i, slab in enumerate(slabs):
            # We use get_orthogonal_c_slab() ensures that the slab lattice is reoriented in c axis for easier adsorption placement.
            # get_sorted_structure() variations in atom ordering that might occur due to how the slab was originally created.
            slab_clean = (
                slab.get_sorted_structure()
                # slab.get_orthogonal_c_slab().get_sorted_structure()
            )
            slabs_dict[f"slab_{i}_{millers}"] = slab_clean.to(fmt="cif")

    return json.dumps(slabs_dict, indent=2)


@tool
def select_slabs_with_strategy(
    slabs_json: str,
    selection_strategy: str = "diverse_miller",
    max_slabs_per_polymorph: int = 3,
) -> str:
    """
    Select slabs based on strategy (diverse_miller, high_coordination, or large_surface).
    diverse_miller selects slabs with different Miller indices,
    high_coordination selects slabs with higher number of surface sites,
    large_surface selects slabs with largest surface areas.

    Args:
        slabs_json: JSON string with slab data
        selection_strategy: Strategy ("diverse_miller", "high_coordination", "large_surface")
        max_slabs_per_polymorph: Maximum slabs to select per polymorph

    Returns:
        JSON string with selected slabs
    """
    slabs = json.loads(slabs_json)

    if selection_strategy == "diverse_miller":
        # Select slabs with different Miller indices
        selected = {}
        miller_indices_seen = set()
        for slab_id, slab_data in slabs.items():
            miller_tuple = tuple(slab_data["miller_index"])
            if (
                miller_tuple not in miller_indices_seen
                and len(selected) < max_slabs_per_polymorph
            ):
                selected[slab_id] = slab_data
                miller_indices_seen.add(miller_tuple)

    elif selection_strategy == "large_surface":
        # Select slabs with largest surface areas
        sorted_slabs = sorted(slabs.items(), key=lambda x: x[1]["area"], reverse=True)
        selected = dict(sorted_slabs[:max_slabs_per_polymorph])

    elif selection_strategy == "high_coordination":
        # Select slabs with higher number of surface sites (proxy for coordination)
        sorted_slabs = sorted(
            slabs.items(), key=lambda x: x[1]["num_sites"], reverse=True
        )
        selected = dict(sorted_slabs[:max_slabs_per_polymorph])

    else:
        # Default: take first max_slabs_per_polymorph
        selected = dict(list(slabs.items())[:max_slabs_per_polymorph])

    return json.dumps(selected, indent=2)


@tool
def select_slabs_with_strategy_to_file(
    slabs_data: str,
    save_path: str,
    selection_strategy: str = "diverse_miller",
    max_slabs_per_polymorph: int = 3,
    is_path: bool = False,
) -> str:
    """
    Select slabs based on strategy (diverse_miller, high_coordination, or large_surface).
    diverse_miller selects slabs with different Miller indices,
    high_coordination selects slabs with higher number of surface sites,
    large_surface selects slabs with largest surface areas.

    Args:
        slabs_data: JSON string or file path with slab data
        save_path: Path where to save the selected slabs JSON
        selection_strategy: Strategy ("diverse_miller", "high_coordination", "large_surface")
        max_slabs_per_polymorph: Maximum slabs to select per polymorph
        is_path: If True, slabs_data is treated as a file path

    Returns:
        Path to the saved JSON file
    """
    # Load slabs data
    if is_path:
        with Path(slabs_data).open("r") as f:
            slabs = json.loads(f.read())
    else:
        slabs = json.loads(slabs_data)

    if selection_strategy == "diverse_miller":
        # Select slabs with different Miller indices
        selected = {}
        miller_indices_seen = set()
        for slab_id, slab_data in slabs.items():
            miller_tuple = tuple(slab_data["miller_index"])
            if (
                miller_tuple not in miller_indices_seen
                and len(selected) < max_slabs_per_polymorph
            ):
                selected[slab_id] = slab_data
                miller_indices_seen.add(miller_tuple)

    elif selection_strategy == "large_surface":
        # Select slabs with largest surface areas
        sorted_slabs = sorted(slabs.items(), key=lambda x: x[1]["area"], reverse=True)
        selected = dict(sorted_slabs[:max_slabs_per_polymorph])

    elif selection_strategy == "high_coordination":
        # Select slabs with higher number of surface sites (proxy for coordination)
        sorted_slabs = sorted(
            slabs.items(), key=lambda x: x[1]["num_sites"], reverse=True
        )
        selected = dict(sorted_slabs[:max_slabs_per_polymorph])

    else:
        # Default: take first max_slabs_per_polymorph
        selected = dict(list(slabs.items())[:max_slabs_per_polymorph])

    # Save to file
    with Path(save_path).open("w") as f:
        json.dump(selected, f, indent=2)

    return save_path


@tool
def generate_adsorbate_slab_configs(
    slab_cif: str, adsorbate_cif: str, adsorption_sites_json: str, height: float = 1.8
) -> str:
    """
    Generate configurations of adsorbates on slab at different adsorption sites.

    Args:
        slab_cif: CIF string of the slab
        adsorbate_cif: CIF string of the adsorbate molecule
        adsorption_sites_json: JSON string with adsorption sites information
        height: Height in Angstroms for initial adsorbate placement

    Returns:
        JSON string mapping site identifiers to adsorbate+slab configurations
    """
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Molecule, Structure

    # Load structures
    slab = Structure.from_str(slab_cif, fmt="cif")

    # Try to load adsorbate as a molecule or structure
    try:
        adsorbate_struct = Structure.from_str(adsorbate_cif, fmt="cif")
        adsorbate = Molecule(
            species=adsorbate_struct.species,
            coords=list(adsorbate_struct.cart_coords),
            charge=0,
        )
    except Exception as e:
        raise ValueError(f"Could not parse adsorbate: {e}") from e

    # Parse adsorption sites
    adsorption_sites = json.loads(adsorption_sites_json)

    # Generate configs for different sites
    configs = {}
    finder = AdsorbateSiteFinder(slab)

    for site_type, sites in adsorption_sites.items():
        # For each site type (top, bridge, hollow), select a few sites
        max_sites = min(3, len(sites))  # Limit to 3 sites per type

        for i in range(max_sites):
            site = sites[i]
            site_coords = site if isinstance(site, list) else list(site)

            try:
                # Add adsorbate to the slab
                ads_slab = finder.add_adsorbate(adsorbate, site_coords, height)

                # Add to configs
                config_id = f"{site_type}_{i}"
                configs[config_id] = {
                    "site_type": site_type,
                    "site_index": i,
                    "site_coords": site_coords,
                    "height": height,
                    "cif": ads_slab.to(fmt="cif"),
                }
            except Exception:
                # Skip sites that cause errors
                continue

    return json.dumps(configs, indent=2)


@tool
def get_mp_surface_properties(material_id: str) -> str:
    """
    Get surface properties for a specific material from the Materials Project. (Material ID, Formula,
    Weighted Surface Energy, Weighted Surface Energy (eV/Å^2), Surface Anisotropy, Shape Factor,
    Has Reconstructed)

    Args:
        material_id: Materials Project ID (e.g., "mp-149")
        api_key: Materials Project API key (optional if set in environment)

    Returns:
        JSON string with surface properties
    """
    from mp_api.client import MPRester

    # Use provided API key or get from environment
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError(
            "Materials Project API key not provided and not found in environment"
        )

    with MPRester(mp_api_key) as mpr:
        # Get surface properties
        try:
            surface_docs = mpr.summary.search(
                material_ids=[material_id],
                fields=[
                    "material_id",
                    "formula_pretty",
                    "weighted_surface_energy",
                    "weighted_surface_energy_EV_PER_ANG2",
                    "surface_anisotropy",
                    "shape_factor",
                    "has_reconstructed",
                ],
            )

            if not surface_docs:
                return json.dumps(
                    {"error": f"No surface properties found for {material_id}"}
                )

            surface_data = []
            for doc in surface_docs:
                data = {
                    "material_id": doc.material_id,
                    "formula_pretty": doc.formula_pretty,
                }

                # Add surface properties if available
                if hasattr(doc, "weighted_surface_energy"):
                    data["weighted_surface_energy"] = doc.weighted_surface_energy
                if hasattr(doc, "weighted_surface_energy_EV_PER_ANG2"):
                    data["weighted_surface_energy_EV_PER_ANG2"] = (
                        doc.weighted_surface_energy_EV_PER_ANG2
                    )
                if hasattr(doc, "surface_anisotropy"):
                    data["surface_anisotropy"] = doc.surface_anisotropy
                if hasattr(doc, "shape_factor"):
                    data["shape_factor"] = doc.shape_factor
                if hasattr(doc, "has_reconstructed"):
                    data["has_reconstructed"] = doc.has_reconstructed

                surface_data.append(data)

            return json.dumps(surface_data, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Error fetching surface properties: {e!s}"})


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
    """
    Prepare tabular dataset for traditional ML models (XGBoost, Random Forest, etc.) from json files create using consolidate polymorph tool.

    Args:
        polymorphs_json_path: Path to polymorphs JSON file
        output_path: Base path for saving dataset files
        target_property: Property to predict (formation_energy_per_atom, energy_above_hull, band_gap)
        feature_engineering: Feature engineering strategy ('basic', 'advanced', 'custom')
        test_split: Fraction for test set
        normalize: Whether to normalize features

    Returns:
        JSON string with dataset preparation results
    """
    import json
    import pickle
    from pathlib import Path

    import numpy as np
    import pandas as pd

    try:
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
            scaler_path = f"{output_path}/scaler.pkl"
            with Path(scaler_path).open("wb") as f:
                pickle.dump(scaler, f)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
            scaler_path = None

        # Save datasets
        train_path = f"{output_path}/train.csv"
        test_path = f"{output_path}/test.csv"

        # Combine features and targets for saving
        train_data = X_train_scaled.copy()
        train_data[target_property] = y_train
        train_data.to_csv(train_path, index=False)

        test_data = X_test_scaled.copy()
        test_data[target_property] = y_test
        test_data.to_csv(test_path, index=False)

        # Save metadata
        metadata_path = f"{output_path}/metadata.json"
        dataset_info = {
            "target_property": target_property,
            "feature_engineering": feature_engineering,
            "normalize": normalize,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(df_features.columns),
            "feature_count": len(df_features.columns),
            "train_path": train_path,
            "test_path": test_path,
            "scaler_path": scaler_path,
            "train_metadata": metadata_train.to_dict("records"),
            "test_metadata": metadata_test.to_dict("records"),
        }

        with Path(metadata_path).open("w") as f:
            json.dump(dataset_info, f, indent=2)

        return json.dumps(
            {
                "success": True,
                "train_path": train_path,
                "test_path": test_path,
                "metadata_path": metadata_path,
                "scaler_path": scaler_path,
                "dataset_info": dataset_info,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        )


@tool
def prepare_neural_network_dataset(
    polymorphs_json_path: str,
    output_path: str,
    target_property: str = "formation_energy_per_atom",
    sequence_features: bool = False,
    embedding_features: bool = True,
    test_split: float = 0.2,
) -> str:
    """
    Prepare dataset for neural network models with embeddings and sequence features.

    Args:
        polymorphs_json_path: Path to polymorphs JSON file
        output_path: Base path for saving dataset files
        target_property: Property to predict
        sequence_features: Whether to create sequence-based features
        embedding_features: Whether to create embedding features
        test_split: Fraction for test set

    Returns:
        JSON string with dataset preparation results
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
    """
    Prepare graph dataset for Graph Neural Networks (GNNs).

    Args:
        polymorphs_json_path: Path to polymorphs JSON file
        output_path: Base path for saving dataset files
        target_property: Property to predict
        cutoff_radius: Cutoff radius for graph edges (Angstroms)
        test_split: Fraction for test set

    Returns:
        JSON string with dataset preparation results
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
def get_mp_thermo_data(material_id: str) -> str:
    """
    Get thermodynamic data for a specific material from the Materials Project. (Material ID, Thermo Type (functional used),
    Formation Energy per Atom, Energy Above Hull, Decomposes To, Is Stable, Energy Type, Uncorrected Energy per Atom)

    Args:
        material_id: Materials Project ID (e.g., "mp-149")

    Returns:
        JSON string with thermodynamic data
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


## NL training models


@tool
def train_xgboost_model(
    train_data_path: str,
    test_data_path: str,
    model_save_path: str,
    target_column: str = "formation_energy_per_atom",
    hyperparameters: dict | None = None,
) -> str:
    """
    Train XGBoost model for formation energy prediction. The input data should be in CSV format with features and target column.

    Args:a
        train_data_path: Path to training CSV file
        test_data_path: Path to test CSV file
        model_save_path: Path to save trained model
        target_column: Name of target column
        hyperparameters: XGBoost hyperparameters

    Returns:
        JSON string with training results and metrics
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
    """
    Evaluate a trained XGBoost model with comprehensive metrics.

    Args:
        model_path: Path to saved XGBoost model
        test_data_path: Path to test data CSV
        target_column: Name of target column
        detailed_analysis: Whether to include detailed analysis

    Returns:
        JSON string with evaluation results
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
    """
    Perform cross-validation on the dataset to assess model stability.

    Args:
        train_data_path: Path to training data CSV
        target_column: Name of target column
        cv_folds: Number of cross-validation folds
        hyperparameters: XGBoost hyperparameters

    Returns:
        JSON string with cross-validation results
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


###

# tools to relax and get energy using mlff
# tool to compute adsorption energy


## OCP - training  creating a dataset and training ML model


@tool
def save_structures_to_db(
    db_path: str,
    structures_json: str,
    structure_type: str = "slab",  # Options: "bulk", "slab", "adsorbate", "adsorbate_slab"
    table_name: str | None = None,
    additional_properties: dict | None = None,
) -> str:
    """
    Save structures (bulk, slabs, adsorbates, or adsorbate+slab) to a SQLite database with appropriate schema.

    Args:
        db_path: Path to SQLite database file.
        structures_json: JSON string with structure data.
        structure_type: Type of structures being saved ("bulk", "slab", "adsorbate", or "adsorbate_slab").
        table_name: Override default table name (default is determined by structure_type).
        additional_properties: Dictionary of additional properties to save for all structures.

    Returns:
        Status string.
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


@tool
def add_descriptor_column_to_db(
    db_path: str,
    table_name: str,
    descriptor_type: str,
    descriptor_function_code: str,
    batch_size: int = 100,
    dependencies: list | None = None,
) -> str:
    """
    Add one or more descriptor columns to a database table and compute values efficiently.
    Specialized for ML feature calculation with support for batch processing.

    Args:
        db_path: Path to database.
        table_name: Table name to modify.
        descriptor_type: Type of descriptor to add (e.g., "coordination", "d_band", "bond_length").
        descriptor_function_code: String of a Python function that accepts a row dict and returns
                                 a dict mapping column names to values.
        batch_size: Number of rows to process in each batch for memory efficiency.
        dependencies: List of other tables this calculation depends on (for joining data).

    Returns:
        Status string with descriptor statistics.
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
    """
    Generate a ML-ready dataset in the specified format with proper train/validation splits.

    Args:
        db_path: Path to SQLite database
        table_name: Table containing the data
        feature_columns: List of column names to use as features
        target_column: Column name for the prediction target
        output_format: Format for the dataset (csv, json, npz)
        output_path: Base path/filename for the output files
        validation_split: Fraction to use for validation set
        normalize_features: Whether to normalize features
        include_metadata: Whether to include feature metadata

    Returns:
        Status string with information about the generated dataset
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


def create_tools() -> dict[str, Tool]:
    """Create all available tools"""
    return {
        "get_structure_from_mp_text": get_structure_from_mp_text,
        "create_slab_from_structure_text": create_slab_from_structure_text,
        "enumerate_slabs_text": enumerate_slabs_text,
        "choose_slab_text": choose_slab_text,
        "get_adsorption_sites_text": get_adsorption_sites_text,
        "choose_adsorption_site_text": choose_adsorption_site_text,
        "add_adsorbate_to_slab_text": add_adsorbate_to_slab_text,
        "generate_reconstructed_slab": generate_reconstructed_slab,
        # OCP hard
        "get_bulk_polymorphs_data": get_bulk_polymorphs_data,
        "sort_and_get_first_from_json": sort_and_get_first_from_json,
        "process_slab_ocdata_style": process_slab_ocdata_style,
        "get_symmetrically_distinct_miller_indices_from_bulk": get_symmetrically_distinct_miller_indices_from_bulk,
        "enumerate_all_possible_miller_indices": enumerate_all_possible_miller_indices,
        "find_all_unique_slabs_upto_millerindex": find_all_unique_slabs_upto_millerindex,
        "find_all_unique_slabs_upto_millerindex_to_file": find_all_unique_slabs_upto_millerindex_to_file,
        "enumerate_slabs_for_list_of_miller_index": enumerate_slabs_for_list_of_miller_index,
        "generate_adsorbate_slab_configs": generate_adsorbate_slab_configs,
        "get_mp_surface_properties": get_mp_surface_properties,
        "get_mp_thermo_data": get_mp_thermo_data,
        # OCP training
        "save_structures_to_db": save_structures_to_db,
        "add_descriptor_column_to_db": add_descriptor_column_to_db,
    }


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
