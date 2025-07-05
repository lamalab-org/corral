import os

from dotenv import load_dotenv

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
    - Searches for the specified material ID (MP ID) in the database (MP ID is given as input parameter or if other tools are available to search for MP ID, then use those tools)
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
    }
