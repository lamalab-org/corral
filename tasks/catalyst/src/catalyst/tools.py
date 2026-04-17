import json
import os
from pathlib import Path

from catalyst.tool_utils import (
    load_structure,
)
from dotenv import load_dotenv
from mp_api.client import MPRester

from corral.backend.tool import Tool, tool

if "MP_API_KEY" not in os.environ:
    load_dotenv("../.env")


# utility function
def get_bulk_polymorphs_data_func(composition: str) -> str:
    """
    Query the Materials Project database to find polymorphs for a given composition.
    This function returns a JSON string containing polymorph data including MP IDs, structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities, volumes, number of sites,symmetry, and stability. The results are sorted by energy above hull.

    Args:
        composition: Chemical composition (e.g., 'TiO2')
        api_key: Materials Project API key (optional if set in environment)

    Returns:
        JSON string containing polymorph data including MP IDs, structures (CIF), energies above hull, formation_energy_per_atom, band gaps, densities, volumes, number of sites, symmetry, and stability. (sorted by energy above hull)
    """

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


####################
# Tools that will return text strings - Catalyst environment
####################


@tool
def get_structure_from_mp_text(mp_id: str) -> str:
    """[BRIEF] Retrieve a pymatgen structure from Materials Project using its API and return CIF content as text. [/BRIEF]
    [DETAILED] This tool connects to the Materials Project database to download crystal structure data for a given material ID.
    It retrieves the structure object and converts it to CIF (Crystallographic Information File) format, which is the standard format for storing crystal structure information.
    CIF is then returned as string [/DETAILED]
    [PROCEDURAL] When to use this tool:
    - Use when you need to retrieve a bulk crystal structure from the Materials Project database
    - Best suited for materials with known MP IDs
    - Usually first step in simulation workflows
    - Recommended for obtaining a crystal structure for preparing bulk structures, supercells, bulk cells, slabs etc.
    - Avoid when you need multiple structures
    [/PROCEDURAL]
    [CONTEXTUAL] How this tool works:
    - Connects to Materials Project API using authentication key (which is already provided in the environment)
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
        mp_id: [ARGS_BRIEF] Materials Project identifier string. [/ARGS_BRIEF]
               [ARGS_DETAILED] The unique identifier used by Materials Project to catalog materials.
               Should be in the format "mp-XXXXX" where XXXXX is a numerical ID.
               This ID corresponds to a specific material entry in the Materials Project database. [/ARGS_DETAILED]
               [ARGS_SYNTACTICAL] "mp-" followed by digits (e.g., "mp-149", "mp-20066") [/ARGS_SYNTACTICAL]
               [ARGS_EXAMPLES] "mp-149" (Silicon), "mp-20066" (CO2), "mp-2" (Li) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF content string containing the crystal structure data. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A properly formatted CIF (Crystallographic Information File) string containing all necessary information about the crystal structure including lattice parameters, atomic positions, space group, and symmetry operations.
             This format is widely compatible with crystallographic software and other structure analysis tools. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "\n_chemical_formula_structural Si\n_cell_length_a 5.468..." [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool generates a surface slab by cleaving a bulk crystal structure along a specified crystallographic plane.
    It creates a two-dimensional periodic surface model suitable for surface chemistry calculations, catalysis studies, and adsorption analysis.
    The tool automatically handles the creation of vacuum space above the surface and ensures proper termination of the crystal structure.
    This is essential for computational surface science studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to create a single slab from a bulk structure with known Miller indices
    - Best suited for straightforward surface generation without need for multiple terminations
    - Recommended when you have specific requirements for slab thickness and vacuum spacing
    - Avoid when you need to explore multiple possible surface terminations
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
    1. [PREREQUISITE] First obtain bulk structure using get_structure_from_mp_text or using other tools that return single structure CIF [/PREREQUISITE]
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
        structure_cif: [ARGS_BRIEF] CIF content string of the bulk crystal structure. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A properly formatted CIF string containing the bulk crystal structure data including lattice parameters, atomic positions, and space group information.
                      This structure will be cleaved to create the surface. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        miller_index: [ARGS_BRIEF] Miller indices for the surface plane. Defaults to (1,1,1). [/ARGS_BRIEF]
                     [ARGS_DETAILED] A tuple of three integers specifying the crystallographic plane along which the structure will be cleaved.
                     These indices define the surface orientation and determine the atomic arrangement at the surface.
                     Common choices include (1,1,1), (1,0,0), and (1,1,0) for different surface orientations. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] tuple of three integers (h, k, l) [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] (1,1,1), (1,0,0), (1,1,0) [/ARGS_EXAMPLES]
        min_slab_size: [ARGS_BRIEF] Minimum slab thickness in Angstroms. Defaults to 12. [/ARGS_BRIEF]
                      [ARGS_DETAILED] The minimum thickness of the slab in the direction perpendicular to the surface plane.
                      This parameter ensures that the slab has sufficient bulk-like character in the center while exposing the desired surface.
                      Larger values provide more accurate representation of bulk properties but increase computational cost. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] positive integer representing thickness in Angstroms [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] 12, 15, 8[/ARGS_EXAMPLES]
        min_vacuum_size: [ARGS_BRIEF] Minimum vacuum spacing in Angstroms. Defaults to 5. [/ARGS_BRIEF]
                        [ARGS_DETAILED] The minimum vacuum space above the surface to prevent interactions between periodic images in surface calculations.
                        This parameter is crucial for accurate surface energy calculations and adsorption studies.
                        Larger values reduce spurious interactions but increase computational requirements. [/ARGS_DETAILED]
                        [ARGS_SYNTACTICAL] positive integer representing vacuum thickness in Angstroms [/ARGS_SYNTACTICAL]
                        [ARGS_EXAMPLES] 5 (minimal), 10 (standard), 15 (large) [/ARGS_EXAMPLES]
        primitive: [ARGS_BRIEF] Whether to create a primitive cell slab. Defaults to True. [/ARGS_BRIEF]
                  [ARGS_DETAILED] Controls whether to use the primitive cell or conventional cell for slab generation.
                  Primitive cells have the minimum number of atoms while maintaining the essential symmetry, leading to smaller, more efficient computational models.
                  Setting to False uses the conventional cell which may be larger but more intuitive. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] boolean value (True/False) [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] True, False [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF content string of the generated surface slab. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A CIF-formatted string containing the surface slab structure with the specified Miller indices, thickness, and vacuum spacing.
             The structure is oriented with the surface normal along the c-axis and includes all necessary crystallographic information for surface calculations. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] CIF string with slab structure having surface atoms and vacuum region [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool generates all possible surface terminations for a given bulk crystal structure along specified Miller indices.
    Unlike creating a single slab, this tool explores different ways to terminate the surface, which is crucial for materials with complex structures or multiple chemically distinct layers.
    Each termination represents a different surface chemistry and reactivity, making this tool essential for comprehensive surface studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to explore all possible surface terminations for a material
    - Best suited for complex materials with multiple distinct atomic layers
    - Recommended for systematic surface studies and comparing different surface chemistries
    - Avoid when you only need a single, well-defined surface
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
        bulk_cif: [ARGS_BRIEF] Bulk crystal structure in CIF string format. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A properly formatted CIF string containing the bulk crystal structure from which surface slabs will be generated. This should be a three-dimensional periodic structure with well-defined atomic positions and lattice parameters.
                 The structure will be analyzed to determine all possible surface terminations. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        miller_index: [ARGS_BRIEF] Miller indices for surface orientation. Defaults to (1,1,1). [/ARGS_BRIEF]
                     [ARGS_DETAILED] A tuple of three integers specifying the crystallographic plane along which all surface terminations will be generated. This determines the surface orientation but allows for different terminations along the same plane.
                     Different Miller indices will produce different surface structures and properties. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] tuple of three integers (h, k, l) [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] (1,1,1), (1,0,0), (1,1,0) [/ARGS_EXAMPLES]
        min_slab_size: [ARGS_BRIEF] Minimum slab thickness in Angstroms. Defaults to 12. [/ARGS_BRIEF]
                      [ARGS_DETAILED] The minimum thickness of each slab in the direction perpendicular to the surface plane.
                      This ensures that all generated slabs have sufficient bulk-like character while exposing different surface terminations.
                      Affects both the structural accuracy and computational requirements. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] positive float representing thickness in Angstroms [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] 10.0 (for thin slab), 12.0 (standard), 15.0 (for thick slab) [/ARGS_EXAMPLES]
        min_vacuum_size: [ARGS_BRIEF] Minimum vacuum layer thickness in Angstroms. Defaults to 5. [/ARGS_BRIEF]
                        [ARGS_DETAILED] The minimum vacuum space above each surface to prevent interactions between periodic images.
                        This parameter is applied to all generated slabs and is crucial for accurate surface calculations.
                        Larger values reduce spurious interactions but increase computational cost. [/ARGS_DETAILED]
                        [ARGS_SYNTACTICAL] positive float representing vacuum thickness in Angstroms [/ARGS_SYNTACTICAL]
                        [ARGS_EXAMPLES] 5.0, 10.0, 15.0 [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string mapping slab indices to their CIF representations. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing a dictionary where keys are slab identifiers (e.g., "slab_0", "slab_1") and values are the corresponding CIF strings for each surface termination.
             This format allows easy selection and comparison of different surface terminations. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "{"slab_0": "CIF content...", "slab_1": "CIF content...", ...}" [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool selects one surface slab from a collection of enumerated slabs based on its index number.
    It's designed to work with the output from enumerate_slabs_text, allowing users to choose a specific surface termination for further analysis.
    This selection process is crucial for focusing on the most relevant or interesting surface termination for catalysis or adsorption studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after enumerate_slabs_text to select a specific surface termination
    - Best suited for systematic exploration of different surface terminations
    - Could be useful for workflows that require a single slab for adsorption or catalysis studies
    - Recommended when you need to compare results from different surface terminations
    - One can randomly pick index to select a slab from the enumerated list if they want to randomly pick a slab
    - Avoid when you only need one slab
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
    [
    `choose_slab_text(slabs_json, 0)`,  # Select first slab
    `choose_slab_text(slabs_json, 1)`,  # Select second slab
    `choose_slab_text(slabs_json)`       # Select first slab (default)
    ]
    [/SYNTACTICAL]

    Args:
        slabs_json: [ARGS_BRIEF] JSON string mapping slab keys to CIF strings. [/ARGS_BRIEF]
                   [ARGS_DETAILED] A JSON-formatted string containing a dictionary where keys are slab identifiers (e.g., "slab_0", "slab_1") and values are the corresponding CIF strings.
                   This should be the output from enumerate_slabs_text tool.
                   The JSON structure must be valid and contain at least one slab entry. [/ARGS_DETAILED]
                   [ARGS_SYNTACTICAL] 'Valid JSON string with "slab_X" keys and CIF string values' [/ARGS_SYNTACTICAL]
                   [ARGS_EXAMPLES] "{"slab_0": "CIF content...", "slab_1": "CIF content..."}" [/ARGS_EXAMPLES]
        index: [ARGS_BRIEF] Index of the slab to select. Defaults to 0. [/ARGS_BRIEF]
              [ARGS_DETAILED] The numerical index of the slab to select from the JSON dictionary.
              This corresponds to the enumeration order from enumerate_slabs_text, where index 0 is the first slab, index 1 is the second, and so on.
              The tool will look for a key named "slab_{index}" in the JSON dictionary. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] non-negative integer [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] 0 (first slab), 1 (second slab), 2 (third slab) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF string for the selected slab. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A properly formatted CIF string containing the structure data for the selected slab.
             This includes atomic positions, lattice parameters, and all necessary crystallographic information.
             The CIF can be used directly with other structure analysis tools. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] CIF string with selected slab structure [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool analyzes a surface slab structure to identify and classify potential adsorption sites where molecules can bind.
    It uses geometric and chemical analysis to determine different types of binding sites such as top sites (above surface atoms), bridge sites (between two atoms), and hollow sites (in multi-atom depressions).
    This analysis is fundamental for understanding surface reactivity and designing catalysts. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to identify all possible adsorption sites on a surface
    - Best suited for systematic studies of surface reactivity and catalysis
    - Essential for understanding how molecules interact with surfaces
    - Recommended before placing adsorbates to understand binding options
    - Avoid when you already know the specific binding site coordinates
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the slab CIF structure to identify surface atoms
    - Uses AdsorbateSiteFinder from pymatgen to geometrically analyze the surface topology
    - Classifies sites based on coordination environment (ontop, bridge, hollow)
    - Calculates fractional coordinates for each potential binding site
    - Returns sites organized by type in a JSON format for easy selection
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] First obtain a slab structure using choose_slab_text or create_slab_from_structure_text [/PREREQUISITE]
    2. [CURRENT] Apply this tool to identify all adsorption sites on the surface [/CURRENT]
    3. [FOLLOW_UP] Use choose_adsorption_site_text to select a specific site for adsorbate placement [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    [
        `get_adsorption_sites_text(slab_cif_string)`,  # Analyze slab structure
        `get_adsorption_sites_text(output_from_choose_slab_text)`,  # Use output
        `get_adsorption_sites_text(create_slab_from_structure_text)`,  # From slab creation
        `get_adsorption_sites_text("CIF string of a slab")`,  # Direct
        `get_adsorption_sites_text("CIF string with surface atoms")`,  # Example with specific slab
    ]
    [/SYNTACTICAL]

    Args:
        slab_cif: [ARGS_BRIEF] CIF string of the surface slab structure. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A properly formatted CIF string containing the surface slab structure with atomic positions, lattice parameters, and surface geometry.
                 This should be a two-dimensional periodic structure with a well-defined surface and vacuum region.
                 The structure is analyzed to identify potential adsorption sites. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string containing classified adsorption sites with fractional coordinates. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing a dictionary where keys are site types (e.g., "top", "bridge", "hollow") and values are lists of fractional coordinates for each site of that type.
             Each coordinate is a list of three numbers [x, y, z] representing the fractional position within the unit cell. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "{"ontop": [[0.0, 0.0, 0.9], [0.5, 0.5, 0.9]], "bridge": [[0.25, 0.25, 0.85]]}" [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool selects one specific adsorption site from a collection of classified sites based on the site type (ontop, bridge, hollow) and index within that type.
    This selection is crucial for systematic studies of different binding environments and their effects on adsorption energetics. [/DETAILED]

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
    [
        `choose_adsorption_site_text(sites_json, "ontop", 0)`,
        `choose_adsorption_site_text(sites_json, "bridge", 1)`,
        `choose_adsorption_site_text(sites_json, "hollow", 0)`,
        `choose_adsorption_site_text(sites_json, "ontop", 1)`,
        `choose_adsorption_site_text(sites_json, "bridge", 0)`
    ]
    [/SYNTACTICAL]

    Args:
        adsorption_sites_json: [ARGS_BRIEF] JSON string mapping site types to lists of fractional coordinates. [/ARGS_BRIEF]
                              [ARGS_DETAILED] A JSON-formatted string containing a dictionary where keys are site types (e.g., "ontop", "bridge", "hollow") and values are lists of fractional coordinates.
                              This should be the output from get_adsorption_sites_text.
                              Each coordinate is a list of three numbers representing position within the unit cell. [/ARGS_DETAILED]
                              [ARGS_SYNTACTICAL] 'Valid JSON string with site type keys and coordinate list values' [/ARGS_SYNTACTICAL]
                              [ARGS_EXAMPLES] "{"ontop": [[0.0, 0.0, 0.9]], "bridge": [[0.25, 0.25, 0.85]]}" [/ARGS_EXAMPLES]
        site_type: [ARGS_BRIEF] Type of adsorption site to select. [/ARGS_BRIEF]
                  [ARGS_DETAILED] The type of binding site to select from the available options.
                  Common types include "ontop" (above surface atoms), "bridge" (between two atoms), and  "hollow" (in multi-atom depressions).
                  The type must exist in the JSON dictionary and determines the coordination environment of the selected site. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] string matching available site types [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] "ontop" (on-top), "bridge" (between atoms), "hollow" (in depression) [/ARGS_EXAMPLES]
        index: [ARGS_BRIEF] Index of the site within the specified type. Defaults to 0. [/ARGS_BRIEF]
              [ARGS_DETAILED] The numerical index of the site to select from the list of sites of the specified type.
              Index 0 selects the first site, index 1 the second, and so on.
              The index must be within the range of available sites for the specified type. [/ARGS_DETAILED]
              [ARGS_SYNTACTICAL] non-negative integer [/ARGS_SYNTACTICAL]
              [ARGS_EXAMPLES] 0 (first site), 1 (second site), 2 (third site) [/ARGS_EXAMPLES]

    Returns:
        list[float]: [RETURNS_BRIEF] Fractional coordinates of the selected adsorption site. [/RETURNS_BRIEF]
                    [RETURNS_DETAILED] A list of three floating-point numbers representing the fractional coordinates [x, y, z] of the selected adsorption site within the unit cell.
                    These coordinates can be used directly for adsorbate placement and represent the optimal binding position for the specified site type. [/RETURNS_DETAILED]
                    [RETURNS_EXAMPLES] [0.0, 0.0, 0.9] or [0.25, 0.25, 0.85] [/RETURNS_EXAMPLES]

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

    [DETAILED] This tool combines a surface slab with an adsorbate molecule by placing the adsorbate at a specific binding site on the surface.
    It handles the geometric placement of the molecule at the correct height above the surface and ensures proper structural integration.
    This is essential for creating realistic surface-adsorbate systems for computational studies of catalysis, adsorption energetics, and surface reactivity. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to create a surface-adsorbate system
    - Recommended for systematic studies of different binding sites or orientations
    - Avoid when you need complex multi-adsorbate systems or surface reconstructions
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the slab CIF structure to identify the surface geometry
    - Loads the adsorbate as a molecular structure (handles both XYZ and CIF formats)
    - Uses AdsorbateSiteFinder from pymatgen to place the adsorbate at the specified site
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
    [
        `add_adsorbate_to_slab_text(slab_cif, adsorbate_cif)`,  # Default height and auto-select site
        `add_adsorbate_to_slab_text(slab_cif, adsorbate_cif, 2.0)`,  # Specify height only
        `add_adsorbate_to_slab_text(slab_cif, adsorbate_cif, 1.5, [0.0, 0.0, 0.9])`,  # Specify height
        `add_adsorbate_to_slab_text(slab_cif, adsorbate_cif, site=[0.5, 0.5, 0.9])`,  # Auto-select height
        `add_adsorbate_to_slab_text(slab_cif, adsorbate_cif, 2.0, None)`  # Specify height, auto-select site
    ]
    [/SYNTACTICAL]

    Args:
        slab_cif: [ARGS_BRIEF] CIF string of the surface slab structure. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A properly formatted CIF string containing the surface slab structure on which the adsorbate will be placed.
                 This should be a two-dimensional periodic structure with a well-defined surface and vacuum region.
                 The slab provides the substrate for molecular adsorption. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        adsorbate_cif: [ARGS_BRIEF] CIF string of the adsorbate molecule structure. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A CIF or XYZ formatted string containing the molecular structure of the adsorbate to be placed on the surface.
                      This can be a small molecule like CO2, H2O, or more complex organic molecules.
                      The tool will attempt to parse both CIF and XYZ formats automatically. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] "Valid CIF or XYZ format string with molecular structure" [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] CIF string for molecules [/ARGS_EXAMPLES]
        height: [ARGS_BRIEF] Height in Angstroms above the surface for adsorbate placement. Defaults to 2.0. [/ARGS_BRIEF]
               [ARGS_DETAILED] The vertical distance above the surface at which the adsorbate will be placed.
               This parameter controls the initial separation between the adsorbate and the surface atoms.
               Typical values range from 1.5 to 3.0 Å depending on the molecular size and expected binding interaction. [/ARGS_DETAILED]
               [ARGS_SYNTACTICAL] positive float representing distance in Angstroms [/ARGS_SYNTACTICAL]
               [ARGS_EXAMPLES] 1.5 (close to slab), 2.0, 2.5 (distant from molecule) [/ARGS_EXAMPLES]
        site: [ARGS_BRIEF] Optional fractional coordinates for adsorbate placement. [/ARGS_BRIEF]
             [ARGS_DETAILED] A list of three floating-point numbers representing the fractional coordinates [x, y, z] where the adsorbate should be placed
             If not provided, the tool will automatically select the first available top site.
             These coordinates should typically come from choose_adsorption_site_text output. [/ARGS_DETAILED]
             [ARGS_SYNTACTICAL] list of three floats [x, y, z] or None [/ARGS_SYNTACTICAL]
             [ARGS_EXAMPLES] [0.0, 0.0, 0.9], [0.5, 0.5, 0.9], None (auto-select) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF string of the combined surface-adsorbate structure. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A properly formatted CIF string containing the combined structure with the adsorbate placed on the surface at the specified position and height.
             This structure includes both the original slab atoms and the adsorbate atoms, properly integrated into a single periodic structure. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] CIF string with both slab and adsorbate atoms [/RETURNS_EXAMPLES]

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

    import numpy as np
    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Molecule, Structure

    # Load slab
    slab = Structure.from_str(slab_cif, fmt="cif")

    # Load adsorbate as a Molecule
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

    # Convert the fractional site coordinate to Cartesian coordinates
    cart_coords = slab.lattice.get_cartesian_coords(site)

    # Get the surface normal vector (c-axis in a standard slab)
    c_vector = slab.lattice.matrix[2]
    c_unit_vector = c_vector / np.linalg.norm(c_vector)

    # Add the height along the surface normal to get the final placement coordinate
    final_cart_coords = cart_coords + height * c_unit_vector

    # Call add_adsorbate with the final Cartesian coordinates.
    combined_struct = finder.add_adsorbate(adsorbate, final_cart_coords)

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

    [DETAILED] This tool creates reconstructed surface slabs that go beyond simple terminations to include complex surface arrangements, atomic rearrangements, and compositional changes.
    Surface reconstructions are crucial for understanding real surface behavior as many materials undergo significant structural changes when cleaved to create surfaces.
    This tool handles sophisticated reconstruction patterns including atomic additions, removals, and rearrangements based on experimental observations. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Best suited for materials known to undergo significant surface rearrangements
    - If the slab has no adsorption site, reconstruction may introduce suitable sites.
    - Essential for accurate modeling of catalytic surfaces with complex structures
    - Recommended for systematic studies of reconstruction effects on surface properties
    - Avoid for simple surface terminations
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Parses the bulk structure and validates Miller indices for the crystal system
    - Interprets complex reconstruction instructions in JSON format
    - Uses ReconstructionGenerator from pymatgen to apply transformation matrices and structural changes
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
    [
        `generate_reconstructed_slab(bulk_cif, (1,1,1), 12.0, 5.0, instructions_json)`,  # Basic reconstruction
        `generate_reconstructed_slab(bulk_cif, (1,0,0), 15.0, 10.0, instructions_json, True)`,  # All variants
        `generate_reconstructed_slab(bulk_cif, (1,1,0), 10.0, 5.0, instructions_json)`,  # Rectangular slab
        `generate_reconstructed_slab(bulk_cif, (1,2,1), 20.0, 15.0, instructions_json, False)`,  # Complex reconstruction
        `generate_reconstructed_slab(bulk_cif, (2,0,0), 25.0, 10.0, instructions_json, True)`,  # Thick slab
    ]
    [/SYNTACTICAL]

    Args:
        bulk_cif: [ARGS_BRIEF] CIF string of the bulk crystal structure. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A properly formatted CIF string containing the bulk crystal structure that will be used as the starting point for reconstruction.
                 This should be a three-dimensional periodic structure with well-defined symmetry and atomic positions.
                 The bulk structure provides the template for surface generation. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        miller_index: [ARGS_BRIEF] Miller indices for the surface orientation. [/ARGS_BRIEF]
                     [ARGS_DETAILED] A tuple of three integers specifying the crystallographic plane along which the reconstruction will be performed.
                     These indices must be compatible with the crystal system and determine the base surface geometry before reconstruction modifications are applied. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] tuple of three integers (h, k, l) [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] (1,1,1) (close-packed), (1,0,0) (square), (1,1,0) (rectangular) [/ARGS_EXAMPLES]
        min_slab_size: [ARGS_BRIEF] Minimum slab thickness in Angstroms. [/ARGS_BRIEF]
                      [ARGS_DETAILED] The minimum thickness of the slab before reconstruction modifications are applied.
                      This ensures adequate bulk-like behavior in the center of the slab while providing sufficient surface area for reconstruction.
                      Larger values improve accuracy but increase computational cost. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] positive float representing thickness in Angstroms [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] 12.0 (standard), 15.0 (thick), 10.0 (thin) [/ARGS_EXAMPLES]
        min_vacuum_size: [ARGS_BRIEF] Minimum vacuum layer thickness in Angstroms. [/ARGS_BRIEF]
                        [ARGS_DETAILED] The minimum vacuum space above the reconstructed surface to prevent interactions between periodic images.
                        This parameter is crucial for accurate surface calculations and should be larger for reconstructions with significant surface protrusions or modifications. [/ARGS_DETAILED]
                        [ARGS_SYNTACTICAL] positive float representing vacuum thickness in Angstroms [/ARGS_SYNTACTICAL]
                        [ARGS_EXAMPLES] 10.0 (standard), 15.0 (large), 5.0 (minimal) [/ARGS_EXAMPLES]
        reconstruction_instructions: [ARGS_BRIEF] JSON string containing detailed reconstruction parameters. [/ARGS_BRIEF]
                                   [ARGS_DETAILED] A comprehensive JSON string specifying all aspects of the reconstruction including transformation matrices, atomic additions/removals, and structural parameters.
                                   Must include required fields like name, transformation_matrix, and modification instructions[/ARGS_DETAILED]
                                   [ARGS_SYNTACTICAL] "Valid JSON string with reconstruction parameters" [/ARGS_SYNTACTICAL]
                                   [ARGS_EXAMPLES] JSON with transformation matrix and atomic modifications [/ARGS_EXAMPLES]
        return_all_variants: [ARGS_BRIEF] Whether to return all reconstruction variants. Defaults to False. [/ARGS_BRIEF]
                           [ARGS_DETAILED] Controls whether to return a single CIF string (False) or a comprehensive JSON with all possible reconstruction variants and metadata (True).
                           When True, provides detailed information about each variant including structural parameters and characteristics. [/ARGS_DETAILED]
                           [ARGS_SYNTACTICAL] boolean value (True/False) [/ARGS_SYNTACTICAL]
                           [ARGS_EXAMPLES] False (single CIF), True (all variants with metadata) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] CIF string of reconstructed slab or JSON with all variants depending on return_all_variants. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] Either a single CIF-formatted string containing the reconstructed surface structure (if return_all_variants=False) or a comprehensive JSON string with all variants, metadata, and structural information (if return_all_variants=True).
             The JSON format includes detailed characterization of each variant. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/RETURNS_EXAMPLES]

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
### Tools for ablations
####################################


@tool
def get_bulk_polymorphs_data(composition: str) -> str:
    """[BRIEF] Query Materials Project database to find all polymorphs for a given chemical composition. [/BRIEF]

    [DETAILED] This tool retrieves comprehensive polymorph data from the Materials Project database for a specific chemical composition. Polymorphs are different crystal structures with the same chemical formula but different atomic arrangements, leading to distinct physical and chemical properties. This tool could be relevant for retrieving structures of the same compoisition. Apart from structure for each polymorph Materials Project ID (MP ID), CIF structure, energy above hull, formation energy per atom, band gap, density, volume, number of sites, space group, and stability information is also retrieved. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need to explore all known structural variants of a single material composition
    - Suitable for identifying thermodynamically stable and metastable phases and other properties like band gap, density, volume, number of sites, space group of the structure.
    - Recommended for retrieving structure and Materials Project ID (MP ID), CIF structure, energy above hull, formation energy per atom, band gap, density, volume, number of sites, space group of one single composition
    - Avoid when you only need a single, well-known structure
    - Avoid when you need data for multiple compositions
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
        composition: [ARGS_BRIEF] Chemical composition formula. [/ARGS_BRIEF]
                    [ARGS_DETAILED] Chemical formula specifying the composition for which polymorphs should be retrieved. Should follow standard chemical notation with element symbols. The tool will find all known crystal structures with this exact composition in the Materials Project database. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] "Standard chemical formula (e.g., TiO2, Al2O3, CaTiO3)" [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "TiO2" (rutile, anatase, brookite), "SiO2" (quartz, cristobalite), "Fe2O3" (hematite, maghemite) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string containing comprehensive polymorph data sorted by stability. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing a list of dictionaries, each representing a polymorph with properties including Materials Project ID, CIF structure, energy above hull, formation energy per atom, band gap, density, volume, number of sites, space group, and stability information. Results are sorted by energy above hull for easy identification of the most stable phases. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] '[{"material_id": "mp-2657", "cif": "...", "energy_above_hull": 0.0, "formation_energy_per_atom": -4.2, ...}]' [/RETURNS_EXAMPLES]

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

     [DETAILED] This tool performs the same comprehensive polymorph retrieval as get_bulk_polymorphs_data but saves the results directly to a JSON file for persistent storage and later analysis.
    This tool retrieves comprehensive polymorph data from the Materials Project database for a specific chemical composition.
    Polymorphs are different crystal structures with the same chemical formula but different atomic arrangements, leading to distinct physical and chemical properties.
    This tool could be relevant for retrieving structures of the same compoisition.
    Apart from structure for each polymorph Materials Project ID (MP ID), CIF structure, energy above hull, formation energy per atom, band gap, density, volume, number of sites, space group, and stability information is also retrieved.
    The result is then saved a JSON file at the specified save_path.
    The file-based approach allows for efficient handling of large datasets and facilitates saving context of llm. [/DETAILED]

     [PROCEDURAL] When to use this tool:
     - Use when you need to explore all known structural variants of a single material composition.
     - Suitable for identifying thermodynamically stable and metastable phases and other properties like band gap, density, volume, number of sites, space group of the structure.
     - Use when you need to store polymorph data for later analysis or sharing or if you want to save context of llm.
     - Best suited for building persistent datasets and material databases
     - Highly recommended if the number of polymorphs for a compoisition could be very big
     - Avoid when you only need temporary data access
     - Avoid when you only need a single structure
     - Avoid when you need data for multiple compositions
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
         composition: [ARGS_BRIEF] Chemical composition formula. [/ARGS_BRIEF]
                     [ARGS_DETAILED] Chemical formula specifying the composition for which polymorphs should be retrieved and saved.
                     Should follow standard chemical notation with element symbols and subscripts.
                     The tool will find all known crystal structures with this exact composition in the Materials Project database. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] "Standard chemical formula (e.g., TiO2, Al2O3, CaTiO3)" [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] "TiO2" (titanium dioxide), "SiO2" (silicon dioxide), "Fe2O3" (iron oxide) [/ARGS_EXAMPLES]
         save_path: [ARGS_BRIEF] File path where JSON data will be saved. [/ARGS_BRIEF]
                   [ARGS_DETAILED] Complete file path including filename and extension where the polymorph data will be saved.
                   The path should be writable and the directory will be created if it doesn't exist.
                   Using .json extension is recommended for clarity.
                   If None, the tool will raise an error as the file path is required. [/ARGS_DETAILED]
                   [ARGS_SYNTACTICAL] "Valid file path with .json extension" [/ARGS_SYNTACTICAL]
                   [ARGS_EXAMPLES] "data/tio2_polymorphs.json", "save_path/tio2_polymorphs.json", "results/Cu2O_polymorphsides.json" [/ARGS_EXAMPLES]

     Returns:
         str: [RETURNS_BRIEF] File path where the polymorph data was saved. [/RETURNS_BRIEF]
              [RETURNS_DETAILED] Returns the exact file path where the JSON data was successfully written.
              This path can be used by subsequent tools for data loading and processing.
              The file contains comprehensive polymorph data in JSON format, sorted by thermodynamic stability. [/RETURNS_DETAILED]
              [RETURNS_EXAMPLES] "data/tio2_polymorphs.json" [/RETURNS_EXAMPLES]

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
def sort_and_get_first_from_json(json_data: str, sort_key: str, return_key: str) -> str:
    """[BRIEF] Sort JSON data by specified key and return the first element's specified value. [/BRIEF]

        [DETAILED] This utility tool provides flexible sorting and extraction capabilities for JSON data, particularly useful for materials data analysis where you need to identify optimal structures based on specific criteria.
        It enables quick identification of the best material according to any numerical property, such as finding the most stable phase, highest band gap material, or densest structure. [/DETAILED]

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
            json_data: [ARGS_BRIEF] JSON string containing the data to be sorted. [/ARGS_BRIEF]
                                [ARGS_DETAILED] A JSON-formatted string containing a list of dictionaries, each representing a material or structure with various properties.
                                The data should be structured consistently with numerical values for the sorting key.
                                This is typically output from polymorph retrieval tools. [/ARGS_DETAILED]
                                [ARGS_SYNTACTICAL] "Valid JSON string containing list of dictionaries" [/ARGS_SYNTACTICAL]
                                [ARGS_EXAMPLES] [
        {
            "material_id": "mp-390",
            "energy_above_hull": 0.0,
            "formation_energy_per_atom": -3.5080554519444442,
            "band_gap": 2.0586,
        },
        {
            "material_id": "mp-1840",
            "energy_above_hull": 0.0030413275,
            "formation_energy_per_atom": -3.505014124444444,
            "band_gap": 2.2862,
        },
    ] [/ARGS_EXAMPLES]
            sort_key: [ARGS_BRIEF] Property name to sort the data by. [/ARGS_BRIEF]
                     [ARGS_DETAILED] The dictionary key name that will be used for sorting the data.
                     This should correspond to a numerical property in the JSON data.
                     The sorting is performed in ascending order, so the first element will have the smallest value for this property.
                     Common keys include energy_above_hull, band_gap, density, formation_energy_per_atom. [/ARGS_DETAILED]
                     [ARGS_SYNTACTICAL] "String matching a key in the JSON data dictionaries" [/ARGS_SYNTACTICAL]
                     [ARGS_EXAMPLES] "energy_above_hull", "band_gap", "density"[/ARGS_EXAMPLES]
            return_key: [ARGS_BRIEF] Property name to return from the first element after sorting. [/ARGS_BRIEF]
                       [ARGS_DETAILED] The dictionary key name for the value that should be returned from the first (optimal) element after sorting.
                       This allows extraction of any property from the optimal structure, such as material_id for identification, cif for structure, or any other calculated property. [/ARGS_DETAILED]
                       [ARGS_SYNTACTICAL] "String matching a key in the JSON data dictionaries" [/ARGS_SYNTACTICAL]
                       [ARGS_EXAMPLES] "material_id", "cif", "formation_energy_per_atom"[/ARGS_EXAMPLES]

        Returns:
            str: [RETURNS_BRIEF] Value of the specified return_key from the first element after sorting. [/RETURNS_BRIEF]
                 [RETURNS_DETAILED] The value corresponding to the return_key from the material that has the smallest value for the sort_key.
                 This could be a string (like material_id or CIF), a number (like energy or band gap), or any other data type stored in the JSON.
                 The returned value represents the optimal material according to the specified sorting criterion. [/RETURNS_DETAILED]
                 [RETURNS_EXAMPLES] "mp-2657" (material ID), "1.23" (energy value), CIF structure string [/RETURNS_EXAMPLES]

        [RAISES] Exceptions:
            JSONDecodeError: [ERROR_WHEN] When the json_data string is not valid JSON [/ERROR_WHEN]
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
    data = json.loads(json_data)

    # Sort the data based on the given key
    sorted_data = sorted(data, key=lambda x: x[sort_key])

    # Return the value of the specified key from the first element
    return sorted_data[0][return_key]


@tool
def get_mp_thermo_data(material_id: str) -> str:
    """[BRIEF] Retrieve comprehensive thermodynamic data for materials from Materials Project database. [/BRIEF]

    [DETAILED] This tool accesses detailed thermodynamic information from the Materials Project database, providing essential data for understanding material stability, phase relationships, and thermodynamic properties.
    It retrieves formation energies, energy above hull, decomposition pathways, and stability information crucial for materials design and selection.
    This thermodynamic data enables informed decisions about material synthesis feasibility and provides benchmarks for computational studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need comprehensive thermodynamic data for specific materials
    - Recommended if you need to retrieve more thermodynamic information of a structure
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
    1. [PREREQUISITE] Ensure material MPID is valid [/PREREQUISITE]
    2. [CURRENT] Retrieve comprehensive thermodynamic data for target material [/CURRENT]
    3. [FOLLOW_UP] Use thermodynamic data for stability analysis or phase diagram studies [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `get_mp_thermo_data("mp-149")   # Silicon thermodynamic data`,
    `get_mp_thermo_data("mp-2657")  # TiO2 thermodynamic properties`,
    `get_mp_thermo_data("mp-1143")  # Al2O3 stability information`,
    [/SYNTACTICAL]

    Args:
        material_id: [ARGS_BRIEF] Materials Project ID for the target material. [/ARGS_BRIEF]
                    [ARGS_DETAILED] The unique Materials Project identifier for the material of interest.
                    Should be in the format "mp-XXXXX" where XXXXX is the numerical ID.
                    The material must exist in the Materials Project database and have thermodynamic calculations available. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] "mp-" followed by digits (e.g., "mp-149", "mp-2657") [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "mp-149" (Silicon), "mp-2657" (TiO2), "mp-1143" (Al2O3) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string containing comprehensive thermodynamic data and stability information. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing thermodynamic properties including material ID, thermodynamic functional used, formation energy per atom, energy above hull, decomposition products, stability status, energy type, and uncorrected energies.
             Returns error information if thermodynamic data is not available. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] '[{"material_id": "mp-149", "formation_energy_per_atom": -4.2, "energy_above_hull": 0.0, "is_stable": true, ...}]' [/RETURNS_EXAMPLES]

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
    """[BRIEF] Generate all unique surface slabs for bulk structure up to specified Miller indices systematically. [/BRIEF]

    [DETAILED] This tool provides comprehensive surface generation by systematically creating all unique surface slabs for a given bulk structure across all Miller indices up to a specified maximum. It implements advanced slab generation algorithms that explore different surface orientations and terminations, essential for systematic surface studies, catalysis research, and comprehensive materials characterization. This approach ensures no important surface orientations are missed in analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need comprehensive exploration of all possible surface orientations
    - Best suited for systematic surface studies and complete materials characterization
    - Essential for identifying optimal surface orientations for catalysis or adsorption
    - Recommended for research requiring exhaustive surface analysis
    - Avoid when you only need specific known surface orientations
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Systematically generates slabs for all Miller indices up to the specified maximum
    - Uses advanced algorithms to identify unique surface terminations and orientations
    - Applies consistent slab thickness and vacuum parameters across all surfaces
    - Calculates surface properties including area, thickness, and atom count
    - Returns comprehensive dataset with detailed metadata for each surface
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] First obtain bulk structure using get_structure_from_mp_text or select from polymorphs [/PREREQUISITE]
    2. [CURRENT] Generate comprehensive collection of all possible surface slabs [/CURRENT]
    3. [FOLLOW_UP] Use select_slabs_with_strategy to filter results or save_structures_to_db for storage [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `find_all_unique_slabs_upto_millerindex(bulk_cif, False, 2, 10, 15, True, 10)`,
    `find_all_unique_slabs_upto_millerindex("bulk_structure.cif", True, 1, 8, 12, True, 5)`,
    `find_all_unique_slabs_upto_millerindex(structure_string, False, 3, 12, 20, False, 15)`,
    [/SYNTACTICAL]

    Args:
        bulk_structure_path_or_string: [ARGS_BRIEF] Path to CIF file or CIF string of bulk structure. [/ARGS_BRIEF]
                                      [ARGS_DETAILED] Either a complete file path to a CIF file containing the bulk crystal structure, or a CIF-formatted string containing the structure data, depending on the from_path parameter. This structure serves as the basis for all surface generation and should be a well-defined three-dimensional crystal. [/ARGS_DETAILED]
                                      [ARGS_SYNTACTICAL] "Valid CIF file path or string in valid CIF syntax" [/ARGS_SYNTACTICAL]
                                      [ARGS_EXAMPLES] "structures/bulk_si.cif", CIF string from Materials Project [/ARGS_EXAMPLES]

        from_path: [ARGS_BRIEF] Boolean indicating if input is a file path. Defaults to False. [/ARGS_BRIEF]
                  [ARGS_DETAILED] Boolean flag that determines how to interpret the bulk_structure_path_or_string parameter. When True, treats the input as a file path to read. When False, treats it as a CIF string to parse directly. This provides flexibility for different data input patterns in workflows. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] boolean value (True/False) [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] True (file input), False (string input) [/ARGS_EXAMPLES]

        max_index: [ARGS_BRIEF] Maximum Miller index to consider. Defaults to 2. [/ARGS_BRIEF]
                  [ARGS_DETAILED] The maximum value for Miller indices (h, k, l) to include in surface generation. Higher values explore more surface orientations but increase computational cost exponentially. Common choices are 1, 2, or 3 depending on the comprehensiveness required and computational resources available. [/ARGS_DETAILED]
                  [ARGS_SYNTACTICAL] positive integer (1, 2, or 3) [/ARGS_SYNTACTICAL]
                  [ARGS_EXAMPLES] 1 (basic orientations), 2 (standard), 3 (comprehensive but expensive) [/ARGS_EXAMPLES]

        min_slab_size: [ARGS_BRIEF] Minimum slab thickness in Angstroms. Defaults to 8. [/ARGS_BRIEF]
                      [ARGS_DETAILED] The minimum thickness of generated slabs in the direction perpendicular to the surface plane. This ensures adequate bulk-like behavior in the slab center while exposing the desired surface. Larger values provide more accurate surface representation but increase computational cost. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] positive float representing thickness in Angstroms [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] 8.0 (minimal), 12.0 (standard), 15.0 (thick) [/ARGS_EXAMPLES]

        min_vacuum_size: [ARGS_BRIEF] Minimum vacuum layer thickness in Angstroms. Defaults to 15. [/ARGS_BRIEF]
                        [ARGS_DETAILED] The minimum vacuum space above each surface to prevent interactions between periodic images in surface calculations. Larger vacuum regions are essential for accurate surface energy calculations and prevent spurious interactions between surface images. [/ARGS_DETAILED]
                        [ARGS_SYNTACTICAL] positive float representing vacuum thickness in Angstroms [/ARGS_SYNTACTICAL]
                        [ARGS_EXAMPLES] 10.0 (minimal), 15.0 (standard), 20.0 (large) [/ARGS_EXAMPLES]

        center_slab: [ARGS_BRIEF] Whether to center slab in vacuum region. Defaults to True. [/ARGS_BRIEF]
                    [ARGS_DETAILED] Boolean flag controlling whether the slab should be positioned in the center of the vacuum region. Centering is generally recommended for symmetric boundary conditions and consistent surface calculations. Setting to False may be useful for specific calculation requirements or interfacial studies. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] boolean value (True/False) [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] True (centered, recommended), False (offset positioning) [/ARGS_EXAMPLES]

        max_normal_search: [ARGS_BRIEF] Maximum number of surface normals to search. Defaults to 10. [/ARGS_BRIEF]
                          [ARGS_DETAILED] The maximum number of surface normal directions to explore for each Miller index. Higher values may find more unique terminations but increase computational cost. This parameter controls the thoroughness of surface termination exploration for complex structures. [/ARGS_DETAILED]
                          [ARGS_SYNTACTICAL] positive integer [/ARGS_SYNTACTICAL]
                          [ARGS_EXAMPLES] 5 (quick), 10 (standard), 20 (thorough) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON dictionary with comprehensive slab data including properties and metadata. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing a dictionary where keys are slab identifiers and values contain comprehensive slab information including Miller indices, termination numbers, CIF structures, surface areas, atom counts, and slab thicknesses. This provides complete characterization of all generated surfaces for analysis and selection. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] "{"111_0": {"miller_index": [1,1,1], "cif": "...", "area": 45.2, "num_sites": 24, ...}}" [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When bulk structure is invalid or parameters are incompatible [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format, negative size parameters, or structure incompatible with Miller indices [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify structure format and ensure all parameters are positive [/ERROR_RECOVERY]
        FileNotFoundError: [ERROR_WHEN] When from_path=True but file doesn't exist [/ERROR_WHEN]
                          [ERROR_DETAILS] Specified file path cannot be found or accessed [/ERROR_DETAILS]
                          [ERROR_RECOVERY] Check file path exists and is readable [/ERROR_RECOVERY]
        MemoryError: [ERROR_WHEN] When max_index is too large for available memory [/ERROR_WHEN]
                    [ERROR_DETAILS] Exponential growth in surface combinations exceeds memory limits [/ERROR_DETAILS]
                    [ERROR_RECOVERY] Reduce max_index or increase available memory [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Computational cost grows exponentially with max_index
    - May generate many similar surfaces for high-symmetry structures
    - Does not perform surface relaxation or energy calculations
    - Cannot predict relative stability or importance of different surfaces
    [/LIMITATIONS]
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
def get_mp_surface_properties(material_id: str) -> str:
    """[BRIEF] Retrieve comprehensive surface properties for materials from Materials Project database. [/BRIEF]

    [DETAILED] This tool retrieves detailed surface properties and energetics for specific materials from the Materials Project database, providing essential information for surface chemistry and catalysis studies.
    It accesses calculated surface energies, anisotropy factors, shape factors, and reconstruction information that are crucial for understanding surface stability and reactivity.
    This data enables informed selection of materials for surface applications and provides theoretical benchmarks for computational studies. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use when you need comprehensive surface property data for specific materials
    - Best suited for surface stability analysis and catalysis material selection
    - Essential for benchmarking computational surface calculations
    - Recommended for systematic surface property studies across material classes
    - Avoid when you only need basic structural information
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Connects to Materials Project API to access surface property database
    - Retrieves calculated surface energies and related thermodynamic properties
    - Provides surface anisotropy and shape factor information for crystal habit prediction
    - Reports reconstruction information for complex surface behavior
    - Returns comprehensive JSON with all available surface properties
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure MPID is valid [/PREREQUISITE]
    2. [CURRENT] Retrieve comprehensive surface properties for target material [/CURRENT]
    3. [FOLLOW_UP] Use surface property data for material selection or computational benchmarking [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    - get_mp_surface_properties("mp-149")  # Silicon surface properties
    - get_mp_surface_properties("mp-2657")  # TiO2 surface properties
    - get_mp_surface_properties("mp-1143")  # Other material surface data
    [/SYNTACTICAL]

    Args:
        material_id: [ARGS_BRIEF] Materials Project ID for the target material. [/ARGS_BRIEF]
                    [ARGS_DETAILED] The unique Materials Project identifier for the material of interest.
                    Should be in the format "mp-XXXXX" where XXXXX is the numerical ID.
                    The material must exist in the Materials Project database and have calculated surface properties available. [/ARGS_DETAILED]
                    [ARGS_SYNTACTICAL] "mp-" followed by digits (e.g., "mp-149", "mp-2657") [/ARGS_SYNTACTICAL]
                    [ARGS_EXAMPLES] "mp-149" (Silicon), "mp-2657" (TiO2 anatase), "mp-1143" (Al2O3) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string containing comprehensive surface properties and energetics data. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing surface properties including material ID, formula, weighted surface energy, surface energy in eV/Å², surface anisotropy, shape factor, and reconstruction information.
             Returns error information if surface properties are not available for the specified material. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] '[{"material_id": "mp-149", "weighted_surface_energy": 1.23, "surface_anisotropy": 0.15, ...}]' [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When Materials Project API key is not available [/ERROR_WHEN]
                   [ERROR_DETAILS] MP_API_KEY environment variable not set or invalid [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Obtain valid API key from Materials Project and set environment variable [/ERROR_RECOVERY]
        ConnectionError: [ERROR_WHEN] When unable to connect to Materials Project API [/ERROR_WHEN]
                        [ERROR_DETAILS] Network connectivity issues or API server problems [/ERROR_DETAILS]
                        [ERROR_RECOVERY] Check internet connection and try again later [/ERROR_RECOVERY]
        KeyError: [ERROR_WHEN] When material ID is not found or has no surface data [/ERROR_WHEN]
                 [ERROR_DETAILS] Invalid material ID or surface properties not calculated [/ERROR_DETAILS]
                 [ERROR_RECOVERY] Verify material ID exists and has surface property calculations [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to materials with calculated surface properties in Materials Project
    - Surface property accuracy depends on computational methodology used
    - May not include very recent calculations or experimental data
    - Cannot provide surface properties for custom or modified structures
    [/LIMITATIONS]
    """

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


@tool
def generate_adsorbate_slab_configs(
    slab_cif: str, adsorbate_cif: str, adsorption_sites_json: str, height: float = 1.8
) -> str:
    """[BRIEF] Generate adsorbate-slab configurations by placing adsorbate molecules at different surface sites. [/BRIEF]

    [DETAILED] This tool creates multiple adsorbate-slab configurations by systematically placing adsorbate molecules at different adsorption sites on a slab surface.
    It uses pymatgen's AdsorbateSiteFinder to handle the geometric placement of adsorbates at specified surface sites, including top, bridge, and hollow sites.
    This is essential for adsorption energy calculations, catalysis studies, and surface reactivity analysis. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - Use after identifying adsorption sites on a slab surface
    - Essential for creating input structures for adsorption energy calculations
    - Use when studying catalytic reactions or surface interactions
    - Recommended for systematic screening of adsorption configurations
    - Avoid when you only need the clean slab surface without adsorbates
    [/PROCEDURAL]

    [CONTEXTUAL] How this tool works:
    - Loads the slab structure from CIF string
    - Converts adsorbate CIF to a molecular structure
    - Parses the adsorption sites information from JSON
    - Uses AdsorbateSiteFinder from pymatgen to place adsorbates at each site
    - Generates multiple configurations with different site types and positions
    - Returns a JSON containing all successfully generated adsorbate-slab configurations
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration example:
    1. [PREREQUISITE] Generate slab structure using slab creation tools [/PREREQUISITE]
    2. [PREREQUISITE] Identify adsorption sites using get_adsorption_sites_text [/PREREQUISITE]
    3. [PREREQUISITE] Have adsorbate molecule structure available [/PREREQUISITE]
    4. [CURRENT] Apply this tool to generate adsorbate-slab configurations [/CURRENT]
    5. [FOLLOW_UP] Use configurations for DFT calculations or energy analysis [/FOLLOW_UP]
    [/WORKFLOW_INTEGRATION]

    [SYNTACTICAL] Usage examples:
    `generate_adsorbate_slab_configs(slab_cif, adsorbate_cif, sites_json, 1.8)`,
    `generate_adsorbate_slab_configs(slab_cif, adsorbate_cif, sites_json, height=2.0)`,
    `generate_adsorbate_slab_configs(slab_cif, adsorbate_cif, sites_json)`,
    [/SYNTACTICAL]

    Args:
        slab_cif: [ARGS_BRIEF] CIF string of the slab structure. [/ARGS_BRIEF]
                 [ARGS_DETAILED] A properly formatted CIF string containing the slab structure on which adsorbates will be placed.
                 The slab should be oriented with the surface normal along the c-axis for proper adsorbate placement. [/ARGS_DETAILED]
                 [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                 [ARGS_EXAMPLES] Output from slab generation tools [/ARGS_EXAMPLES]
        adsorbate_cif: [ARGS_BRIEF] CIF string of the adsorbate molecule. [/ARGS_BRIEF]
                      [ARGS_DETAILED] A CIF-formatted string containing the adsorbate molecule structure that will be placed on the slab surface.
                      The molecule should be properly oriented and have reasonable geometry for surface adsorption. [/ARGS_DETAILED]
                      [ARGS_SYNTACTICAL] string in valid CIF syntax [/ARGS_SYNTACTICAL]
                      [ARGS_EXAMPLES] "# generated using pymatgen\ndata_Si\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.83996459\n_cell_length_b   3.83996459\n_cell_length_c   18.81190774\n_cell_angle_alpha   90.00000000\n_cell_angle_beta   90.00000000\n_cell_angle_gamma   120.00000000\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   Si\n_chemical_formula_sum   Si8\n_cell_volume   240.22483885\n_cell_formula_units_Z   8\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Si  Si0  1  0.83333333  0.41666667  0.10416667  1.0\n  Si  Si1  1  0.50000000  0.75000000  0.06250000  1.0\n  Si  Si2  1  0.16666667  0.08333333  0.27083333  1.0\n  Si  Si3  1  0.83333333  0.41666667  0.22916667  1.0\n  Si  Si4  1  0.50000000  0.75000000  0.43750000  1.0\n  Si  Si5  1  0.16666667  0.08333333  0.39583333  1.0\n  Si  Si6  1  0.83333333  0.41666667  0.60416667  1.0\n  Si  Si7  1  0.50000000  0.75000000  0.56250000  1.0\n" [/ARGS_EXAMPLES]
        adsorption_sites_json: [ARGS_BRIEF] JSON string containing adsorption sites information. [/ARGS_BRIEF]
                              [ARGS_DETAILED] A JSON-formatted string containing information about potential adsorption sites on the slab surface, typically organized by site type (top, bridge, hollow) with coordinates for each site. [/ARGS_DETAILED]
                              [ARGS_SYNTACTICAL] "JSON string with site types and coordinates" [/ARGS_SYNTACTICAL]
                              [ARGS_EXAMPLES] {"top": [[x1,y1,z1], [x2,y2,z2]], "bridge": [...]} [/ARGS_EXAMPLES]
        height: [ARGS_BRIEF] Height in Angstroms for initial adsorbate placement. Defaults to 1.8. [/ARGS_BRIEF]
               [ARGS_DETAILED] The initial height above the surface at which the adsorbate will be placed.
               This is the starting geometry for optimization and should be reasonable for the specific adsorbate-surface system.
               Typical values are 1.5-2.5 Å. [/ARGS_DETAILED]
               [ARGS_SYNTACTICAL] positive float representing height in Angstroms [/ARGS_SYNTACTICAL]
               [ARGS_EXAMPLES] 1.5 (close), 1.8 (standard), 2.2 (distant) [/ARGS_EXAMPLES]

    Returns:
        str: [RETURNS_BRIEF] JSON string containing all generated adsorbate-slab configurations. [/RETURNS_BRIEF]
             [RETURNS_DETAILED] A JSON-formatted string containing all successfully generated adsorbate-slab configurations, with each configuration including site information, coordinates, and the complete CIF structure ready for calculations. [/RETURNS_DETAILED]
             [RETURNS_EXAMPLES] {"top_0": {"site_coords": [x,y,z], "cif": "..."}, ...} [/RETURNS_EXAMPLES]

    [RAISES] Exceptions:
        ValueError: [ERROR_WHEN] When CIF structures are malformed or incompatible [/ERROR_WHEN]
                   [ERROR_DETAILS] Invalid CIF format for slab or adsorbate, or parsing errors [/ERROR_DETAILS]
                   [ERROR_RECOVERY] Verify CIF formats and ensure structures are valid [/ERROR_RECOVERY]
        StructureError: [ERROR_WHEN] When adsorbate placement fails [/ERROR_WHEN]
                       [ERROR_DETAILS] Geometric conflicts or invalid adsorption sites [/ERROR_DETAILS]
                       [ERROR_RECOVERY] Check adsorption sites and adjust height parameter [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Limited to 3 sites per site type to manage computational cost
    - Does not optimize adsorbate geometry or consider surface relaxation
    - May skip sites that cause geometric conflicts
    - No validation of chemical reasonableness of adsorption sites
    [/LIMITATIONS]
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
        # OCP hard -  Distractors
        "get_bulk_polymorphs_data": get_bulk_polymorphs_data,
        "sort_and_get_first_from_json": sort_and_get_first_from_json,
        "find_all_unique_slabs_upto_millerindex": find_all_unique_slabs_upto_millerindex,
        "generate_adsorbate_slab_configs": generate_adsorbate_slab_configs,
        "get_mp_surface_properties": get_mp_surface_properties,
        "get_mp_thermo_data": get_mp_thermo_data,
    }
