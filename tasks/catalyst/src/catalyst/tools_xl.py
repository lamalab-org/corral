import os

from dotenv import load_dotenv

from corral.base import Tool
from corral.utils import tool

load_dotenv("../.env")


####################
# Tools that will return text strings
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
    - Searches for the specified material ID in the database
    - Retrieves the pymatgen Structure object containing atomic positions and lattice parameters
    - Converts the structure to CIF format string for compatibility with other tools
    - Returns standardized crystallographic data suitable for further processing
    [/CONTEXTUAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure MP_API_KEY environment variable is set with valid Materials Project API key [/PREREQUISITE]
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
                 [ERROR_RECOVERY] Verify MP ID exists on Materials Project website or check MP ID suntax[/ERROR_RECOVERY]
        AuthenticationError: [ERROR_WHEN] When API key is invalid or missing [/ERROR_WHEN]
                             [ERROR_DETAILS] MP_API_KEY environment variable not set or expired [/ERROR_DETAILS]
                             [ERROR_RECOVERY] Obtain valid API key from Materials Project and set environment variable [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known limitations:
    - Requires valid Materials Project API key and internet connection
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

    [WORKFLOW_INTEGRATION] Typical workflow integration:
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
    """
    Enumerate possible slabs from a bulk structure (given as CIF text) using SlabGenerator.
    Returns a JSON string mapping slab indices to CIF strings.

    Args:
        bulk_cif: Bulk structure in CIF string format.
        miller_index: Miller index (e.g. (1,1,1)).
        min_slab_size: Minimum slab thickness (Å).
        min_vacuum_size: Minimum vacuum layer (Å).

    Returns:
        str: JSON dictionary: {"slab_0": "<cif_string>", "slab_1": "<cif_string>", ...}
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
    """
    Selects one slab from the JSON dictionary of slabs (by its index) and returns its CIF string.

    Args:
        slabs_json: JSON string mapping slab keys to CIF strings.
        index: Index of the slab to select (default 0).

    Returns:
        str: CIF string for the selected slab.
    """
    import json

    slabs = json.loads(slabs_json)
    key = f"slab_{index}"
    if key not in slabs:
        raise ValueError(f"Slab index {index} not found.")
    return slabs[key]


@tool
def get_adsorption_sites_text(slab_cif: str) -> str:
    """
    Determine possible adsorption sites on a slab.
    Returns a JSON string that contains lists of binding sites (e.g. top, bridge, hollow).

    Args:
        slab_cif: CIF string of the slab.

    Returns:
        str: JSON dictionary of adsorption sites. (list of fractional coordinates)
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
    """
    Selects one adsorption site from the JSON dictionary of sites (by its type and index) and returns its fractional coordinates.

    Args:
        adsorption_sites_json: JSON string mapping site types to lists of fractional coordinates.
        site_type: Type of the site (e.g. "top", "bridge", "hollow").
        index: Index of the site to select (default 0).

    Returns:
        list: Fractional coordinates of the selected site.
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
    """
    Place an adsorbate (given as a CIF string) on a slab at a specified adsorption site.
    If no site is specified, choose one from the top sites automatically.

    Args:
        slab_cif: CIF string of the slab.
        adsorbate_cif: CIF string of the adsorbate.
        height: Height (Å) above the slab surface where the adsorbate should be placed.
        site: Optional fractional coordinate [x, y, z] for placement.
            If None, the first top site will be used.

    Returns:
        str: CIF string of the combined (adsorbate+slab) structure.
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
    """
    Generate reconstructed slab(s) from a bulk structure.

    Example reconstruction_instructions JSON: # https://pymatgen.org/pymatgen.core.html#module-pymatgen.core.surface
    {
        "name": "fcc_111_2x2_octopolar",
        "description": "Octopolar reconstruction of FCC (111) surface",
        "miller_index": [1, 1, 1],
        "Woods_notation": "p(2x2)",
        "reference": "Optional reference to publication or source",
        "spacegroup": {"symbol": "Fm-3m", "number": 225},
        "transformation_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        "SlabGenerator_parameters": {
            "center_slab": true,
            "in_unit_planes": false,
            "primitive": false,
            "lll_reduce": true
        },
        "points_to_remove": [
            [0.25, 0.25, 0],
            [0.75, 0.75, 0]
        ],
        "points_to_add": [
            [0.5, 0.5, 0.2, "Cu", {"charge": 1}]
        ],
        "variant_info": [
            {
                "index": 0,
                "description": "Standard octopolar reconstruction",
                "characteristics": ["most stable", "C3v symmetric"]
            }
        ],
        "base_reconstruction": null  # Optional reference to another reconstruction
    }

    Args:
        bulk_cif: CIF string of bulk structure
        miller_index : Miller indices (h,k,l) for surface orientation
        min_slab_size : Minimum slab thickness (Å)
        min_vacuum_size : Minimum vacuum thickness (Å)
        reconstruction_instructions : JSON string with reconstruction parameters
        return_all_variants : If True, returns all slab variants as JSON

    Returns:
        str: Either a single CIF string (if return_all_variants=False) or a
             JSON string with all variants and metadata (if return_all_variants=True)
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
