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
    """
    Retrieve a pymatgen structure from Materials Project using its API and return its
    CIF content as a text string.

    Args:
        mp_id: Materials Project id.

    Returns:
        CIF content string.
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
    """
    Create a slab from a structure given as CIF-formatted string. Returns slab as CIF string.

    Args:
        structure_cif: CIF content (structure) as text.
        miller_index: Miller index to cleave the slab.
        min_slab_size: Minimum slab thickness.
        min_vacuum_size: Vacuum distance needed.
        primitive: Whether to create a primitive cell slab.

    Returns:
        Slab CIF content as string.
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
        adsorbate_struct = Structure.from_str(adsorbate_cif, fmt="cif")
        adsorbate = Molecule(
            species=adsorbate_struct.species,
            coords=adsorbate_struct.cart_coords,
            charge=0,
        )
    except Exception:
        adsorbate = Molecule.from_str(adsorbate_cif, fmt="cif")

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
    Generate reconstructed slab(s) from a bulk structure with full parameter utilization.

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
        bulk_cif (str): CIF string of bulk structure
        miller_index (tuple[int, int, int]): Miller indices (h,k,l) for surface orientation
        min_slab_size (float): Minimum slab thickness (Å)
        min_vacuum_size (float): Minimum vacuum thickness (Å)
        reconstruction_instructions (str): JSON string with reconstruction parameters
        return_all_variants (bool): If True, returns all slab variants as JSON

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
