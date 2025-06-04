import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tool_utils import (
    find_surface_atoms_with_voronoi,
    load_structure,
    set_fixed_atom_constraints,
    standardize_bulk,
    tile_atoms,
)

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


### Tools relevant for ocp - hard


@tool
def get_bulk_polymorphs_data(composition: str) -> str:
    """
    Query the Materials Project database to find polymorphs for a given composition.

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
def sort_and_get_first_from_json(polymorph_data_json: str) -> str:
    """
    From a JSON string, sort the data based on a given key and return the first element of the specified key.

    Args:
        json_data: JSON string containing the data to be sorted.
        sort_key: Key to sort the data by.
        return_key: Key of the first element to return after sorting.

    Returns:
        Value of the specified return_key from the first element after sorting.
    """

    def sort_and_get_first(json_data: str, sort_key: str, return_key: str) -> any:
        data = json.loads(json_data)

        # Sort the data based on the given key
        sorted_data = sorted(data, key=lambda x: x[sort_key])

        # Return the value of the specified key from the first element
        return sorted_data[0][return_key]

    # Example usage for polymorph data
    return sort_and_get_first(polymorph_data_json, "energy_above_hull", "cif")


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
