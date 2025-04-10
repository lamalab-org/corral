import json
import os
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pymatgen.core import Structure

if "CORRAL_WORK_DIR" not in os.environ:
    raise OSError("Environment variable 'CORRAL_WORK_DIR' is not set.")
BASE_WORK_DIR = os.environ["CORRAL_WORK_DIR"]


def resolve_path(path_or_str: str) -> str:
    """
    Resolves a path that might be relative to the base work directory.
    Also cleans up common input format issues.
    """
    # Handle various input issues
    if isinstance(path_or_str, str):
        # Remove "answer:" prefix if present
        if path_or_str.startswith("answer:"):
            path_or_str = path_or_str.replace("answer:", "", 1).strip()

        # Replace escaped quotes that might come from JSON strings
        path_or_str = path_or_str.replace('\\"', '"').replace("\\'", "'")

    try:
        # If it's an absolute path or already exists, return as is
        if Path(path_or_str).is_absolute() or Path(path_or_str).exists():
            return path_or_str

        # Try to resolve against base directory
        full_path = Path(BASE_WORK_DIR) / path_or_str
        if full_path.exists():
            return str(full_path)

        # If we can't resolve it, return the original
        return path_or_str
    except Exception:
        # If there's any error treating it as a path, return the original
        return path_or_str


def check_slabs_json(slabs_json: str) -> float:
    """
    Check that the slabs JSON contains at least one valid slab by trying to parse
    the CIF string for one of the slabs. Accepts either a path to a JSON file or a raw JSON string.
    """
    import json
    from pathlib import Path

    from pymatgen.core import Structure

    try:
        # Try loading from file if it's a valid path
        json_data = None
        if Path(slabs_json).exists():
            with Path(slabs_json).open() as f:
                json_data = json.load(f)
        else:
            json_data = json.loads(slabs_json)

        if not json_data or not isinstance(json_data, dict):
            return 0.0

        # Choose one slab and validate
        for cif in json_data.values():
            try:
                struct = Structure.from_str(cif, fmt="cif")
                if struct and len(struct) > 0:
                    return 1.0
            except Exception:
                continue
        return 0.0

    except Exception:
        return 0.0


def check_mp_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a structure from Materials Project.
    """
    logger.info("check_mp_structure")
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
        path_or_cif = resolve_path(path_or_cif)

        # Then continue with the existing logic
        if Path(path_or_cif).exists():
            structure = Structure.from_file(path_or_cif)
        else:
            structure = Structure.from_str(path_or_cif, fmt="cif")

        return 1.0 if structure and len(structure) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error validating structure: {e}")
        return 0.0


def check_slab_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a slab structure.

    Args:
        path_or_cif: Either a path to a CIF file or a CIF string

    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info("check_slab_structure")
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
        path_or_cif = resolve_path(path_or_cif)
        # Determine if the input is a path or a CIF string
        if Path(path_or_cif).exists():
            structure = Structure.from_file(path_or_cif)
        else:
            structure = Structure.from_str(path_or_cif, fmt="cif")

        # Check if the structure is valid
        return 1.0 if structure and len(structure) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error validating slab structure: {e}")
        return 0.0


def check_co2_molecule_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a molecule structure (e.g., CO2).

    Args:
        path_or_cif: Either a path to a CIF file or a CIF string

    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info("check_molecule_structure")
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
        path_or_cif = resolve_path(path_or_cif)
        # Determine if the input is a path or a CIF string
        if Path(path_or_cif).exists():
            structure = Structure.from_file(path_or_cif)
        else:
            structure = Structure.from_str(path_or_cif, fmt="cif")

        # Check if the structure is valid
        if structure and len(structure) > 0:
            # Check for CO2 molecule (simple check for C and O atoms)
            has_carbon = any(site.species_string == "C" for site in structure)
            has_oxygen = any(site.species_string == "O" for site in structure)

            if has_carbon and has_oxygen:
                # Look for correct stoichiometry (1 C, 2 O)
                c_count = sum(1 for site in structure if site.species_string == "C")
                o_count = sum(1 for site in structure if site.species_string == "O")

                return 1.0 if c_count == 1 and o_count == 2 else 0.75
            return 0.5  # Valid structure but missing C or O
        return 0.25  # Empty but valid structure
    except Exception as e:
        logger.error(f"Error validating molecule structure: {e}")
        return 0.0


def check_adsorption_structure(
    slab_elements: list[str], adsorbate_elements: list[str]
) -> Callable[[str], float]:
    """Returns a scoring function customized to given slab and adsorbate elements"""

    def score_fn(path_or_cif: str) -> float:
        try:
            from pathlib import Path

            from pymatgen.core import Structure

            if Path(path_or_cif).exists():
                structure = Structure.from_file(path_or_cif)
            else:
                structure = Structure.from_str(path_or_cif, fmt="cif")

            if not structure or len(structure) == 0:
                return 0.25

            atoms = {str(site.specie) for site in structure}
            has_slab = all(e in atoms for e in slab_elements)
            has_adsorbate = all(e in atoms for e in adsorbate_elements)

            # abc = structure.lattice.abc # TODO: Check if slab-like
            # is_slab_like = abc[2] > 2 * max(abc[0], abc[1]) # need bot always in c direction

            if has_slab and has_adsorbate:
                return 1.0  # if is_slab_like else 0.75
            elif has_slab or has_adsorbate:
                return 0.5
            else:
                return 0.25
        except Exception:
            return 0.0

    return score_fn


def check_adsorption_sites(sites_json_or_path: str) -> float:
    """
    Check if the JSON string contains valid adsorption sites.

    Args:
        sites_json_or_path: JSON string containing adsorption sites or path to a JSON file

    Returns:
        float: Score between 0.0 and 1.0
    """

    try:
        if Path(sites_json_or_path).is_file():
            with Path(sites_json_or_path).open() as f:
                sites_json_or_path = f.read()

        sites = json.loads(sites_json_or_path)

        # Check if the structure contains expected site types
        type_aliases = {
            "top": "ontop",
            "ontop": "ontop",
            "bridge": "bridge",
            "hollow": "hollow",
        }
        found_types = [alias for alias, canon in type_aliases.items() if alias in sites]

        if not found_types:
            return 0.25  # No recognized site types

        # Check if sites have coordinates
        has_coords = any(
            isinstance(sites.get(site_type), list) and len(sites.get(site_type)) > 0
            for site_type in found_types
        )

        if not has_coords:
            return 0.5  # Has site types but all are empty

        # Check if at least one site type has valid coordinates
        has_valid_coords = any(
            all(
                isinstance(coord, list) and len(coord) == 3
                for coord in sites.get(site_type, [])
            )
            for site_type in found_types
            if sites.get(site_type)
        )

        if not has_valid_coords:
            return 0.75  # Has coordinates but they're malformed

        return 1.0  # At least one site type has valid coordinates

    except Exception as e:
        logger.error(f"Error validating adsorption sites: {e}")
        return 0.0
