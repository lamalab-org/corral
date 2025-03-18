import json
import os
from pathlib import Path

from loguru import logger
from pymatgen.core import Structure

BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


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

    # Rest of your existing logic
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
    the CIF string for one of the slabs.
    """
    import json

    from pymatgen.core import Structure

    try:
        slabs_json = resolve_path(slabs_json)
        slabs = json.loads(slabs_json)
        if not slabs:
            return 0.0
        # Choose one slab and validate
        for cif in slabs.values():
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

        if structure and len(structure) > 0:
            return 1.0  # Valid structure
        return 0.0  # Invalid structure
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
        if structure and len(structure) > 0:
            return 1.0  # Valid structure
        return 0.0
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

                if c_count == 1 and o_count == 2:
                    return 1.0
                else:
                    return 0.75  # Has C and O but not correct stoichiometry
            return 0.5  # Valid structure but missing C or O
        return 0.25  # Empty but valid structure
    except Exception as e:
        logger.error(f"Error validating molecule structure: {e}")
        return 0.0


def check_adsorption_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a slab with an adsorbed molecule.

    Args:
        path_or_cif: Either a path to a CIF file or a CIF string

    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info("check_adsorption_structure")
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
            # Check for a slab with CO2 molecule
            has_silicon = any(site.species_string == "Si" for site in structure)
            has_carbon = any(site.species_string == "C" for site in structure)
            has_oxygen = any(site.species_string == "O" for site in structure)

            if has_silicon and has_carbon and has_oxygen:
                # Determine if the structure has slab-like characteristics
                lattice = structure.lattice
                abc = lattice.abc
                if abc[2] > 2 * max(
                    abc[0], abc[1]
                ):  # c significantly larger than a or b
                    return 1.0
                return 0.75  # Has all atoms but may not be in a slab configuration
            return 0.5  # Missing some atoms
        return 0.25  # Empty but valid structure
    except Exception as e:
        logger.error(f"Error validating adsorption structure: {e}")
        return 0.0


def check_adsorption_sites(sites_json: str) -> float:
    """
    Check if the JSON string contains valid adsorption sites.

    Args:
        sites_json: JSON string containing adsorption sites

    Returns:
        float: Score between 0.0 and 1.0
    """
    try:
        sites = json.loads(sites_json)

        # Check if the structure contains expected site types
        expected_types = ["top", "bridge", "hollow"]
        found_types = [site_type for site_type in expected_types if site_type in sites]

        if not found_types:
            return 0.25  # No recognized site types

        # Check if sites have coordinates
        has_coords = all(
            isinstance(sites.get(site_type), list) and len(sites.get(site_type)) > 0
            for site_type in found_types
        )

        if not has_coords:
            return 0.5  # Has site types but no coordinates

        # Check structure of coordinates
        valid_coords = all(
            all(
                isinstance(coord, list) and len(coord) == 3
                for coord in sites.get(site_type, [])
            )
            for site_type in found_types
        )

        if not valid_coords:
            return 0.75  # Has coordinates but they're not in the expected format

        return 1.0  # Valid sites with coordinates
    except Exception as e:
        logger.error(f"Error validating adsorption sites: {e}")
        return 0.0
