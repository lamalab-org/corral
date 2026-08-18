import json
import os
import secrets
import string
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from pymatgen.core import Structure

from corral.utils.tool_helpers import smart_resolve_path

# Generate a random 4-letter unique identifier
uid = "".join(secrets.choice(string.ascii_lowercase) for _ in range(6))

# Default base work dir when CORRAL_WORK_DIR is unset (relative path + UID).
# NOTE: intentionally do NOT write CORRAL_WORK_DIR back into the process
# environment here. That pins every concurrent task execution to one shared
# directory and breaks execution workspace isolation. Scoring resolves the
# submitted answer against the task execution's workspace via
# Environment._resolve_answer.
#
# Resolve to an ABSOLUTE path at import time for server-side materialization and
# evaluation. Native agent harnesses are deliberately not given this path;
# their task file access goes through the execution-scoped MCP tools instead.
# `resolve()` is evaluated against the server's cwd, which is exactly where the
# workspace directories are created.
BASE_WORK_DIR = str(
    Path(
        os.environ.get("CORRAL_WORK_DIR", f"../CORRAL_WORK_DIR/catalyst_{uid}")
    ).resolve()
)


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


def check_valid_json_file(json_path: str) -> float:
    """
    Check if a valid JSON file exists at the given path.

    Args:
        json_path: Path to the JSON file to validate

    Returns:
        float: 1.0 if valid JSON file exists, 0.0 otherwise
    """
    try:
        json_path = json_path.strip()
        if not json_path or not json_path.strip():
            logger.warning("Empty path provided to check_valid_json_file")
            return 0.0

        # Check if file exists
        if not Path(json_path).exists():
            logger.info(f"JSON file not found at: {json_path}")
            return 0.0

        # Check if it's a file (not a directory)
        if not Path(json_path).is_file():
            logger.info(f"Path exists but is not a file: {json_path}")
            return 0.0

        # Try to load and parse the JSON
        with Path(json_path).open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Additional validation - check if it's not empty
        if json_data is None:
            logger.info("JSON file contains null")
            return 0.0  # Valid JSON but null content

        logger.info(f"Valid JSON file found with {type(json_data).__name__} content")
        return 1.0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in file {json_path}: {e}")
        return 0.0
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading file {json_path}: {e}")
        return 0.0
    except PermissionError as e:
        logger.error(f"Permission denied reading file {json_path}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error validating JSON file {json_path}: {e}", exc_info=True)
        return 0.0


def check_slabs_json(slabs_json: str) -> float:
    """
    Check that the slabs JSON contains at least one valid slab by trying to parse
    the CIF string for one of the slabs. Accepts either a path to a JSON file or a raw JSON string.
    """
    try:
        logger.info(f"check_slabs_json: input={slabs_json!r}")

        # Try to resolve as path
        resolved_input = smart_resolve_path(slabs_json)
        logger.info(f"check_slabs_json: resolved={resolved_input!r}")

        # Try loading from file if it's a valid path
        json_data = None
        if Path(resolved_input).exists():
            with Path(resolved_input).open() as f:
                json_data = json.load(f)
        else:
            # Try parsing as raw JSON string (try original first, then resolved)
            try:
                json_data = json.loads(slabs_json)
            except json.JSONDecodeError:
                json_data = json.loads(resolved_input)

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

    except Exception as e:
        logger.error(f"Error in check_slabs_json: {e}", exc_info=True)
        return 0.0


def check_mp_structure(path_or_cif: str) -> float:
    """
    Check if the path points to a valid CIF file containing a structure from Materials Project.
    """
    logger.info("check_mp_structure")
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
        # Try first as a CIF string since that's more common
        try:
            logger.info(f"Trying to parse as CIF string first: {path_or_cif}")
            structure = Structure.from_str(path_or_cif, fmt="cif")
            logger.info("Successfully parsed as CIF string")
        except Exception as e:
            logger.info(f"Could not parse as CIF string: {e}")
            # If that fails, try as a file path
            if Path(path_or_cif).exists():
                logger.info(f"Input is a valid file path: {path_or_cif}")
                structure = Structure.from_file(path_or_cif)
            else:
                logger.error(
                    f"Input is neither a valid CIF string nor a file path: {path_or_cif}"
                )
                return 0.0

        return 1.0 if structure and len(structure) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error validating structure: {e}")
        return 0.0


def check_slab_structure(path_or_cif: str) -> float:  # TODO: better slab check.
    """
    Check if the path points to a valid CIF file containing a slab structure.

    Args:
        path_or_cif: Either a path to a CIF file or a CIF string

    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
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
    Check if the path points to a valid CIF file contains a CO2 molecule.

    Caveat: Does not work if there is more than one CO2 molecule.
    Also does not check for connectivity of the atoms.

    Args:
        path_or_cif: Either a path to a CIF file or a CIF string

    Returns:
        float: Score between 0.0 and 1.0
    """
    logger.info(f"Input path_or_cif: {path_or_cif}")
    try:
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

                return 1.0 if c_count == 1 and o_count == 2 else 0
            return 0  # Valid structure but missing C or O
        return 0
    except Exception as e:
        logger.error(f"Error validating molecule structure: {e}")
        return 0.0


def check_adsorption_structure(
    slab_elements: list[str], adsorbate_elements: list[str]
) -> Callable[[str], float]:
    """Returns a scoring function customized to given slab and adsorbate elements"""

    logger.info(
        f"Creating adsorption structure checker for slab_elements={slab_elements}, adsorbate_elements={adsorbate_elements}"
    )

    def score_fn(path_or_cif: str) -> float:
        try:
            logger.info(f"check_adsorption_structure: input={path_or_cif!r}")

            # Check if file exists
            if Path(path_or_cif).exists():
                logger.info(f"File exists at resolved path: {path_or_cif}")
                structure = Structure.from_file(path_or_cif)
                logger.info("Successfully loaded structure from file")
            else:
                logger.warning(f"File does not exist at resolved path: {path_or_cif}")
                # If resolved path doesn't exist, try original input as CIF string
                try:
                    logger.info("Trying to parse original input as CIF string")
                    structure = Structure.from_str(path_or_cif, fmt="cif")
                    logger.info("Successfully parsed original input as CIF")
                except Exception as e1:
                    logger.warning(f"Failed to parse original input as CIF: {e1}")
                    # If that fails too, try resolved input as CIF string
                    logger.info("Trying to parse resolved input as CIF string")
                    structure = Structure.from_str(path_or_cif, fmt="cif")
                    logger.info("Successfully parsed resolved input as CIF")

            if not structure:
                logger.error("Structure is None")
                return 0.0

            if len(structure) == 0:
                logger.error("Structure is empty")
                return 0.0

            logger.info(f"Structure loaded successfully with {len(structure)} sites")

            atoms = {site.specie.symbol for site in structure}
            logger.info(f"Found atoms in structure: {atoms}")

            has_slab = all(e in atoms for e in slab_elements)
            has_adsorbate = all(e in atoms for e in adsorbate_elements)

            logger.info(f"Required slab elements {slab_elements}: {has_slab}")
            logger.info(
                f"Required adsorbate elements {adsorbate_elements}: {has_adsorbate}"
            )

            if has_slab and has_adsorbate:
                logger.info(
                    "SUCCESS: Structure contains both slab and adsorbate elements"
                )
                return 1.0
            elif has_slab or has_adsorbate:
                logger.info(
                    "PARTIAL: Structure contains only slab or adsorbate elements"
                )
                return 0.0
            else:
                logger.error("FAILURE: Structure missing required elements")
                return 0.0

        except Exception as e:
            logger.error(f"Exception in check_adsorption_structure: {e}", exc_info=True)
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
        if not sites_json_or_path or not sites_json_or_path.strip():
            logger.warning("Empty input provided to check_adsorption_sites")
            return 0.0

        logger.info(f"check_adsorption_sites: input={sites_json_or_path!r}")

        # Try to resolve as path
        resolved_input = smart_resolve_path(sites_json_or_path.strip())
        logger.info(f"check_adsorption_sites: resolved={resolved_input!r}")

        # Try to load from file first
        if Path(resolved_input).is_file():
            with Path(resolved_input).open() as f:
                json_content = f.read()
        else:
            # If no file exists, treat as raw JSON string
            # Try original input first, then resolved input
            json_content = (
                sites_json_or_path
                if not resolved_input.endswith(".json")
                else resolved_input
            )

        sites = json.loads(json_content)

        # Check if the structure contains expected site types
        type_aliases = {
            "top": "ontop",
            "ontop": "ontop",
            "bridge": "bridge",
            "hollow": "hollow",
        }
        found_types = [alias for alias, canon in type_aliases.items() if alias in sites]

        if not found_types:
            return 0  # No recognized site types

        # Check if sites have coordinates
        has_coords = any(
            isinstance(sites.get(site_type), list) and len(sites.get(site_type)) > 0
            for site_type in found_types
        )

        if not has_coords:
            return 0  # Has site types but all are empty

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
            return 0  # Has coordinates but they're malformed

        return 1.0  # At least one site type has valid coordinates

    except Exception as e:
        logger.error(f"Error validating adsorption sites: {e}", exc_info=True)
        return 0.0
