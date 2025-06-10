import json
import os
from collections.abc import Callable
from pathlib import Path

import joblib
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


def check_slab_structure(path_or_cif: str) -> float:  # TODO: better slab check.
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
        logger.error(f"Error validating adsorption sites: {e}")
        return 0.0


### OCP Hard scoring function


def bulk_diversity_score(polymorph_data_path):
    """
    Score 1.0 if:
    - At least 3 different polymorphs
    - At least 2 different space groups
    - At least 1 metastable structure with E_above_hull < 0.2 eV/atom
    """
    import json

    if not Path(polymorph_data_path).exists():
        return 0.0

    try:
        with Path(polymorph_data_path).open() as f:
            polymorph_data = json.load(f)

        if not polymorph_data or len(polymorph_data) < 3:
            return 0.0

        # Check space group diversity
        space_groups = set()
        metastable_found = False

        for entry in polymorph_data:
            if "space_group" in entry:
                space_groups.add(entry["space_group"])

            # Check for metastable structures
            if "energy_above_hull" in entry and 0 < entry["energy_above_hull"] < 0.2:
                metastable_found = True

        if len(space_groups) >= 2 and metastable_found:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def database_structure_quality(database_path):
    """
    Score 1.0 if:
    - Database exists with bulk and slab tables
    - Tables contain essential columns
    - At least one valid entry in each table
    """
    import sqlite3
    from pathlib import Path

    if not Path(database_path).exists():
        return 0.0

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]

        bulk_table = next((t for t in tables if "bulk" in t.lower()), None)
        slab_table = next((t for t in tables if "slab" in t.lower()), None)

        if not bulk_table or not slab_table:
            conn.close()
            return 0.0

        # Check for essential columns in bulk table
        cursor.execute(f"PRAGMA table_info({bulk_table})")
        bulk_columns = [col[1] for col in cursor.fetchall()]
        bulk_essential = {"material_id", "cif", "energy_above_hull"}

        # Check for essential columns in slab table
        cursor.execute(f"PRAGMA table_info({slab_table})")
        slab_columns = [col[1] for col in cursor.fetchall()]
        slab_essential = {"slab_id", "miller_index", "cif"}

        # Check if tables have data
        cursor.execute(f"SELECT COUNT(*) FROM {bulk_table}")
        bulk_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM {slab_table}")
        slab_count = cursor.fetchone()[0]

        conn.close()

        # Score 1.0 if all criteria are met
        bulk_columns_ok = all(col in bulk_columns for col in bulk_essential)
        slab_columns_ok = all(col in slab_columns for col in slab_essential)

        if bulk_columns_ok and slab_columns_ok and bulk_count > 0 and slab_count > 0:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def miller_indices_coverage(miller_indices_path):
    """
    Score 1.0 if:
    - At least 5 symmetrically distinct Miller indices
    - Includes low-index planes {100}, {110}, and {111}
    """
    import json

    if not Path(miller_indices_path).exists():
        return 0.0

    try:
        with Path(miller_indices_path).open() as f:
            data = json.load(f)

        # Extract Miller indices from various possible formats
        if isinstance(data, list):
            miller_indices = data
        elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            for value in data.values():
                if isinstance(value, list) and len(value) > 0:
                    miller_indices = value
                    break
            else:
                return 0.0
        else:
            return 0.0

        # Process indices to standard format
        processed_indices = []
        for idx in miller_indices:
            if isinstance(idx, str):
                try:
                    if "," in idx:
                        processed_indices.append(
                            tuple(map(int, idx.strip("()[]").split(",")))
                        )
                    else:
                        processed_indices.append(tuple(int(i) for i in idx))
                except Exception:
                    continue
            elif isinstance(idx, list):
                processed_indices.append(tuple(idx))
            elif isinstance(idx, tuple):
                processed_indices.append(idx)

        # Check if we have enough indices
        if len(processed_indices) < 5:
            return 0.0

        # Check for low-index planes
        low_index_families = [
            {(1, 0, 0), (0, 1, 0), (0, 0, 1)},  # {100}
            {(1, 1, 0), (1, 0, 1), (0, 1, 1)},  # {110}
            {(1, 1, 1)},  # {111}
        ]

        family_found = [False, False, False]

        for idx in processed_indices:
            normalized = tuple(sorted([abs(i) for i in idx], reverse=True))
            for i, family in enumerate(low_index_families):
                if normalized in family:
                    family_found[i] = True

        if all(family_found) and len(processed_indices) >= 5:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def slab_diversity_score(slab_terminations_path):
    """
    Score 1.0 if:
    - Slabs for at least 3 different Miller indices
    - At least 2 terminations per Miller index for 70% of indices
    - All structures are valid CIFs
    """
    import json
    from collections import defaultdict

    if not Path(slab_terminations_path).exists():
        return 0.0

    try:
        with Path(slab_terminations_path).open() as f:
            slab_data = json.load(f)

        if not slab_data:
            return 0.0

        miller_indices = set()
        terminations = defaultdict(set)
        valid_slabs = 0

        for slab_info in slab_data.values():
            # Extract Miller index
            if "miller_index" in slab_info:
                miller_idx = slab_info["miller_index"]
                if isinstance(miller_idx, list):
                    miller_idx = tuple(miller_idx)
                miller_indices.add(str(miller_idx))

                # Track terminations
                if "termination" in slab_info:
                    terminations[str(miller_idx)].add(slab_info["termination"])

            # Check structure validity
            if "cif" in slab_info and len(slab_info["cif"]) > 100:
                valid_slabs += 1

        if len(miller_indices) < 3:
            return 0.0

        # Check termination diversity
        diverse_count = sum(1 for terms in terminations.values() if len(terms) >= 2)
        termination_ratio = diverse_count / len(miller_indices) if miller_indices else 0

        # All slabs must be valid
        all_valid = valid_slabs == len(slab_data)

        if len(miller_indices) >= 3 and termination_ratio >= 0.7 and all_valid:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def relaxation_sampling_efficiency(relaxed_structures_path):
    """
    Score 1.0 if:
    - Relaxed structures cover at least 3 different Miller indices
    - Energy values provided for all structures
    - 10-30 structures total (optimal sampling size)
    """
    import json

    if not Path(relaxed_structures_path).exists():
        return 0.0

    try:
        with Path(relaxed_structures_path).open() as f:
            relaxed_data = json.load(f)

        if not relaxed_data:
            return 0.0

        miller_indices = set()
        structures_with_energy = 0
        total_structures = len(relaxed_data)

        for struct_info in relaxed_data.values():
            # Track Miller indices
            if "miller_index" in struct_info:
                miller_idx = struct_info["miller_index"]
                if isinstance(miller_idx, list):
                    miller_idx = tuple(miller_idx)
                miller_indices.add(str(miller_idx))

            # Check energy values
            if ("energy" in struct_info and struct_info["energy"] is not None) or (
                "energy_per_atom" in struct_info
                and struct_info["energy_per_atom"] is not None
            ):
                structures_with_energy += 1

        # Check criteria
        diverse_miller = len(miller_indices) >= 3
        complete_energy = structures_with_energy == total_structures
        optimal_sampling = 10 <= total_structures <= 30

        if diverse_miller and complete_energy and optimal_sampling:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def adsorbate_library_completeness(adsorbate_library_path):
    """
    Score 1.0 if:
    - Library includes at least 7 essential adsorbates (CO, H, O, OH, etc.)
    - All adsorbates have gas-phase energy values
    """
    import json

    if not Path(adsorbate_library_path).exists():
        return 0.0

    try:
        with Path(adsorbate_library_path).open() as f:
            adsorbate_data = json.load(f)

        if not adsorbate_data:
            return 0.0

        essential_adsorbates = {"CO", "H", "O", "OH", "N", "NH", "CO2"}
        found_adsorbates = set()
        energy_count = 0

        for ads_id, ads_info in adsorbate_data.items():
            # Identify adsorbate from formula or ID
            if "formula" in ads_info:
                formula = ads_info["formula"].upper()
                for essential in essential_adsorbates:
                    if essential.upper() == formula or essential.upper() in formula:
                        found_adsorbates.add(essential)
            else:
                for essential in essential_adsorbates:
                    if essential.lower() in ads_id.lower():
                        found_adsorbates.add(essential)

            # Check energy values
            if any(key in ads_info for key in ["energy", "gas_phase_energy"]):
                energy_count += 1

        essential_count = len(found_adsorbates)
        all_energies = energy_count == len(adsorbate_data)

        if essential_count >= 7 and all_energies:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def adsorption_site_mapping_quality(adsorption_sites_path):
    """
    Score 1.0 if:
    - Maps include all basic site types (top, bridge, hollow)
    - All sites have valid 3D coordinates
    - Consistent site classification across all surfaces
    """
    import json

    if not Path(adsorption_sites_path).exists():
        return 0.0

    try:
        with Path(adsorption_sites_path).open() as f:
            site_data = json.load(f)

        if not site_data:
            return 0.0

        # Check for basic site types
        basic_sites = {"top", "bridge", "hollow"}
        found_sites = {k.lower() for k in site_data}

        all_basic_found = all(
            any(basic in found for found in found_sites) for basic in basic_sites
        )

        # Check coordinate validity
        valid_coords = True
        for sites in site_data.values():
            if not isinstance(sites, list):
                valid_coords = False
                break

            for site in sites:
                if not isinstance(site, list) or len(site) != 3:
                    valid_coords = False
                    break
                if not all(isinstance(coord, int | float) for coord in site):
                    valid_coords = False
                    break

        if all_basic_found and valid_coords:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def configuration_sampling_score(configs_path):
    """
    Score 1.0 if:
    - Configurations cover all basic site types (top, bridge, hollow)
    - At least 2 different orientations per adsorbate
    - All structures are valid CIFs
    """
    import json
    from collections import defaultdict

    if not Path(configs_path).exists():
        return 0.0

    try:
        with Path(configs_path).open() as f:
            config_data = json.load(f)

        if not config_data:
            return 0.0

        site_types = set()
        orientations = defaultdict(set)
        valid_count = 0

        for config_info in config_data.values():
            # Track site types
            if "site_type" in config_info:
                site_types.add(config_info["site_type"].lower())

            # Track orientations
            if "orientation" in config_info:
                if "site_type" in config_info:
                    orientations[config_info["site_type"]].add(
                        str(config_info["orientation"])
                    )
                else:
                    orientations["unknown"].add(str(config_info["orientation"]))

            # Check structure validity
            if "cif" in config_info and len(config_info["cif"]) > 100:
                valid_count += 1

        # Check basic site type coverage
        basic_sites = {"top", "bridge", "hollow"}
        basic_covered = all(
            any(basic in site for site in site_types) for basic in basic_sites
        )

        # Check orientation diversity
        diverse_orientations = all(
            len(orients) >= 2 for orients in orientations.values()
        )

        # Check structure validity
        all_valid = valid_count == len(config_data)

        if basic_covered and diverse_orientations and all_valid:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


def energy_calculation_accuracy(adsorption_energies_path):
    """
    Score 1.0 if:
    - Adsorption energies calculated for all configurations
    - Consistent reference states used
    - Energy values are physically reasonable (-5 to 5 eV range)
    """
    import json

    if not Path(adsorption_energies_path).exists():
        return 0.0

    try:
        with Path(adsorption_energies_path, "r").open() as f:
            energy_data = json.load(f)

        if not energy_data:
            return 0.0

        # Check for complete energy data
        all_have_energies = True
        consistent_references = True
        reasonable_values = True

        for config_info in energy_data.values():
            # Check for adsorption energy
            if "adsorption_energy" not in config_info:
                all_have_energies = False
                break

            # Check for reference energies
            reference_keys = {
                "adsorbate_slab_energy",
                "slab_energy",
                "adsorbate_gas_energy",
            }
            if not all(key in config_info for key in reference_keys):
                consistent_references = False
                break

            # Check for reasonable energy values
            e_ads = config_info["adsorption_energy"]
            if not isinstance(e_ads, int | float) or not (-5.0 <= e_ads <= 5.0):
                reasonable_values = False
                break

        if all_have_energies and consistent_references and reasonable_values:
            return 1.0
        return 0.0

    except Exception:
        return 0.0


## Scoring based on comparison with ground truth


def compare_with_ground_truth(
    generated_path, ground_truth_path, comparison_mode="strict", tolerance=0.05
):
    """
    Generic function to compare a generated JSON file with a ground truth JSON file.

    Args:
        generated_path: Path to the generated JSON file
        ground_truth_path: Path to the ground truth JSON file
        comparison_mode: Mode of comparison:
                        - "strict": Exact matching of structure and values
                        - "keys": Only check if all required keys exist
                        - "numerical": Compare numerical values with tolerance
                        - "subset": Check if generated contains at least a subset of ground truth
        tolerance: Tolerance for numerical comparisons (as a fraction)

    Returns:
        1.0 if generated matches ground truth according to the comparison mode, 0.0 otherwise
    """
    import json

    # Check if both files exist
    if not Path(generated_path).exists() or not Path(ground_truth_path).exists():
        return 0.0

    try:
        # Load both files
        with Path(generated_path, "r").open() as f:
            generated = json.load(f)

        with Path(ground_truth_path, "r").open() as f:
            ground_truth = json.load(f)

        # Handle different comparison modes
        if comparison_mode == "strict":
            # Direct equality check
            return 1.0 if generated == ground_truth else 0.0

        elif comparison_mode == "keys":
            # Check if all required keys from ground truth exist in generated
            if isinstance(ground_truth, dict) and isinstance(generated, dict):
                missing_keys = [key for key in ground_truth if key not in generated]
                return 1.0 if not missing_keys else 0.0
            elif isinstance(ground_truth, list) and isinstance(generated, list):
                # For lists, check if they have the same length
                if len(ground_truth) != len(generated):
                    return 0.0
                # If items are dictionaries, check keys for each item
                if all(isinstance(item, dict) for item in ground_truth):
                    for i, gt_item in enumerate(ground_truth):
                        if i >= len(generated):
                            return 0.0
                        missing_keys = [
                            key for key in gt_item if key not in generated[i]
                        ]
                        if missing_keys:
                            return 0.0
                return 1.0
            else:
                return 0.0

        elif comparison_mode == "numerical":
            # Compare numerical values with tolerance
            def compare_with_tolerance(val1, val2, tol):
                if isinstance(val1, int | float) and isinstance(val2, int | float):
                    # Use relative tolerance for non-zero values
                    if abs(val2) > 1e-10:
                        return abs((val1 - val2) / val2) <= tol
                    # Use absolute tolerance for values near zero
                    else:
                        return abs(val1 - val2) <= tol
                elif isinstance(val1, dict) and isinstance(val2, dict):
                    # Compare dictionaries recursively
                    return all(
                        k in val1 and compare_with_tolerance(val1[k], val2[k], tol)
                        for k in val2
                    )
                elif isinstance(val1, list) and isinstance(val2, list):
                    # Compare lists recursively
                    return len(val1) == len(val2) and all(
                        compare_with_tolerance(v1, v2, tol)
                        for v1, v2 in zip(val1, val2, strict=False)
                    )
                else:
                    # For non-numerical values, use strict equality
                    return val1 == val2

            return (
                1.0
                if compare_with_tolerance(generated, ground_truth, tolerance)
                else 0.0
            )

        elif comparison_mode == "subset":
            # Check if generated contains at least the required subset
            def is_subset(generated_val, ground_truth_val):
                if isinstance(ground_truth_val, dict) and isinstance(
                    generated_val, dict
                ):
                    # Check if all required keys and values match
                    for k, v in ground_truth_val.items():
                        if k not in generated_val or not is_subset(generated_val[k], v):
                            return False
                    return True
                elif isinstance(ground_truth_val, list) and isinstance(
                    generated_val, list
                ):
                    # For lists, check if all ground truth items are in generated
                    # This is a simplified approach that works for primitive values
                    for gt_item in ground_truth_val:
                        if isinstance(gt_item, dict | list):
                            # For complex items, check if any generated item is a superset
                            if not any(
                                is_subset(gen_item, gt_item)
                                for gen_item in generated_val
                            ):
                                return False
                        else:
                            # For simple items, just check if it's in the list
                            if gt_item not in generated_val:
                                return False
                    return True
                else:
                    # For primitive values, check equality
                    return generated_val == ground_truth_val

            return 1.0 if is_subset(generated, ground_truth) else 0.0

        else:
            raise ValueError(f"Unknown comparison mode: {comparison_mode}")

    except Exception:
        return 0.0


def compare_adsorption_energies(generated_path, ground_truth_path, tolerance=0.1):
    """
    Compare calculated adsorption energies with ground truth values.

    Args:
        generated_path: Path to generated adsorption energies JSON
        ground_truth_path: Path to ground truth adsorption energies
        tolerance: Tolerance for energy differences (eV)

    Returns:
        1.0 if energies match within tolerance, 0.0 otherwise
    """
    import json

    if not Path(generated_path).exists() or not Path(ground_truth_path).exists():
        return 0.0

    try:
        # Load both files
        with Path(generated_path).open() as f:
            generated = json.load(f)

        with Path(ground_truth_path).open() as f:
            ground_truth = json.load(f)

        # Check all configurations from ground truth
        for config_id, gt_data in ground_truth.items():
            # Skip if config not in generated data
            if config_id not in generated:
                return 0.0

            # Get adsorption energies
            if (
                "adsorption_energy" not in gt_data
                or "adsorption_energy" not in generated[config_id]
            ):
                return 0.0

            gt_energy = gt_data["adsorption_energy"]
            gen_energy = generated[config_id]["adsorption_energy"]

            # Check if energies match within tolerance
            if abs(gt_energy - gen_energy) > tolerance:
                return 0.0

        return 1.0

    except Exception:
        return 0.0


def bulk_diversity_comparisopn_score(polymorph_data_path, ground_truth_path=None):
    """
    Score bulk diversity with optional ground truth comparison.

    Args:
        polymorph_data_path: Path to the generated polymorph data JSON
        ground_truth_path: Optional path to ground truth data JSON
    """
    # If ground truth is provided, use it for comparison
    return compare_with_ground_truth(
        polymorph_data_path,
        ground_truth_path,
        comparison_mode="subset",  # Allow additional polymorphs beyond ground truth
        tolerance=0.1,  # 10% tolerance for numerical values
    )


def ml_pipeline_score(model_path: str) -> float:
    """
    Comprehensive scoring function for the single-task ML pipeline.

    Evaluates the entire pipeline from data generation to model evaluation.
    This function looks for evidence of all pipeline steps and evaluates
    the final model quality.
    """
    try:
        if not Path(model_path).exists():
            return 0.0

        # Try to load the model
        try:
            import joblib

            model = joblib.load(model_path)
            if not hasattr(model, "predict"):
                return 0.2
        except Exception:
            return 0.1

        score = 0.3  # Base score for model existence

        # Look for evidence of dataset creation
        work_dir = Path(model_path).parent

        # Check for evaluation results
        eval_files = list(work_dir.glob("*evaluation*.json")) + list(
            work_dir.glob("*results*.json")
        )
        if eval_files:
            try:
                # Load the most recent evaluation file
                latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
                with latest_eval.open() as f:
                    eval_results = json.load(f)

                # Check model performance
                if "evaluation_metrics" in eval_results:
                    metrics = eval_results["evaluation_metrics"]
                    r2 = metrics.get("r2", 0)
                    mae = metrics.get("mae", float("inf"))

                    if r2 >= 0.8 and mae <= 0.3:
                        score += 0.3
                    elif r2 >= 0.6 and mae <= 0.5:
                        score += 0.2
                    elif r2 >= 0.4:
                        score += 0.1

                # Check for cross-validation
                if "cross_validation_results" in eval_results:
                    score += 0.1

            except Exception:
                pass

        return min(1.0, score)

    except Exception as e:
        logger.error(f"Error scoring comprehensive ML pipeline: {e}")
        return 0.0


def composition_list_quality(compositions_path: str) -> float:
    """
    Score the quality of generated oxide composition list.

    Criteria:
    - At least 15 valid oxide compositions
    - Diverse element coverage (at least 8 different elements)
    - Valid chemical formulas
    - Mix of binary and ternary oxides
    """
    import re

    try:
        if not Path(compositions_path).exists():
            return 0.0

        with Path(compositions_path).open() as f:
            data = json.load(f)

        compositions = data.get("compositions", [])

        if len(compositions) < 10:
            return 0

        # Check for valid oxide formulas
        valid_oxides = []
        elements_found = set()
        binary_count = 0
        ternary_count = 0

        for comp in compositions:
            # Check if it contains oxygen and valid elements
            if re.search(r"O\d*$", comp) or "O" in comp:
                # Extract elements (simple pattern)
                elements = re.findall(r"[A-Z][a-z]?", comp)
                if "O" in elements:
                    valid_oxides.append(comp)
                    elements_found.update(elem for elem in elements if elem != "O")

                    # Count binary vs ternary
                    non_oxygen = [e for e in elements if e != "O"]
                    if len(non_oxygen) == 1:
                        binary_count += 1
                    elif len(non_oxygen) >= 2:
                        ternary_count += 1

        # Scoring criteria
        valid_ratio = len(valid_oxides) / len(compositions)
        has_both_types = binary_count > 0 and ternary_count > 0

        if valid_ratio > 0.5 and has_both_types:
            return 1.0

    except Exception as e:
        logger.error(f"Error scoring composition list: {e}")
        return 0.0


def polymorph_retrieval_success(retrieval_results_path: str) -> float:
    """
    Score the success of batch polymorph retrieval.

    Criteria:
    - At least 80% of compositions have polymorphs retrieved
    - Total polymorphs >= 50
    - Reasonable distribution across compositions
    """
    try:
        if not Path(retrieval_results_path).exists():
            return 0.0

        with Path(retrieval_results_path).open() as f:
            results = json.load(f)

        successful = len(results.get("successful_compositions", []))
        failed = len(results.get("failed_compositions", []))
        total_compositions = successful + failed
        total_polymorphs = results.get("total_polymorphs", 0)

        if total_compositions == 0:
            return 0.0

        success_rate = successful / total_compositions

        score = 0.0

        # Success rate scoring
        if success_rate >= 0.8:
            score += 0.4
        elif success_rate >= 0.6:
            score += 0.3
        elif success_rate >= 0.4:
            score += 0.2

        # Total polymorphs scoring
        if total_polymorphs >= 100:
            score += 0.3
        elif total_polymorphs >= 50:
            score += 0.2
        elif total_polymorphs >= 20:
            score += 0.1

        # Distribution check (average polymorphs per successful composition)
        if successful > 0:
            avg_per_comp = total_polymorphs / successful
            if 3 <= avg_per_comp <= 10:
                score += 0.3
            elif 2 <= avg_per_comp <= 12:
                score += 0.2

        return min(1.0, score)

    except Exception as e:
        logger.error(f"Error scoring polymorph retrieval: {e}")
        return 0.0


def score_polymorph_dataset(
    consolidated_json_path: str,
) -> float:
    """
    Analyzes a consolidated JSON file of polymorphs to identify if there are
    compositions with multiple polymorphs.

    Args:
        consolidated_json_path: Path to the consolidated JSON file
                                (e.g., created by consolidate_polymorph_datasets).

    Returns:
        A float: 1.0 if at least one composition with multiple polymorphs is found,
        otherwise 0.0. Returns 0.0 if the file is not found or an error occurs.
    """
    try:
        # Check if the consolidated JSON file exists
        # We need Path imported to check for file existence
        from pathlib import Path

        consolidated_json_file = Path(consolidated_json_path)
        if not consolidated_json_file.exists():
            logger.error(
                f"Consolidated JSON file not found at: {consolidated_json_path}"
            )
            return 0.0

        with consolidated_json_file.open("r") as f:
            all_polymorphs_data = json.load(f)

        # Dictionary to store polymorphs grouped by composition
        polymorphs_by_composition = {}
        for polymorph in all_polymorphs_data:
            composition = polymorph.get("source_composition")
            if composition:
                if composition not in polymorphs_by_composition:
                    polymorphs_by_composition[composition] = []
                polymorphs_by_composition[composition].append(polymorph)
            else:
                logger.warning(
                    f"Polymorph without 'source_composition' found: {polymorph.get('material_id', 'N/A')}"
                )

        # Check if any composition has multiple polymorphs
        for polymorphs in polymorphs_by_composition.values():
            if len(polymorphs) > 1:
                # If we find at least one composition with multiple polymorphs, return 1.0
                return 1.0

        # If the loop completes and no composition with multiple polymorphs is found
        return 0.0

    except Exception as e:
        logger.error(f"Error in score_polymorph_dataset: {e}", exc_info=True)
        return 0.0


def ml_dataset_preparation_quality_binary(ml_metadata_path: str) -> int:
    """
    Scores the quality of ML dataset preparation as binary (0 for fail, 1 for pass).

    A dataset preparation passes (1) if it meets the following criteria:
    - Both train and test files exist.
    - Sufficient sample sizes: at least 40 training samples and 10 test samples.
    - A reasonable number of features: at least 10 features.

    Args:
        ml_metadata_path: Path to the ML dataset metadata JSON file.

    Returns:
        1 if the dataset preparation quality meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        if not Path(ml_metadata_path).exists():
            logger.info(f"Metadata file not found: {ml_metadata_path}")
            return 0

        with Path(ml_metadata_path).open() as f:
            metadata = json.load(f)

        # Criteria 1: Check train/test files exist
        train_path = metadata.get("train_path", "")
        test_path = metadata.get("test_path", "")
        if not (
            train_path
            and Path(train_path).exists()
            and test_path
            and Path(test_path).exists()
        ):
            logger.info(
                "Binary check failed: Train or test files are missing or paths are empty."
            )
            # Added more specific logging for clarity
            if not train_path:
                logger.info("train_path is empty in metadata.")
            elif not Path(train_path).exists():
                logger.info("Train file does not exist at: {train_path}")
            if not test_path:
                logger.info("test_path is empty in metadata.")
            elif not Path(test_path).exists():
                logger.info("Test file does not exist at: {test_path}")

            return 0

        # Criteria 2: Check sufficient sample sizes
        train_samples = metadata.get("train_samples", 0)
        test_samples = metadata.get("test_samples", 0)
        if not (train_samples >= 40 and test_samples >= 10):
            logger.info(
                f"Binary check failed: Insufficient sample sizes (Train: {train_samples}, Test: {test_samples})."
            )
            return 0

        # Criteria 3: Check reasonable feature count
        feature_count = metadata.get("feature_count", 0)
        if not (feature_count >= 10):
            logger.info(
                f"Binary check failed: Insufficient feature count ({feature_count})."
            )
            return 0

        logger.info("Binary check passed: ML dataset preparation quality is good.")
        return 1

    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in {ml_metadata_path}")
        return 0
    except Exception as e:
        logger.error(f"Error scoring ML dataset preparation quality: {e}")
        return 0


def model_training_success_binary(model_path: str) -> int:
    """
    Scores the success of XGBoost model training as binary (0 for fail, 1 for pass).

    Model training passes (1) if it meets the following criteria:
    - A valid model file exists and can be loaded.
    - Associated training results indicate successful training.
    - Reasonable performance metrics: R-squared (r2) >= 0.6 and Mean Absolute Error (mae) <= 0.5.

    Args:
        model_path: Path to the trained model file (e.g., .pkl).

    Returns:
        1 if the model training success meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        if not Path(model_path).exists():
            logger.info(f"Model file not found: {model_path}")
            return 0

        # Criteria 1: Try to load the model
        try:
            model = joblib.load(model_path)
            if not hasattr(model, "predict"):
                logger.info(
                    "Binary check failed: Loaded model does not have a 'predict' method."
                )
                return 0
        except Exception as e:
            logger.info(
                f"Binary check failed: Could not load the model from {model_path}. Error: {e}"
            )
            return 0

        # Criteria 2 & 3: Look for associated results file and check metrics
        results_path = str(model_path).replace(".pkl", "_training_results.json")
        if not Path(results_path).exists():
            results_path = str(model_path).replace(
                ".pkl", "_predictions.json"
            )  # Try alternative naming

        if Path(results_path).exists():
            try:
                with Path(results_path).open() as f:
                    results = json.load(f)

                test_metrics = results.get("test_metrics", {})
                test_r2 = test_metrics.get("r2", 0)
                test_mae = test_metrics.get("mae", float("inf"))

                if not (test_r2 >= 0.5 and test_mae <= 0.5):
                    logger.info(
                        f"Binary check failed: Performance metrics below threshold (R2: {test_r2}, MAE: {test_mae})."
                    )
                    return 0
            except json.JSONDecodeError:
                logger.info(
                    f"Binary check failed: Invalid JSON in results file: {results_path}"
                )
                return 0
            except Exception as e:
                logger.info(
                    f"Binary check failed: Error processing results file {results_path}. Error: {e}"
                )
                return 0
        else:
            logger.info(
                "Binary check failed: No associated training results file found."
            )
            return 0

        logger.info("Binary check passed: Model training success criteria met.")
        return 1

    except Exception as e:
        logger.error(f"Error scoring model training success: {e}")
        return 0


def model_evaluation_completeness_binary(evaluation_results_path: str) -> int:
    """
    Scores the completeness of model evaluation as binary (0 for fail, 1 for pass).

    Model evaluation completeness passes (1) if it meets the following criteria:
    - An evaluation results file exists.
    - All required basic evaluation metrics (mae, rmse, r2) are present.
    - Performance quality: R-squared (r2) in evaluation metrics is at least 0.7.
    - Cross-validation results are present, including mean and standard deviation for r2.
    - Feature importance analysis is included in the evaluation metrics.
    - Detailed analysis is present (error_analysis and prediction_range).

    Args:
        evaluation_results_path: Path to the model evaluation results JSON file.

    Returns:
        1 if the model evaluation completeness meets the defined passing criteria,
        0 otherwise (including errors).
    """
    try:
        if not Path(evaluation_results_path).exists():
            logger.info(f"Evaluation results file not found: {evaluation_results_path}")
            return 0

        with Path(evaluation_results_path).open() as f:
            results = json.load(f)

        # Criteria 1 & 2: Check for basic evaluation metrics and all required metrics
        if "evaluation_metrics" not in results:
            logger.info("Binary check failed: 'evaluation_metrics' not found.")
            return 0
        metrics = results["evaluation_metrics"]
        required_metrics = ["mae", "rmse", "r2"]
        if not all(metric in metrics for metric in required_metrics):
            logger.info(
                "Binary check failed: Not all required metrics (mae, rmse, r2) are present."
            )
            return 0

        # Criteria 3: Check performance quality (r2 >= 0.7)
        if not ("r2" in metrics and metrics["r2"] >= 0.7):
            logger.info(
                f"Binary check failed: R2 ({metrics.get('r2', 'N/A')}) is below 0.7."
            )
            return 0

        # Criteria 4: Check for cross-validation results
        if "cross_validation_results" not in results:
            logger.info("Binary check failed: 'cross_validation_results' not found.")
            return 0
        cv_results = results["cross_validation_results"]
        if not ("r2_mean" in cv_results and "r2_std" in cv_results):
            logger.info(
                "Binary check failed: Cross-validation results missing 'r2_mean' or 'r2_std'."
            )
            return 0

        # Criteria 5: Check for feature importance
        if "feature_importance" not in metrics:
            logger.info(
                "Binary check failed: 'feature_importance' not found in evaluation metrics."
            )
            return 0

        # Criteria 6: Check for detailed analysis
        if not ("error_analysis" in metrics and "prediction_range" in metrics):
            logger.info(
                "Binary check failed: Detailed analysis (error_analysis or prediction_range) missing."
            )
            return 0

        logger.info("Binary check passed: Model evaluation completeness criteria met.")
        return 1

    except json.JSONDecodeError:
        logger.error(f"Error: Invalid JSON format in {evaluation_results_path}")
        return 0
    except Exception as e:
        logger.error(f"Error scoring model evaluation completeness: {e}")
        return 0
