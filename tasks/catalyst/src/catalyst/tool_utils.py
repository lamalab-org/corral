import math
from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from loguru import logger
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Utilities adopted from https://github.com/FAIR-Chem/fairchem


def generate_output_capture_code() -> str:
    """
    Generate the Python code string for capturing execution results.

    This utility function creates the code that gets appended to user code
    to capture variables and prepare them for JSON serialization.

    Returns:
        str: Python code string for variable capture
    """
    return """

import json
import types
import sys
from collections.abc import Iterable

result_ = {}

def is_json_serializable(obj, max_depth=10, current_depth=0):
    '''Check if object is JSON serializable with depth limit'''
    if current_depth > max_depth:
        return False

    try:
        # Handle basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True

        # Handle sequences (but not strings)
        if isinstance(obj, (list, tuple)):
            return all(is_json_serializable(item, max_depth, current_depth + 1) for item in obj)

        # Handle dictionaries
        if isinstance(obj, dict):
            return all(
                isinstance(k, str) and is_json_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()
            )

        # Quick test with actual JSON serialization for edge cases
        json.dumps(obj)
        return True
    except (TypeError, ValueError, RecursionError, OverflowError):
        return False

def safe_convert_to_serializable(obj):
    '''Convert common non-serializable types to serializable ones'''
    try:
        import numpy as np
        # Handle numpy arrays
        if hasattr(obj, '__module__') and obj.__module__ == 'numpy':
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
    except ImportError:
        pass

    # Handle sets
    if isinstance(obj, set):
        return list(obj)

    # Handle other iterables (but not strings/bytes)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return list(obj)
        except:
            pass

    return obj

# Try to capture from preferred variable names first
preferred_vars = ['output', 'result', 'filtered_data', 'processed_data', 'dataset']
captured = False

for var_name in preferred_vars:
    if var_name in locals():
        var_value = locals()[var_name]
        converted_value = safe_convert_to_serializable(var_value)
        if is_json_serializable(converted_value):
            result_[var_name] = converted_value
            captured = True
            break

# Fallback: capture any user-defined variables
if not captured:
    excluded = {
        '__builtins__', '__name__', '__doc__', '__package__', '__loader__',
        '__spec__', '__annotations__', '__cached__', '__file__',
        'json', 'types', 'sys', 'Iterable', 'result_', 'preferred_vars',
        'captured', 'excluded', 'local_vars', 'is_json_serializable',
        'safe_convert_to_serializable', 'var_name', 'var_value', 'converted_value'
    }

    local_vars = dict(locals())  # Create snapshot

    # Collect all suitable variables
    suitable_vars = {}
    for var_name, var_value in local_vars.items():
        if (not var_name.startswith('_') and
            var_name not in excluded and
            not isinstance(var_value, types.ModuleType) and
            not callable(var_value)):

            converted_value = safe_convert_to_serializable(var_value)
            if is_json_serializable(converted_value):
                suitable_vars[var_name] = converted_value

    # If we have suitable variables, include them all
    if suitable_vars:
        result_.update(suitable_vars)

print('EXECUTION_RESULT:', json.dumps(result_))
"""


def safe_convert_timeout(timeout) -> int:
    """
    Safely convert timeout parameter to integer.

    Args:
        timeout: Timeout value (could be string, int, float, or None)

    Returns:
        int: Valid timeout value
    """
    if timeout is None:
        return 300

    try:
        return int(float(timeout))  # Handle both "60" and "60.5"
    except (ValueError, TypeError):
        return 300  # Default fallback


def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the parent directory of a file path exists.

    Args:
        file_path: Path to a file
    """
    from pathlib import Path

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def parse_execution_output(stdout: str) -> tuple[dict, list[str]]:
    """
    Parse subprocess output to extract execution results and regular output.

    Args:
        stdout: Raw stdout from subprocess

    Returns:
        tuple: (execution_result dict, output_lines list)
    """
    import json
    from contextlib import suppress

    stdout_lines = stdout.strip().split("\n") if stdout.strip() else []
    execution_result = {}
    output_lines = []

    for line in stdout_lines:
        if line.startswith("EXECUTION_RESULT:"):
            with suppress(json.JSONDecodeError):
                execution_result = json.loads(line[17:])
        else:
            output_lines.append(line)

    return execution_result, output_lines


def load_structure(
    bulk_structure_path_or_string: str, from_path: bool = False
) -> Structure:
    """Loads pymatgen structure from CIF string."""
    from pymatgen.core import Structure

    try:
        if from_path:
            return Structure.from_file(bulk_structure_path_or_string)
        else:
            from pymatgen.core import Structure

            return Structure.from_str(bulk_structure_path_or_string, fmt="cif")
    except Exception as e:
        raise ValueError(f"Failed to parse structure CIF: {e}") from e


def load_adsorbate_mol(ads_cif_or_xyz: str) -> Molecule:
    """Loads pymatgen molecule from CIF or XYZ string."""
    try:
        # Prioritize XYZ for molecules
        return Molecule.from_str(ads_cif_or_xyz, fmt="xyz")
    except Exception:
        try:
            struct = Structure.from_str(ads_cif_or_xyz, fmt="cif")
            # Ensure coords is a list of lists/tuples
            coords = [list(site.coords) for site in struct]
            return Molecule(
                species=[site.species_string for site in struct],
                coords=coords,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse adsorbate CIF/XYZ: {e}") from e


def standardize_bulk(atoms_or_struct):
    """Standardizes a bulk structure."""
    if isinstance(atoms_or_struct, Structure):
        struct = atoms_or_struct
    else:  # Assume ase.Atoms
        struct = AseAtomsAdaptor.get_structure(atoms_or_struct)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    return sga.get_conventional_standard_structure()


def flip_struct(struct: Structure):
    """Flips a structure upside down (for slabs)."""
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.wrap()
    atoms.rotate(180, "x", rotate_cell=True, center="COM")
    if atoms.cell[2, 2] < 0.0:
        atoms.cell[2] = -atoms.cell[2]
    if np.cross(atoms.cell[0], atoms.cell[1])[2] < 0.0:
        atoms.cell[1] = -atoms.cell[1]
    atoms.center()
    atoms.wrap()
    return AseAtomsAdaptor.get_structure(atoms)


def calculate_coordination_of_bulk_atoms(bulk_atoms):
    """Calculates coordination numbers for atoms in the bulk."""
    voronoi_nn = VoronoiNN(tol=0.1)
    bulk_struct = AseAtomsAdaptor.get_structure(bulk_atoms)
    sga = SpacegroupAnalyzer(bulk_struct)
    try:
        sym_struct = sga.get_symmetrized_structure()
    except TypeError:  # Handle cases where symmetrization might fail
        logger.warning(
            "Could not get symmetrized structure for bulk coordination check, using original."
        )
        sym_struct = bulk_struct

    bulk_cn_dict = defaultdict(set)
    # Check if equivalent_indices is available, otherwise iterate all sites
    indices_to_check = []
    if hasattr(sym_struct, "equivalent_indices"):
        indices_to_check = [
            idx[0] for idx in sym_struct.equivalent_indices
        ]  # Check one from each set
    else:
        indices_to_check = list(range(len(sym_struct)))  # Check all if no symmetry info

    if not indices_to_check:  # Fallback if symmetry analysis failed completely
        indices_to_check = list(range(len(sym_struct)))

    for idx in indices_to_check:
        site = sym_struct[idx]
        try:
            cn = voronoi_nn.get_cn(sym_struct, idx, use_weights=True)
            cn = round(cn, 5)
            bulk_cn_dict[site.species_string].add(cn)
        except Exception as e:
            logger.warning(
                f"Could not calculate coordination for site {idx} ({site.species_string}): {e}"
            )
            # Assign a default or skip if CN calculation fails
            bulk_cn_dict[site.species_string].add(99)  # Use a placeholder if CN fails

    # Ensure all species have at least one CN value, even if calculation failed
    all_species = {site.species_string for site in bulk_struct}
    for specie in all_species:
        if not bulk_cn_dict[specie]:
            logger.warning(
                f"Assigning default high CN (99) to {specie} due to calculation errors."
            )
            bulk_cn_dict[specie].add(99)

    return bulk_cn_dict


def find_surface_atoms_by_height(surface_atoms):
    """Tags surface atoms based on height."""
    try:
        unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
        if (
            unit_cell_height < 1e-6
        ):  # Avoid division by zero for 2D materials or weird cells
            logger.warning(
                "Near-zero cell height in Z, using absolute Z coordinates for height tagging."
            )
            positions = surface_atoms.get_positions()
            max_height = max(pos[2] for pos in positions)
            threshold = max_height - 2.0
            return [0 if pos[2] < threshold else 1 for pos in positions]
        else:
            scaled_positions = surface_atoms.get_scaled_positions()
            scaled_max_height = max(sp[2] for sp in scaled_positions)
            # Wrap max height to handle cases where highest atom is near z=0
            scaled_max_height = scaled_max_height % 1.0
            # Adjust threshold calculation to handle wrapped coordinates correctly
            scaled_threshold = (scaled_max_height - 2.0 / unit_cell_height) % 1.0

            tags = []
            for sp in scaled_positions:
                scaled_z = sp[2] % 1.0
                # Check if scaled_z is within the "top" 2 Angstrom range, handling wrap-around
                is_top = False
                if scaled_max_height >= scaled_threshold:  # Normal case, no wrap
                    if scaled_z >= scaled_threshold and scaled_z <= scaled_max_height:
                        is_top = True
                else:  # Wrap-around case (e.g., max_height=0.95, threshold=0.8 -> range [0.8, 0.95])
                    # or (e.g., max_height=0.1, threshold=0.9 -> ranges [0.9, 1.0) U [0, 0.1])
                    if scaled_z >= scaled_threshold or scaled_z <= scaled_max_height:
                        is_top = True

                tags.append(1 if is_top else 0)
            return tags

    except Exception as e:
        logger.error(f"Error in height tagging: {e}. Defaulting all tags to 0.")
        return [0] * len(surface_atoms)


def find_surface_atoms_with_voronoi(bulk_atoms_ase, slab_atoms_ase):
    """Combines height and Voronoi coordination check for tagging."""
    height_tags = find_surface_atoms_by_height(slab_atoms_ase)
    bulk_cn_dict = calculate_coordination_of_bulk_atoms(bulk_atoms_ase)

    surface_struct = AseAtomsAdaptor.get_structure(slab_atoms_ase)
    voronoi_nn = VoronoiNN(tol=0.1)
    final_tags = list(height_tags)  # Start with height tags

    for idx, site in enumerate(surface_struct):
        # Only refine if height tag is 0 (potential bulk)
        if height_tags[idx] == 0:
            try:
                cn = voronoi_nn.get_cn(surface_struct, idx, use_weights=True)
                cn = round(cn, 5)
                min_bulk_cn = min(bulk_cn_dict[site.species_string])
                if cn < min_bulk_cn:
                    final_tags[idx] = 1  # It's undercoordinated -> surface atom
            except Exception as e:
                logger.warning(
                    f"Voronoi check failed for site {idx}, keeping height tag: {e}"
                )
                # Keep height tag if Voronoi fails
                # Keep height_tags[idx] which is 0

    # Sanity check: ensure at least one surface atom is tagged if slab is not tiny
    if len(slab_atoms_ase) > 5 and sum(final_tags) == 0:
        logger.warning(
            "No surface atoms tagged by Voronoi/height. Tagging highest atom."
        )
        positions = slab_atoms_ase.get_positions()
        highest_idx = np.argmax(positions[:, 2])
        if highest_idx < len(final_tags):
            final_tags[highest_idx] = 1

    return final_tags


def tile_atoms(atoms: Atoms, min_ab: float = 8.0):
    """Tiles ASE atoms object."""
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    na = (
        1 if a_length < 1e-6 else int(math.ceil(min_ab / a_length))
    )  # Handle 1D/2D cases
    nb = 1 if b_length < 1e-6 else int(math.ceil(min_ab / b_length))
    n_abc = (max(1, na), max(1, nb), 1)  # Ensure at least 1x1x1 repeat
    if n_abc == (1, 1, 1):
        return atoms.copy()  # No tiling needed
    else:
        return atoms.repeat(n_abc)


def set_fixed_atom_constraints(atoms):
    """Applies FixAtoms constraint based on tags."""
    atoms = atoms.copy()
    mask = [atom.tag == 0 for atom in atoms]  # Fix bulk atoms (tag 0)
    # Clear existing FixAtoms constraints before adding new ones
    new_constraints = [c for c in atoms.constraints if not isinstance(c, FixAtoms)]
    new_constraints.append(FixAtoms(mask=mask))
    atoms.constraints = new_constraints
    return atoms


def tile_tag_constrain(
    unit_slab_struct: Structure, bulk_atoms_ase: Atoms, min_ab: float = 8.0
) -> Atoms:
    """Applies the full tiling, tagging, and constraining sequence."""
    slab_atoms_ase = AseAtomsAdaptor.get_atoms(unit_slab_struct)
    # Ensure bulk_atoms_ase is ASE format
    if not isinstance(bulk_atoms_ase, Atoms):
        bulk_atoms_ase = AseAtomsAdaptor.get_atoms(standardize_bulk(bulk_atoms_ase))

    # Tagging requires untiled slab for coordination check relative to bulk
    tags = find_surface_atoms_with_voronoi(bulk_atoms_ase, slab_atoms_ase)
    slab_atoms_ase.set_tags(tags)

    # Tile the tagged atoms
    tiled_atoms = tile_atoms(slab_atoms_ase, min_ab)

    return set_fixed_atom_constraints(tiled_atoms)
