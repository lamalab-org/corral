import os
from pathlib import Path

from dotenv import load_dotenv

from corral.base import Tool
from corral.utils import tool

load_dotenv("../.env")


@tool
def get_structure_from_mp(mp_id: str, path_to_write_dir: str) -> Path:
    """Get pymatgen structure from MP API given material id and save it as a cif file.

    Args:
        mp_id: Materials Project id
        path_to_write_dir: Path to directory where cif file will be saved
    """
    from mp_api.client import MPRester

    with MPRester(os.getenv("MP_API_KEY")) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[str(mp_id)], fields=["structure"]
        )
        structure = docs[0].structure
        from pathlib import Path

        path = Path(path_to_write_dir) / f"{mp_id}_structure.cif"
        structure.to(path, fmt="cif")
        return path


@tool
def create_pymatgen_structure_from_cif(cif_path: str) -> str:
    """Create pymatgen structure from cif file given path to cif file and save it as a pickle file.

    Args:
        cif_path: Path to cif file
    """
    import pickle

    from pymatgen.core import Structure

    structure = Structure.from_file(cif_path)
    # cif path -> pickle path
    path = cif_path.replace(".cif", ".pkl")
    with Path(path).open("wb") as f:
        pickle.dump(structure, f)
    return path


@tool
def create_slab_from_structure(
    structure_path: str,
    miller_index: tuple = (1, 1, 1),
    min_slab_size: int = 12,
    min_vacuum_size: int = 5,
    primitive: bool = True,
) -> str:
    """Create slab from structure and save it as a cif file.

    Args:
        structure_path: Path to cif or pickle file containing structure
        miller_index: Miller index of the surface
        min_slab_size: Minimum slab size
        min_vacuum_size: Minimum vacuum size
        primitive: Whether to create a primitive slab
    """
    import pickle

    from pymatgen.core.structure import Structure
    from pymatgen.core.surface import SlabGenerator

    # Load structure from cif or pickle
    # if miller_index is None:
    #     miller_index = (1, 0, 0)
    if structure_path.endswith(".cif"):
        structure = Structure.from_file(structure_path)
    elif structure_path.endswith(".pkl"):
        with Path(structure_path).open("rb") as f:
            structure = pickle.load(f)
    else:
        raise ValueError("Invalid structure file format")
    # Create slab
    slab_gen = SlabGenerator(
        structure, miller_index, min_slab_size, min_vacuum_size, primitive=primitive
    )
    slab = slab_gen.get_slab()
    slab = slab.get_orthogonal_c_slab().get_sorted_structure()
    # cif path -> pickle path
    path = structure_path.replace(".cif", "_slab.cif").replace(".pkl", "_slab.cif")
    slab.to(path, fmt="cif")
    return path


def create_tools() -> dict[str, Tool]:
    """Create all available tools"""
    return {
        "get_structure_from_mp": get_structure_from_mp,
        "create_pymatgen_structure_from_cif": create_pymatgen_structure_from_cif,
        "create_slab_from_structure": create_slab_from_structure,
    }
