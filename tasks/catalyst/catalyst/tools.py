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


@tool
def get_molecule_structure(smiles: str, path_to_write_dir: str) -> str:
    """Get molecule structure from SMILES string and save it as a cif file.

    Args:
        smiles: SMILES string of the molecule
        path_to_write_dir: Path to directory where cif file will be saved
    """
    # Create a safe filename from SMILES
    import hashlib

    from pymatgen.io.babel import BabelMolAdaptor

    safe_name = hashlib.md5(smiles.encode()).hexdigest()[:10]

    # Generate 3D structure from SMILES
    adaptor = BabelMolAdaptor.from_string(smiles, "smi")
    adaptor.add_hydrogen()
    adaptor.make3D()

    # Convert to pymatgen Molecule
    mol = adaptor.pymatgen_mol

    # Save to file
    path = Path(path_to_write_dir) / f"{safe_name}_molecule.cif"
    mol.to(str(path), fmt="cif")

    return str(path)


@tool
def get_molecule_from_mp(mp_id: str, path_to_write_dir: str) -> Path:
    """Get pymatgen molecule strucutre from MP API given material id and save it as a cif file.

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
def add_molecule_to_slab(
    slab_path: str,
    molecule_path: str,
    height: float = 2.0,
    site: tuple[float, float, float] | None = None,
) -> str:
    """Add molecule to slab and save as a new cif file.

    Args:
        slab_path: Path to slab cif file
        molecule_path: Path to molecule cif file
        height: Height above the surface to place the molecule (Å)
        site: Optional (x,y,z) coordinates to place the molecule. If None, places at center.
    """
    import pickle

    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.core import Molecule, Structure

    # Load structures
    slab = Structure.from_file(slab_path)

    # Check if molecule file is cif or pkl
    if molecule_path.endswith(".cif"):
        # CIF files may contain structures or molecules
        try:
            molecule = Structure.from_file(molecule_path)
            # Convert to molecule if it's a structure
            molecule = Molecule(
                species=molecule.species, coords=molecule.cart_coords, charge=0
            )
        except Exception:
            # Try loading as a molecule directly
            molecule = Molecule.from_file(molecule_path)
    else:
        with Path(molecule_path).open("rb") as f:
            molecule = pickle.load(f)

    # Find adsorption sites if site not specified
    if site is None:
        finder = AdsorbateSiteFinder(slab)
        sites = finder.find_adsorption_sites()
        # Choose a top site by default
        site = sites["top"][0]

    # Add molecule to slab
    # Convert molecule to adsorbate format
    ads_struct = finder.add_adsorbate(molecule, site, height)

    # Save the combined structure
    output_path = slab_path.replace(".cif", "_with_molecule.cif")
    ads_struct.to(output_path, fmt="cif")

    return output_path


@tool
def optimize_structure(structure_path: str, method: str = "uff") -> str:
    """Perform a quick structural optimization using a force field.

    Args:
        structure_path: Path to structure file (.cif)
        method: Force field method ("uff" or "mmff94")
    """
    import ase.io
    from pymatgen.core import Structure
    from pymatgen.io.babel import BabelMolAdaptor

    # Try loading as Structure first
    try:
        struct = Structure.from_file(structure_path)
        # Convert to ASE Atoms
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(struct)

        # For periodic structures, use ASE's UFF implementation
        from ase.calculators.uff import UFF
        from ase.optimize import BFGS

        atoms.calc = UFF()
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05, steps=100)

        # Save optimized structure
        output_path = structure_path.replace(".cif", "_optimized.cif")
        ase.io.write(output_path, atoms)

    except Exception:
        # Try as a molecule
        try:
            # Load the molecule using OpenBabel
            adaptor = BabelMolAdaptor.from_file(structure_path)

            # Optimize the molecule
            adaptor.optimize(method)

            # Get the optimized molecule
            mol_opt = adaptor.pymatgen_mol

            # Save optimized molecule
            output_path = structure_path.replace(".cif", "_optimized.cif")
            mol_opt.to(output_path, fmt="cif")

        except Exception as e:
            raise ValueError(f"Could not optimize structure: {e}") from e

    return output_path


def create_tools() -> dict[str, Tool]:
    """Create all available tools"""
    return {
        "get_structure_from_mp": get_structure_from_mp,
        "create_pymatgen_structure_from_cif": create_pymatgen_structure_from_cif,
        "create_slab_from_structure": create_slab_from_structure,
        "get_molecule_structure": get_molecule_structure,
        "add_molecule_to_slab": add_molecule_to_slab,
        "optimize_structure": optimize_structure,
    }
