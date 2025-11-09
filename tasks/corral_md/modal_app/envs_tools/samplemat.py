from __future__ import annotations

from modal import Image

pymatgen_image = Image.debian_slim(python_version="3.12").pip_install(
    [
        "pymatgen",
    ]
)

with pymatgen_image.imports():
    from pymatgen.analysis.energy_calculators import EnergyCalculator
    from pymatgen.core import Structure


def calculate_lattice_energy(structure_file: str) -> float:
    """
    Calculate the lattice energy of a crystal structure.

    Args:
        structure_file: Path to structure file (CIF, POSCAR, etc.)

    Returns:
        Lattice energy in eV
    """
    try:
        structure = Structure.from_file(structure_file)
        calculator = EnergyCalculator()
        return calculator.get_energy(structure)
    except Exception:
        return None
