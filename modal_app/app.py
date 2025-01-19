from __future__ import annotations

import os

from envs_tools.samplemat import (
    calculate_lattice_energy as _calculate_lattice_energy,
)
from envs_tools.samplemat import (
    pymatgen_image,
)
from general_tools.quantum_espresso import (
    _quantum_espresso_image,
    _run_quantum_espresso,
)
from modal import App

simagent_name = os.getenv("SIMAGENT_NAME", "")
if simagent_name and not simagent_name.startswith("-"):
    simagent_name = f"-{simagent_name}"

# Create the app
app = App(f"simagent{simagent_name}")


@app.function(image=_quantum_espresso_image, cpu=1.0, memory=5120)
def run_quantum_espresso(pw_command, options, input) -> str:
    """
    Run Quantum Espresso calculation.

    Args:
        pw_command: Path to pw.x executable
        options: List of command line options
        input: Input file contents

    Returns:
        Output of the calculation

    Raises:
        ValueError: If the calculation fails

    Examples:
        >>> run_quantum_espresso("pw.x", ["nk=4"], "...")
        "..."
    """
    return _run_quantum_espresso(pw_command, options, input)


@app.function(image=pymatgen_image)
def calculate_lattice_energy(structure_file: str) -> float:
    """
    Calculate the lattice energy of a crystal structure.

    Args:
        structure_file: Path to structure file (CIF, POSCAR, etc.)

    Returns:
        Lattice energy in eV

    Raises:
        ValueError: If the lattice energy calculation fails

    Examples:
        >>> calculate_lattice_energy("NaCl.cif")
        -3.2
    """
    energy = _calculate_lattice_energy(structure_file)
    if energy is None:
        raise ValueError("Failed to calculate lattice energy")
    else:
        return energy
