from ase import Atoms
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
from ase.io import write

# Define FCC aluminum lattice
lattice_constant = 4.05  # Angstrom
size = (5, 5, 5)  # Corresponding to the simulation box
aluminum_fcc = bulk("Al", "fcc", a=lattice_constant, cubic=True) * size

# Set atomic mass
aluminum_fcc.set_masses([26.98] * len(aluminum_fcc))

# Write the structure to a LAMMPS data file
write_lammps_data("structure.xyz", aluminum_fcc, atom_style="atomic")