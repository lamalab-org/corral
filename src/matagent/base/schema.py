from __future__ import annotations

from typing import Dict, List, TypedDict


# Base inpus output schema  for all tasks
class AtomicStructure(TypedDict):
    atomic_numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]


class SurfaceOutput(TypedDict):
    structure: AtomicStructure
    surface_atoms: List[int]
    area: float
    thickness: float
    bulk_energy: float


class DFTSettings(TypedDict):
    xc: str
    encut: float
    kpts: List[int]


class AdsorbateOutput(TypedDict):
    structures: List[AtomicStructure]
    binding_sites: List[Dict[str, float]]
    site_types: List[str]


class RelaxationOutput(TypedDict):
    final_structure: AtomicStructure
    converged: bool
    forces: List[List[float]]
    energies: List[float]
    n_steps: int


class EnergyOutput(TypedDict):
    adsorption_energy: float
    binding_energy: float
    energy_components: Dict[str, float]
    is_stable: bool
