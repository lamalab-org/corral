from __future__ import annotations

from typing import Dict, List, TypedDict

class BaseTypedDict(TypedDict):
    _type_marker: str  # Marker attribute

def create_typed_dict(cls: type[BaseTypedDict], **kwargs) -> BaseTypedDict:
    instance = cls(**kwargs)
    instance["_type_marker"] = cls.__name__  # Set marker to the class name automatically
    return instance

def matches_typed_dict_type(obj, typed_dict_types: tuple) -> bool:
    """Checks if `obj` matches any type in `typed_dict_types` based on `_type_marker`."""
    if isinstance(obj, dict) and "_type_marker" in obj:
        marker = obj["_type_marker"]
        return any(str(typed_dict_type).split('.')[-1].rstrip("'>") == marker for typed_dict_type in typed_dict_types)
    return False


# Base inpus output schema  for all tasks
class AtomicStructure(BaseTypedDict):
    atomic_numbers: List[int]
    positions: List[List[float]]
    cell: List[List[float]]
    pbc: List[bool]


class SurfaceOutput(BaseTypedDict):
    structure: AtomicStructure
    surface_atoms: List[int]
    area: float
    thickness: float
    bulk_energy: float


class DFTSettings(BaseTypedDict):
    xc: str
    encut: float
    kpts: List[int]


class AdsorbateOutput(BaseTypedDict):
    structures: List[AtomicStructure]
    binding_sites: List[Dict[str, float]]
    site_types: List[str]


class RelaxationOutput(BaseTypedDict):
    final_structure: AtomicStructure
    converged: bool
    forces: List[List[float]]
    energies: List[float]
    n_steps: int


class EnergyOutput(BaseTypedDict):
    adsorption_energy: float
    binding_energy: float
    energy_components: Dict[str, float]
    is_stable: bool
