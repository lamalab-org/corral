from typing import TypedDict


class BaseTypedDict(TypedDict):
    _type_marker: str  # Marker attribute


def create_typed_dict(cls: type[BaseTypedDict], **kwargs) -> BaseTypedDict:
    instance = cls(**kwargs)
    instance["_type_marker"] = (
        cls.__name__
    )  # Set marker to the class name automatically
    return instance


def matches_typed_dict_type(obj, typed_dict_types: tuple) -> bool:
    """Checks if `obj` matches any type in `typed_dict_types` based on `_type_marker`."""
    if isinstance(obj, dict) and "_type_marker" in obj:
        marker = obj["_type_marker"]
        return any(
            str(typed_dict_type).split(".")[-1].rstrip("'>") == marker
            for typed_dict_type in typed_dict_types
        )
    return False


# Base inpus output schema  for all tasks
class AtomicStructure(BaseTypedDict):
    atomic_numbers: list[int]
    positions: list[list[float]]
    cell: list[list[float]]
    pbc: list[bool]


class SurfaceOutput(BaseTypedDict):
    structure: AtomicStructure
    surface_atoms: list[int]
    area: float
    thickness: float
    bulk_energy: float


class DFTSettings(BaseTypedDict):
    xc: str
    encut: float
    kpts: list[int]


class AdsorbateOutput(BaseTypedDict):
    structures: list[AtomicStructure]
    binding_sites: list[dict[str, float]]
    site_types: list[str]


class RelaxationOutput(BaseTypedDict):
    final_structure: AtomicStructure
    converged: bool
    forces: list[list[float]]
    energies: list[float]
    n_steps: int


class EnergyOutput(BaseTypedDict):
    adsorption_energy: float
    binding_energy: float
    energy_components: dict[str, float]
    is_stable: bool
