from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from math import log10
from pathlib import Path
from typing import Any

import reaktoro as rk
from wetlab.colors import (
    PALETTE,
    PRECIPITATE_COLORS,
    closest_color_names,
    mix_colors,
    solution_color,
)

WETCHEM_PATH = Path(__file__).with_name("WetChem.yaml")
# Public chemistry data must be loaded by the trusted bootstrap. Tool workers
# cannot reopen the private task package after dropping their Unix identity.
_WETCHEM_CONTENTS = WETCHEM_PATH.read_bytes()
ZERO = 1e-20


class VolumeError(ValueError):
    pass


class NotEquilibratedError(RuntimeError):
    pass


class NegativeMassError(ArithmeticError):
    pass


PSEUDO_ELEMENTS = {
    "C(+2)",
    "C(+4)",
    "N(-3)",
    "N(+5)",
    "S(-2)",
    "S(0)",
    "S(+6)",
    "Fe(+2)",
    "Fe(+3)",
    "Hg(+2)",
    "Hg(+1)",
    "Ox",
    "dmg",
}


def _speciate(element_str: str):
    elements = set(element_str.strip().split())
    elements |= {"H", "O"}

    if "C" in elements:
        elements = elements - {"C"} | {"C(+2)", "C(+4)"}
    if "N" in elements:
        elements = elements - {"N"} | {"N(-3)", "N(+5)"}
    if "S" in elements:
        elements = elements - {"S"} | {"S(-2)", "S(0)", "S(+6)"}
    if "Fe" in elements:
        elements = elements - {"Fe"} | {"Fe(+2)", "Fe(+3)"}
    if "Hg" in elements:
        elements = elements - {"Hg"} | {"Hg(+1)", "Hg(+2)"}

    # Set iteration order varies across Python processes. Sorting is necessary
    # for a stable species ordering and therefore deterministic checkpoints.
    real = sorted(elements - PSEUDO_ELEMENTS)
    pseudo = sorted(elements.intersection(PSEUDO_ELEMENTS))

    results = rk.speciate(real)
    results.symbols += pseudo
    return results


def _reaktoro_version() -> str:
    try:
        return importlib.metadata.version("reaktoro")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(rk, "__version__", "unknown"))


def _database_sha256() -> str:
    return hashlib.sha256(_WETCHEM_CONTENTS).hexdigest()


def _json_copy(value: Any) -> Any:
    """Copy and validate one JSON-compatible chemistry snapshot."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class ChemicalSystemSpec:
    """Canonical information required to reconstruct a chemistry engine."""

    elements: str
    database_sha256: str = field(default_factory=_database_sha256)
    reaktoro_version: str = field(default_factory=_reaktoro_version)
    solver_epsilon: float = ZERO
    use_ideal_activity_models: bool = True
    database: str = WETCHEM_PATH.name

    def __post_init__(self) -> None:
        if not self.elements.strip():
            raise ValueError("Chemical system elements cannot be empty")
        if len(self.database_sha256) != 64:
            raise ValueError("WetChem database hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "database": self.database,
            "database_sha256": self.database_sha256,
            "reaktoro_version": self.reaktoro_version,
            "solver_epsilon": self.solver_epsilon,
            "use_ideal_activity_models": self.use_ideal_activity_models,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChemicalSystemSpec:
        return cls(
            elements=str(data["elements"]),
            database=str(data.get("database", WETCHEM_PATH.name)),
            database_sha256=str(data["database_sha256"]),
            reaktoro_version=str(data["reaktoro_version"]),
            solver_epsilon=float(data.get("solver_epsilon", ZERO)),
            use_ideal_activity_models=bool(data.get("use_ideal_activity_models", True)),
        )

    def assert_compatible(self) -> None:
        if self.database != WETCHEM_PATH.name:
            raise RuntimeError(
                f"Unsupported wetlab database {self.database!r}; "
                f"expected {WETCHEM_PATH.name!r}"
            )
        actual_hash = _database_sha256()
        if self.database_sha256 != actual_hash:
            raise RuntimeError(
                "WetChem.yaml does not match the database recorded in WetlabState"
            )
        actual_version = _reaktoro_version()
        if self.reaktoro_version != actual_version:
            raise RuntimeError(
                "Reaktoro version does not match WetlabState: "
                f"checkpoint={self.reaktoro_version!r}, runtime={actual_version!r}"
            )


@dataclass(frozen=True)
class WetlabState:
    """Durable, JSON-only snapshot of a wetlab chemical system and inventory."""

    chemical_system: ChemicalSystemSpec
    inventory: Mapping[str, Mapping[str, Any]]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported WetlabState schema {self.schema_version}")
        for label, item in self.inventory.items():
            if not isinstance(label, str) or not isinstance(item, Mapping):
                raise TypeError("Wetlab inventory must map labels to objects")
            if item.get("type") not in {"Precipitate", "Solution", "StockSolution"}:
                raise ValueError(f"Invalid inventory item type for {label!r}")

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(
            {
                "schema_version": self.schema_version,
                "chemical_system": self.chemical_system.to_dict(),
                "inventory": dict(self.inventory),
            }
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WetlabState:
        raw_system = data.get("chemical_system")
        raw_inventory = data.get("inventory")
        if not isinstance(raw_system, Mapping) or not isinstance(
            raw_inventory, Mapping
        ):
            raise TypeError(
                "WetlabState requires chemical_system and inventory objects"
            )
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            chemical_system=ChemicalSystemSpec.from_dict(raw_system),
            inventory=_json_copy(dict(raw_inventory)),
        )


class WetlabEngine:
    """Disposable Reaktoro runtime reconstructed from `ChemicalSystemSpec`."""

    def __init__(self, spec: ChemicalSystemSpec):
        spec.assert_compatible()
        self.spec = spec
        database = rk.Database.fromStringYAML(_WETCHEM_CONTENTS.decode("utf-8"))
        aqueous_phase = rk.AqueousPhase(_speciate(spec.elements))
        try:
            self.system = rk.ChemicalSystem(database, aqueous_phase, rk.MineralPhases())
        except RuntimeError:
            self.system = rk.ChemicalSystem(database, aqueous_phase)

        options = rk.EquilibriumOptions()
        options.epsilon = spec.solver_epsilon
        options.use_ideal_activity_models = spec.use_ideal_activity_models
        self.solver = rk.EquilibriumSolver(self.system)
        self.solver.setOptions(options)

    def new_empty_state(self) -> rk.ChemicalState:
        state = rk.ChemicalState(self.system)
        state.setSpeciesAmounts(ZERO)
        return state

    def serialize_chemical_state(self, state: rk.ChemicalState) -> dict[str, Any]:
        return {
            "temperature": float(state.temperature()),
            "pressure": float(state.pressure()),
            "species": [sp.name() for sp in state.system().species()],
            "amounts": [float(amount) for amount in state.speciesAmounts()],
        }

    def deserialize_chemical_state(self, data: Mapping[str, Any]) -> rk.ChemicalState:
        state = self.new_empty_state()
        state.setTemperature(float(data["temperature"]))
        state.setPressure(float(data["pressure"]))
        expected_species = [sp.name() for sp in self.system.species()]
        species = list(data["species"])
        if expected_species != species:
            raise RuntimeError(
                "Species list does not match the recorded wetlab chemical system"
            )
        amounts = [float(amount) for amount in data["amounts"]]
        if len(amounts) != len(expected_species):
            raise ValueError("Chemical-state species and amounts lengths differ")
        state.setSpeciesAmounts(amounts)
        return state

    def precipitate(
        self, composition: Mapping[str, float], description: str | None = None
    ) -> Precipitate:
        return Precipitate(dict(composition), description, _engine=self)

    def stock_solution(
        self, composition: Mapping[str, float], description: str | None = None
    ) -> StockSolution:
        return StockSolution(dict(composition), description, _engine=self)

    def solution(
        self,
        composition: Mapping[str, float],
        *,
        volume: float,
        description: str | None = None,
    ) -> Solution:
        return Solution(
            composition=dict(composition),
            description=description,
            volume=volume,
            _engine=self,
        )

    def snapshot(self, inventory: Mapping[str, Any]) -> WetlabState:
        return WetlabState(
            chemical_system=self.spec,
            inventory=_json_copy(
                {label: item.to_dict() for label, item in inventory.items()}
            ),
        )

    def restore(self, state: WetlabState) -> dict[str, Any]:
        if state.chemical_system != self.spec:
            raise RuntimeError("WetlabState chemical system does not match this engine")
        inventory: dict[str, Any] = {}
        for label, raw_item in state.inventory.items():
            item = dict(raw_item)
            item_type = item.get("type")
            if item_type == "Precipitate":
                inventory[label] = Precipitate.from_dict(item, engine=self)
            elif item_type == "StockSolution":
                inventory[label] = StockSolution.from_dict(item, engine=self)
            elif item_type == "Solution":
                inventory[label] = Solution.from_dict(item, engine=self)
            else:  # guarded by WetlabState, retained for defensive clarity
                raise TypeError(f"Unknown inventory item type for {label!r}")
        return inventory


@dataclass
class Precipitate:
    composition: dict[str, float]
    description: str | None = None
    _engine: WetlabEngine = field(repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        total_mass = 0
        total_mol = 0
        state = self._engine.new_empty_state()
        for solid, mol in self.composition.items():
            if mol < 0:
                raise NegativeMassError(f"Negative amount for {solid}: {mol}")
            state.set(solid + "(s)", mol, "mol")
            total_mass += state.speciesMass(solid + "(s)")
            total_mol += mol

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "components", list(self.composition.keys()))
        object.__setattr__(self, "mass", 1000 * float(total_mass))
        object.__setattr__(self, "total_mol", total_mol)

    def __getitem__(self, item):
        return self.composition.get(item, 0.0)

    def __add__(self, other: Precipitate) -> Precipitate:
        if other is None:
            return self

        if isinstance(other, Precipitate):
            if self.state.system().id() != other.state.system().id():
                raise ArithmeticError(
                    "Cannot mix Precipitates from different chemical systems!"
                )
            all_species = set(self.components + other.components)
            new_composition = {sp: self[sp] + other[sp] for sp in all_species}
            return self._engine.precipitate(new_composition)

        return NotImplemented

    __radd__ = __add__

    def __rmul__(self, other: float) -> Precipitate | None:
        if other == 0:
            return None

        if isinstance(other, float) and other > 0:
            new_composition = {sp: other * self[sp] for sp in self.components}
            return self._engine.precipitate(new_composition)

        return NotImplemented

    def clone(self) -> Precipitate:
        return self._engine.precipitate(self.composition, description=self.description)

    @property
    def sys_species(self) -> Generator:
        sys = self.state.system()
        for sp in sys.species():
            yield sp.name()

    @property
    def color(self) -> str:
        fracs = [mol / self.total_mol for mol in self.composition.values()]
        color_names = [
            PRECIPITATE_COLORS.get(name, "white") for name in self.components
        ]
        color_hexes = [PALETTE[name] for name in color_names]
        mixture = list(zip(color_hexes, fracs, strict=False))
        return mix_colors(mixture)

    @property
    def color_name(self) -> str:
        hex_color = self.color
        return closest_color_names(hex_color, mode="precipitate")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Precipitate",
            "composition": self.composition,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, engine: WetlabEngine) -> Precipitate:
        if data.get("type") != "Precipitate":
            raise TypeError("Expected a serialized Precipitate")
        return cls(
            composition=data["composition"],
            description=data["description"],
            _engine=engine,
        )


@dataclass
class StockSolution:
    composition: dict[str, float]
    description: str | None = None
    _engine: WetlabEngine | None = field(
        default=None, repr=False, compare=False, kw_only=True
    )

    def __post_init__(self):
        for sp, conc in self.composition.items():
            if conc < 0:
                raise NegativeMassError(f"Negative concentration for {sp}: {conc}")

    def __rmul__(self, vol_mL: float) -> Solution:
        if self._engine is None:
            raise RuntimeError("StockSolution is not bound to a WetlabEngine")
        return self._engine.solution(
            self.composition, description=self.description, volume=vol_mL
        )

    def __getitem__(self, item):
        return self.composition.get(item, 0.0)

    def clone(self, *, engine: WetlabEngine | None = None) -> StockSolution:
        return StockSolution(
            composition=dict(self.composition),
            description=self.description,
            _engine=engine or self._engine,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "StockSolution",
            "composition": self.composition,
            "description": self.description,
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, engine: WetlabEngine
    ) -> StockSolution:
        if data.get("type") != "StockSolution":
            raise TypeError("Expected a serialized StockSolution")
        return cls(
            composition=data["composition"],
            description=data["description"],
            _engine=engine,
        )


@dataclass(kw_only=True)
class Solution(StockSolution):
    volume: float

    def __post_init__(self):
        super().__post_init__()

        if self.volume < 0:
            raise VolumeError("Solution volume cannot be negative!")
        if self._engine is None:
            raise RuntimeError("Solution is not bound to a WetlabEngine")

        state = self._engine.new_empty_state()
        vol_L = self.volume / 1000
        state.set("H2O", vol_L, "kg")

        for sp, conc in self.composition.items():
            mol = conc * vol_L
            state.set(sp, mol, "mol")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "components", list(self.composition.keys()))
        object.__setattr__(self, "is_equilibrated", False)
        object.__setattr__(self, "has_precipitate", False)
        object.__setattr__(self, "precipitate", None)
        object.__setattr__(self, "_invisible_solids", None)
        object.__setattr__(self, "_eq_iters", 0)
        object.__setattr__(self, "_eq_error", 0.0)

    def __rmul__(self, vol_mL: float) -> Solution:
        if type(vol_mL) not in [int, float]:
            return NotImplemented

        if vol_mL > self.volume + 1e-8:
            raise VolumeError(
                f"Requested to draw {vol_mL:.2f} mL but only {self.volume:.2f} mL available."
            )

        if self.has_precipitate:
            raise VolumeError(
                "Cannot draw from solutions that have precipitation. Filter it first!"
            )

        frac = round(vol_mL, 2) / (self.volume + 1e-8)
        self.volume -= vol_mL
        new_solution = self._engine.solution(self.composition, volume=vol_mL)
        object.__setattr__(new_solution, "is_equilibrated", self.is_equilibrated)

        if self._invisible_solids is not None:
            invisible_solids = frac * self._invisible_solids
            for sp in invisible_solids.components:
                new_solution.state.set(sp + "(s)", invisible_solids[sp], "mol")
            object.__setattr__(new_solution, "_invisible_solids", invisible_solids)

        return new_solution

    def __add__(self, other: Solution) -> Solution:
        if other is None:
            return self

        if isinstance(other, Solution):
            if self.state.system().id() != other.state.system().id():
                raise ArithmeticError(
                    "Cannot mix Solutions from different chemical systems!"
                )

            new_volume = self.volume + other.volume
            all_aq_species = set(self.components + other.components) - {"H2O"}

            new_composition = {}
            for sp in all_aq_species:
                total_mol = self[sp] * self.volume + other[sp] * other.volume
                new_composition[sp] = total_mol / new_volume

            new_solution = self._engine.solution(new_composition, volume=new_volume)

            invisible_solids = None
            if (self._invisible_solids is not None) or (
                other._invisible_solids is not None
            ):
                invisible_solids = self._invisible_solids + other._invisible_solids
                object.__setattr__(new_solution, "_invisible_solids", invisible_solids)

            precipitate = None
            if self.has_precipitate or other.has_precipitate:
                precipitate = self.precipitate + other.precipitate
                object.__setattr__(new_solution, "precipitate", precipitate)
                object.__setattr__(new_solution, "has_precipitate", True)

            if (invisible_solids is not None) or (precipitate is not None):
                all_solids = invisible_solids + precipitate
                for sp in all_solids.components:
                    new_solution.state.set(sp + "(s)", all_solids[sp], "mol")

            self.volume = 0.0
            other.volume = 0.0
            return new_solution

        if isinstance(other, Precipitate):
            new_precipitate = self.precipitate + other
            all_solids = self._invisible_solids + new_precipitate

            new_solution = self._engine.solution(self.composition, volume=self.volume)
            for sp in all_solids.components:
                new_solution.state.set(sp + "(s)", all_solids[sp], "mol")

            object.__setattr__(new_solution, "has_precipitate", True)
            object.__setattr__(new_solution, "precipitate", new_precipitate)
            object.__setattr__(
                new_solution, "_invisible_solids", self._invisible_solids
            )

            return new_solution

        return NotImplemented

    __radd__ = __add__

    def equilibrate(
        self, precipitation_threshold=8e-5, retry=3, error_threshold=4e-8
    ) -> None:
        # trying to solve the equilibrium equations
        self._eq_iters = 0
        converged = False
        for _ in range(retry):
            eq_result = self._engine.solver.solve(self.state)
            self._eq_iters += 1
            error = eq_result.optima.error
            if eq_result.succeeded() or (error < error_threshold):
                self._eq_error = error
                converged = True
                break
        if not converged:
            self._eq_error = error
            err_percent = 100 * (error - error_threshold) / error_threshold
            raise NotEquilibratedError(
                f"Equilibrium calculations failed to converge after {retry} attempts! ({err_percent:.1f} %)"
            )

        visible_solids = {}
        invisible_solids = {}
        aqueous = {}
        for sp in self.sys_species:
            mol = self.state.speciesAmount(sp)
            if sp.endswith("(s)") and (mol >= 10 * ZERO):
                if 1000 * mol / self.volume >= precipitation_threshold:
                    visible_solids[sp.removesuffix("(s)")] = float(mol)
                else:
                    invisible_solids[sp.removesuffix("(s)")] = float(mol)

            elif (sp != "H2O") and (mol >= 10 * ZERO):
                aqueous[sp] = 1000 * float(mol) / self.volume

        precipitate = (
            self._engine.precipitate(visible_solids) if visible_solids else None
        )
        invisible_solids = (
            self._engine.precipitate(invisible_solids) if invisible_solids else None
        )
        has_precipitate = precipitate is not None

        object.__setattr__(self, "composition", aqueous)
        object.__setattr__(self, "components", list(aqueous.keys()))
        object.__setattr__(self, "is_equilibrated", True)
        object.__setattr__(self, "precipitate", precipitate)
        object.__setattr__(self, "has_precipitate", has_precipitate)
        object.__setattr__(self, "_invisible_solids", invisible_solids)

        return

    def filter(self) -> tuple[Solution, Precipitate]:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "Only equilibrated solutions can be filtered. "
                "Call .equilibrate() first."
            )

        precipitate = None
        if (self.precipitate is not None) or (self._invisible_solids is not None):
            precipitate = self.precipitate + self._invisible_solids

        if self.volume > 0:
            filtrate = self._engine.solution(self.composition, volume=self.volume)
            object.__setattr__(filtrate, "is_equilibrated", True)
            return filtrate, precipitate
        else:
            raise VolumeError("Cannot filter solutions with a volume of zero!")

    def add_solid(self, solid: Precipitate) -> None:
        if not isinstance(solid, Precipitate):
            raise TypeError("The given solid must be a Precipitate object!")

        if self.state.system().id() != solid.state.system().id():
            raise ArithmeticError(
                "Cannot add a solid from a different chemical system!"
            )

        new_precipitate = solid + self.precipitate

        for sp in new_precipitate.components:
            self.state.set(sp + "(s)", solid[sp], "mol")

        object.__setattr__(self, "has_precipitate", True)
        object.__setattr__(self, "precipitate", new_precipitate)
        object.__setattr__(self, "is_equilibrated", False)

        return

    def clone(self) -> Solution:
        new_solution = self._engine.solution(
            self.composition,
            volume=self.volume,
            description=self.description,
        )
        new_state = self.state.clone()
        new_precipitate = (
            self.precipitate.clone() if self.precipitate is not None else None
        )
        new_invisible_solids = (
            self._invisible_solids.clone()
            if self._invisible_solids is not None
            else None
        )
        object.__setattr__(new_solution, "state", new_state)
        object.__setattr__(new_solution, "is_equilibrated", self.is_equilibrated)
        object.__setattr__(new_solution, "has_precipitate", self.has_precipitate)
        object.__setattr__(new_solution, "precipitate", new_precipitate)
        object.__setattr__(new_solution, "_invisible_solids", new_invisible_solids)
        return new_solution

    @property
    def sys_species(self) -> Generator:
        sys = self.state.system()
        for sp in sys.species():
            yield sp.name()

    @property
    def pH(self) -> float:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "pH is unavailable until the solution is equilibrated. "
                "Call .equilibrate() first."
            )
        return -log10(self["H+"])

    @property
    def color(self) -> str:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "color is unavailable until the solution is equilibrated. "
                "Call .equilibrate() first."
            )

        return solution_color(self.composition)

    @property
    def color_name(self) -> str:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "color_name is unavailable until the solution is equilibrated. "
                "Call .equilibrate() first."
            )
        hex_color = self.color
        return closest_color_names(hex_color, mode="solution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Solution",
            "composition": self.composition,
            "description": self.description,
            "volume": float(self.volume),
            "state": self._engine.serialize_chemical_state(self.state),
            "has_precipitate": self.has_precipitate,
            "is_equilibrated": self.is_equilibrated,
            "precipitate": self.precipitate.to_dict() if self.has_precipitate else None,
            "_invisible_solids": self._invisible_solids.to_dict()
            if self._invisible_solids is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, engine: WetlabEngine) -> Solution:
        if data.get("type") != "Solution":
            raise TypeError("Expected a serialized Solution")
        sol = cls(
            composition=data["composition"],
            description=data["description"],
            volume=data["volume"],
            _engine=engine,
        )

        state = engine.deserialize_chemical_state(data["state"])
        precipitate = (
            Precipitate.from_dict(data["precipitate"], engine=engine)
            if data["precipitate"] is not None
            else None
        )
        _invisible_solids = (
            Precipitate.from_dict(data["_invisible_solids"], engine=engine)
            if data["_invisible_solids"] is not None
            else None
        )

        object.__setattr__(sol, "state", state)
        object.__setattr__(sol, "is_equilibrated", data["is_equilibrated"])
        object.__setattr__(sol, "has_precipitate", data["has_precipitate"])
        object.__setattr__(sol, "precipitate", precipitate)
        object.__setattr__(sol, "_invisible_solids", _invisible_solids)
        return sol
