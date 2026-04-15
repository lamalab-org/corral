from dataclasses import dataclass
from math import log10
from typing import Optional, Dict, Generator
from wetlab.colors import mix_colors, solution_color, closest_color_names, PRECIPITATE_COLORS, PALETTE
import reaktoro as rk
import numpy as np

class VolumeError(ValueError):
    pass


class NotEquilibratedError(RuntimeError):
    pass


class NegativeMassError(ArithmeticError):
    pass


import pathlib as _pathlib
DB = rk.Database.fromFile(str(_pathlib.Path(__file__).with_name('WetChem.yaml')))
ZERO = 1e-20
SOLVER_OPTIONS = rk.EquilibriumOptions()
SOLVER_OPTIONS.epsilon = ZERO
SOLVER_OPTIONS.use_ideal_activity_models = True


PSEUDO_ELEMENTS = {
    'C(+2)', 'C(+4)', 'N(-3)', 'N(+5)', 'S(-2)', 'S(0)', 'S(+6)',
    'Fe(+2)', 'Fe(+3)', 'Hg(+2)', 'Hg(+1)', 'Ox', 'dmg',
}

def _speciate(element_str):
    elements = set(element_str.strip().split())
    elements |= {'H', 'O'}
    
    if 'C' in elements:
        elements = elements - {'C'} | {'C(+2)', 'C(+4)'}
    if 'N' in elements:
        elements = elements - {'N'} | {'N(-3)', 'N(+5)'}
    if 'S' in elements:
        elements = elements - {'S'} | {'S(-2)', 'S(0)', 'S(+6)'}
    if 'Fe' in elements:
        elements = elements - {'Fe'} | {'Fe(+2)', 'Fe(+3)'}
    if 'Hg' in elements:
        elements = elements - {'Hg'} | {'Hg(+1)', 'Hg(+2)'}

    real = list(elements - PSEUDO_ELEMENTS)
    pseudo = list(elements.intersection(PSEUDO_ELEMENTS))

    results = rk.speciate(real)
    results.symbols += pseudo
    return results


def set_chemical_system(system: str | rk.ChemicalSystem) -> rk.ChemicalSystem:
    global DEFAULT_SYS, SOLVER, SOLVER_OPTIONS

    if isinstance(system, rk.ChemicalSystem):
        DEFAULT_SYS = system
    else:
        aq_phase = rk.AqueousPhase(_speciate(system))
        try:
            DEFAULT_SYS = rk.ChemicalSystem(DB, aq_phase, rk.MineralPhases())
        except RuntimeError:
            DEFAULT_SYS = rk.ChemicalSystem(DB, aq_phase)

    SOLVER = rk.EquilibriumSolver(DEFAULT_SYS)
    SOLVER.setOptions(SOLVER_OPTIONS)
    return DEFAULT_SYS


def get_chemical_system() -> rk.ChemicalSystem:
    return DEFAULT_SYS


def _new_empty_state(sys: Optional[rk.ChemicalSystem]=None) -> rk.ChemicalState:
    sys = sys or DEFAULT_SYS
    state = rk.ChemicalState(sys)
    state.setSpeciesAmounts(ZERO)
    return state

def  _serialize_state(state: rk.ChemicalState) -> dict:
    return {
        "T": float(state.temperature()),
        "P": float(state.pressure()),
        "species": [sp.name() for sp in state.system().species()],
        "amounts": np.asarray(state.speciesAmounts(), dtype=float)
    }

def _deserialize_state(data: dict) -> rk.ChemicalState:
    state = _new_empty_state()
    state.setTemperature(data["T"])
    state.setPressure(data["P"])
    sys_species = [sp.name() for sp in state.system().species()]
    assert sys_species == data["species"], "Mismatch in species list when deserializing state!"
    state.setSpeciesAmounts(data["amounts"])
    return state


@dataclass
class Precipitate:
    composition: Dict[str, float]
    description: Optional[str] = None

    def __post_init__(self):

        total_mass = 0
        total_mol = 0
        state = _new_empty_state()
        for solid, mol in self.composition.items():
            if mol < 0:
                raise NegativeMassError(f"Negative amount for {solid}: {mol}")
            state.set(solid+"(s)", mol, "mol")
            total_mass += state.speciesMass(solid+"(s)")
            total_mol += mol
        
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "components", list(self.composition.keys()))
        object.__setattr__(self, "mass", 1000*float(total_mass))
        object.__setattr__(self, "total_mol", total_mol)


    def __getitem__(self, item):
        return self.composition.get(item, 0.0)
    

    def __add__(self, other: "Precipitate") -> "Precipitate":
        if other is None:
            return self
        
        if isinstance(other, Precipitate):
            if self.state.system().id() != other.state.system().id():
                raise ArithmeticError("Cannot mix Precipitates from different chemical systems!")
            all_species = set(self.components + other.components)
            new_composition = {sp: self[sp]+other[sp] for sp in all_species}
            return Precipitate(new_composition)
        
        return NotImplemented
    
    __radd__ = __add__ 

    def __rmul__(self, other: float) -> "Precipitate":
        if other == 0:
            return None
        
        if type(other) == float and other > 0 :
            new_composition = {sp: other*self[sp] for sp in self.components}
            return Precipitate(new_composition)
        
        return NotImplemented
    
    def clone(self) -> "Precipitate":
        new_precipitate = Precipitate(self.composition, description=self.description)
        return new_precipitate

    @property
    def sys_species(self) -> Generator:
        sys = self.state.system()
        for sp in sys.species():
            yield sp.name()


    @property
    def color(self) -> str:
        fracs = [mol/self.total_mol for mol in self.composition.values()]
        color_names = [PRECIPITATE_COLORS.get(name, 'white') for name in self.components]
        color_hexes = [PALETTE[name] for name in color_names]
        mixture = list(zip(color_hexes, fracs))
        return mix_colors(mixture)
    

    @property
    def color_name(self) -> str:
        hex = self.color
        return closest_color_names(hex, mode='precipitate')
    
    def to_dict(self) -> Dict:
        return {
            "type": "Precipitate",
            "composition": self.composition,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Precipitate":
        assert data["type"] == "Precipitate", "Type mismatch when reading from dict!"
        return cls(
            composition=data["composition"],
            description=data["description"]
        )

@dataclass
class StockSolution:
    composition: Dict[str, float]
    description: Optional[str] = None

    def __post_init__(self):
        for sp, conc in self.composition.items():
            if conc < 0:
                raise NegativeMassError(f"Negative concentration for {sp}: {conc}")
    

    def __rmul__(self, vol_mL: float) -> "Solution":
        return Solution(self.composition, description=self.description, volume=vol_mL)
    

    def __getitem__(self, item):
        return self.composition.get(item, 0.0)
    
    
    def clone(self) -> "StockSolution":
        return StockSolution(composition=self.composition, description=self.description)
    
    def to_dict(self) -> Dict:
        return {
            "type": "StockSolution",
            "composition": self.composition,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StockSolution":
        assert data["type"] == "StockSolution", "Type mismatch when reading from dict!"
        return cls(
            composition=data["composition"],
            description=data["description"]
        )

@dataclass(kw_only=True)
class Solution(StockSolution):
    volume: float

    def __post_init__(self):
        super().__post_init__()

        if self.volume <= 0:
            raise VolumeError("Solution volume must be positive!")
        
        state = _new_empty_state()
        vol_L = self.volume/1000
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
    

    def __rmul__(self, vol_mL: float) -> "Solution":

        if type(vol_mL) not in [int, float]:
            return NotImplemented
        
        if vol_mL > self.volume + 1e-8:
            raise VolumeError(f"Requested to draw {vol_mL:.2f} mL but only {self.volume:.2f} mL available.")
        
        if self.has_precipitate:
            raise VolumeError("Cannot draw from solutions that have precipitation. Filter it first!")
        
        frac = round(vol_mL,2) / (self.volume + 1e-8)
        self.volume -= vol_mL
        new_solution = Solution(self.composition, volume=vol_mL)
        object.__setattr__(new_solution, "is_equilibrated", self.is_equilibrated)
        
        if self._invisible_solids is not None:
            invisible_solids = frac * self._invisible_solids
            for sp in invisible_solids.components:
                new_solution.state.set(sp+"(s)", invisible_solids[sp], 'mol')
            object.__setattr__(new_solution, "_invisible_solids", invisible_solids)

        return new_solution
    

    def __add__(self, other: "Solution") -> "Solution":
        if other is None:
            return self
        
        if isinstance(other, Solution):
            if self.state.system().id() != other.state.system().id():
                raise ArithmeticError("Cannot mix Solutions from different chemical systems!")
        
            new_volume = self.volume + other.volume
            all_aq_species = set(self.components + other.components) - {'H2O'}

            new_composition = {}
            for sp in all_aq_species:
                total_mol = self[sp] * self.volume + other[sp] * other.volume
                new_composition[sp] = total_mol/new_volume

            new_solution = Solution(new_composition, volume=new_volume)

            invisible_solids = None
            if (self._invisible_solids is not None) or (other._invisible_solids is not None):
                invisible_solids = self._invisible_solids + other._invisible_solids
                object.__setattr__(new_solution, "_invisible_solids", invisible_solids)

            precipitate = None
            if self.has_precipitate or other.has_precipitate:
                precipitate =  self.precipitate + other.precipitate
                object.__setattr__(new_solution, "precipitate", precipitate)
                object.__setattr__(new_solution, "has_precipitate", True)
            
            if (invisible_solids is not None) or (precipitate is not None):
                all_solids = invisible_solids + precipitate
                for sp in all_solids.components:
                    new_solution.state.set(sp+"(s)", all_solids[sp], 'mol')

            self.volume = 0.0
            other.volume = 0.0
            return new_solution
        
        if isinstance(other, Precipitate):
            new_precipitate = self.precipitate + other
            all_solids = self._invisible_solids + new_precipitate
            
            new_solution = Solution(self.composition, volume=self.volume)
            for sp in all_solids.components:
                new_solution.state.set(sp+"(s)", all_solids[sp], 'mol')
        
            object.__setattr__(new_solution, "has_precipitate", True)
            object.__setattr__(new_solution, "precipitate", new_precipitate)
            object.__setattr__(new_solution, "_invisible_solids", self._invisible_solids)

            return new_solution
        
        return NotImplemented
    

    __radd__ = __add__ 


    def equilibrate(self, precipitation_threshold=8e-5, retry=3, error_threshold=4e-8) -> None:
    
        # trying to solve the equilibrium equations
        self._eq_iters = 0
        converged = False
        for _ in range(retry):
            eq_result = SOLVER.solve(self.state)
            self._eq_iters += 1
            error = eq_result.optima.error
            if eq_result.succeeded() or (error < error_threshold):
                self._eq_error = error
                converged = True
                break
        if not converged:
            self._eq_error = error
            err_percent = 100*(error-error_threshold)/error_threshold
            raise NotEquilibratedError(f"Equilibrium calculations failed to converge after {retry} attempts! ({err_percent:.1f} %)")

        visible_solids = {}
        invisible_solids = {}
        aqueous = {}
        for sp in self.sys_species:
            mol = self.state.speciesAmount(sp)
            if sp.endswith('(s)') and (mol >= 10*ZERO):
                if (1000*mol/self.volume >= precipitation_threshold):
                    visible_solids[sp.removesuffix('(s)')] = float(mol)
                else:
                    invisible_solids[sp.removesuffix('(s)')] = float(mol)

            elif (sp != 'H2O') and (mol >= 10*ZERO):
                aqueous[sp] = 1000*float(mol)/self.volume

        precipitate = Precipitate(visible_solids) if visible_solids != {} else None
        invisible_solids = Precipitate(invisible_solids) if invisible_solids != {} else None
        has_precipitate = precipitate is not None
        
        object.__setattr__(self, "composition", aqueous)
        object.__setattr__(self, "components", list(aqueous.keys()))
        object.__setattr__(self, "is_equilibrated", True)
        object.__setattr__(self, "precipitate", precipitate)
        object.__setattr__(self, "has_precipitate", has_precipitate)
        object.__setattr__(self, "_invisible_solids", invisible_solids)

        return None


    def filter(self) -> tuple["Solution", Precipitate]:
        
        if not self.is_equilibrated:
            raise NotEquilibratedError("Only equilibrated solutions can be filtered. "
                                       "Call .equilibrate() first.")
        
        precipitate = None
        if (self.precipitate is not None) or (self._invisible_solids is not None):
            precipitate = self.precipitate + self._invisible_solids
        
        if self.volume > 0:
            filtrate = Solution(self.composition, volume=self.volume)
            object.__setattr__(filtrate, "is_equilibrated", True)
            return filtrate, precipitate
        else:
            raise VolumeError("Cannot filter solutions with a volume of zero!")



    def add_solid(self, solid: Precipitate) -> None:
        if not isinstance(solid, Precipitate):
            raise TypeError("The given solid must be a Precipitate object!")
        
        if self.state.system().id() != solid.state.system().id():
            raise ArithmeticError("Cannot add a solid from a different chemical system!")
        
        new_precipitate = solid + self.precipitate

        for sp in new_precipitate.components:
            self.state.set(sp+"(s)", solid[sp], 'mol')
        
        object.__setattr__(self, "has_precipitate", True)
        object.__setattr__(self, "precipitate", new_precipitate)
        object.__setattr__(self, "is_equilibrated", False)

        return None
    
    def clone(self) -> "Solution":
        new_solution = Solution(self.composition, volume=self.volume, description=self.description)
        new_state = self.state.clone()
        new_precipitate = self.precipitate.clone() if self.precipitate is not None else None
        new_invisible_solids = self._invisible_solids.clone() if self._invisible_solids is not None else None
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
                "Call .equilibrate() first.")
        else:
            return -log10(self['H+'])


    @property
    def color(self) -> str:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "color is unavailable until the solution is equilibrated. "
                "Call .equilibrate() first.")
        
        else:
            return solution_color(self.composition)
    

    @property
    def color_name(self) -> str:
        if not self.is_equilibrated:
            raise NotEquilibratedError(
                "color_name is unavailable until the solution is equilibrated. "
                "Call .equilibrate() first.")
        else:
            hex = self.color
            return closest_color_names(hex, mode='solution')

    def to_dict(self) -> Dict:
        return {
            "type": "Solution",
            "composition": self.composition,
            "description": self.description,
            "volume": max(0.01, self.volume), # workaround for zero volumes
            "state": _serialize_state(self.state),
            "has_precipitate": self.has_precipitate,
            "is_equilibrated": self.is_equilibrated,
            "precipitate": self.precipitate.to_dict() if self.has_precipitate else None,
            "_invisible_solids": self._invisible_solids.to_dict() if self._invisible_solids is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Solution":
        assert data["type"] == "Solution", "Type mismatch when reading from dict!"
        sol = cls(
            composition=data["composition"],
            description=data["description"],
            volume=data["volume"]
        )

        state = _deserialize_state(data["state"])
        precipitate = Precipitate.from_dict(data["precipitate"]) if data["precipitate"] is not None else None
        _invisible_solids = Precipitate.from_dict(data["_invisible_solids"]) if data["_invisible_solids"] is not None else None

        object.__setattr__(sol, "state", state)
        object.__setattr__(sol, "is_equilibrated", data["is_equilibrated"])
        object.__setattr__(sol, "has_precipitate", data["has_precipitate"])
        object.__setattr__(sol, "precipitate", precipitate)
        object.__setattr__(sol, "_invisible_solids", _invisible_solids)
        return sol
