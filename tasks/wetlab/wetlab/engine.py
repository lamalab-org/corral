from dataclasses import dataclass
from math import log10
from typing import Optional, Dict, Generator
from colors import mix_colors, solution_color, closest_color_names, PRECIPITATE_COLORS, PALLETT
import reaktoro as rk

class VolumeError(ValueError):
    pass


class NotEquilibratedError(RuntimeError):
    pass

class NegativeMassError(ArithmeticError):
    pass

DB = rk.Database.fromFile('WetChem.yaml')
ZERO = 1e-20
SOLVER_OPTIONS = rk.EquilibriumOptions()
SOLVER_OPTIONS.epsilon = ZERO
SOLVER_OPTIONS.use_ideal_activity_models = True


def _speciate(element_str):
    elements = set(element_str.strip().split())
    elements |= set({'H', 'O'})
    pseudo_elements = set([
        'C(+2)', 'C(+4)', 'N(-3)', 'N(+5)', 'S(-2)', 'S(0)', 'S(+6)',
        'Fe(+2)', 'Fe(+3)', 'Hg(+2)', 'Hg(+1)',
        'Ox', 'dmg',
    ])
    real = [element for element in elements if (element not in pseudo_elements) and (element not in ['C', 'N', 'S', 'Fe', 'Hg'])]
    pseudo = [element for element in elements if element in pseudo_elements]
    if 'C' in elements:
        pseudo.extend(['C(+2)', 'C(+4)'])
    if 'N' in elements:
        pseudo.extend(['N(-3)', 'N(+5)'])
    if 'S' in elements:
        pseudo.extend(['S(-2)', 'S(0)', 'S(+6)'])
    if 'Fe' in elements:
        pseudo.extend(['Fe(+2)', 'Fe(+3)'])
    if 'Hg' in elements:
        pseudo.extend(['Hg(+1)', 'Hg(+2)'])
    pseudo = list(set(pseudo))
    results = rk.speciate(real)
    results.symbols += pseudo
    return results


def set_chemical_system(elements_str: str) -> rk.ChemicalSystem:
    global DEFAULT_SYS, SOLVER, SOLVER_OPTIONS
    aq_phase = rk.AqueousPhase(_speciate(elements_str))
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
                raise ValueError("Cannot mix Precipitates from different chemical systems!")
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
    
    @property
    def sys_species(self) -> Generator:
        sys = self.state.system()
        for sp in sys.species():
            yield sp.name()


    @property
    def color(self) -> str:
        fracs = [mol/self.total_mol for mol in self.composition.values()]
        color_names = [PRECIPITATE_COLORS.get(name, 'white') for name in self.components]
        color_hexes = [PALLETT[name] for name in color_names]
        mixture = list(zip(color_hexes, fracs))
        return mix_colors(mixture)
    

    @property
    def color_name(self) -> str:
        hex = self.color
        return closest_color_names(hex, mode='precipitate')



@dataclass
class StockSolution:
    composition: Dict[str, float]
    description: Optional[str] = None

    def __post_init__(self):
        for sp, conc in self.composition.items():
            if conc < 0:
                raise NegativeMassError(f"Negative concentration for {sp}: {conc}")
    

    def __rmul__(self, vol_mL: float) -> "Solution":
        return Solution(self.composition, volume=vol_mL)
    

    def __getitem__(self, item):
        return self.composition.get(item, 0.0)



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
    

    def __rmul__(self, vol_mL: float) -> "Solution":

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
                raise ValueError("Cannot mix Solutions from different chemical systems!")
        
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


    def equilibrate(self, precipitation_threshold=8e-5) -> None:

        SOLVER.solve(self.state)

        visible_solids = {}
        invisible_solids = {}
        aqueous = {}
        for sp in self.sys_species:
            mol = self.state.speciesAmount(sp)
            if sp.endswith('(s)'):
                if (1000*mol/self.volume >= precipitation_threshold):
                    visible_solids[sp.removesuffix('(s)')] = float(mol)
                else:
                    invisible_solids[sp.removesuffix('(s)')] = float(mol)

            elif (sp != 'H2O') and (mol > 1.1*ZERO):
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
        
        precipitate = self.precipitate + self._invisible_solids
        filtrate = Solution(self.composition, volume=self.volume)
        object.__setattr__(filtrate, "is_equilibrated", True)
        
        return filtrate, precipitate


    def add_solid(self, solid: Precipitate) -> None:
        if not isinstance(solid, Precipitate):
            raise TypeError("The given solid must be a Precipitate object!")
        
        if self.state.system().id() != solid.state.system().id():
            raise ValueError("Cannot add a solid from a different chemical system!")
        
        new_precipitate = solid + self.precipitate

        for sp in new_precipitate.components:
            self.state.set(sp+"(s)", solid[sp], 'mol')
        
        object.__setattr__(self, "has_precipitate", True)
        object.__setattr__(self, "precipitate", new_precipitate)
        object.__setattr__(self, "is_equilibrated", False)

        return None
    
    
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

            
            
