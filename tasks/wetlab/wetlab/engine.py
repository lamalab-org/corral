from dataclasses import dataclass
from typing import Dict
import reaktoro as rk

class VolumeError(ValueError):
    pass

db = rk.Database.fromFile('WetChem.yaml')

aq_phase = rk.AqueousPhase('H2O H+ OH- NO3- Cl- I- K+ '
                           'NH3 NH4+ '
                           'H2S HS- S-2 '
                           'Ag+ Ag(NH3)+ Ag(NH3)2+ AgCl2- '
                           'Cu+2 Cu(NH3)+2 Cu(NH3)2+2 Cu(NH3)3+2 Cu(NH3)4+2 '
                           'Mn+2 Pb+2 ')
solids = [
    'AgCl(s)', 'AgI(s)',
    'PbCl2(s)', 'PbI2(s)',
    'Ag2S(s)', 'CuS(s)', 'MnS(s)', 'PbS(s)',
    'Cu(OH)2(s)', 'Mn(OH)2(s)', 'Pb(OH)2(s)'
]
solid_phases = [rk.SolidPhase(solid) for solid in solids]

DEFAULT_SYS = rk.ChemicalSystem(db, aq_phase, *solid_phases)
ZERO = 1e-25

solver = rk.EquilibriumSolver(DEFAULT_SYS)
options = rk.EquilibriumOptions()
options.epsilon = ZERO
solver.setOptions(options)

def _new_empty_state(sys=DEFAULT_SYS):
    state = rk.ChemicalState(sys)
    state.setSpeciesAmounts(ZERO)
    return state

def _species_names(sys=DEFAULT_SYS):
    for sp in sys.species():
        yield sp.name()

def _copy_state(state):
    new = _new_empty_state(state.system())
    for name in _species_names(state.system()):
        mol = state.speciesAmount(name)
        if mol > ZERO:
            new.setSpeciesAmount(name, mol, "mol")
    return new

@dataclass
class StockSolution:
    composition: Dict[str, float]
    description: str=None
    
    def __post_init__(self):
        for sp, C in self.composition.items():
            if C < 0:
                raise ValueError(f"Negative concentration for {sp}: {C}")
    
    def __rmul__(self, V_mL: float) -> "Solution":
        return Solution(self.composition, volume=V_mL)

@dataclass(kw_only=True)
class Solution(StockSolution):
    volume: float

    def __post_init__(self):
        if self.volume <= 0:
            raise VolumeError("Solution volume must be positive!")
        
        state = _new_empty_state()
        V_L = self.volume/1000
        state.setSpeciesMass("H2O", V_L, "kg")

        for sp, C in self.composition.items():
            mol = C * V_L
            state.setSpeciesAmount(sp, mol, "mol")

        object.__setattr__(self, "state", state)

    def __rmul__(self, V_mL: float) -> "Solution":

        if V_mL > self.volume + 1e-8:
            raise VolumeError(f"Requested to draw {V_mL:.2f} mL but only {self.volume:.2f} mL available.")
        
        self.volume -= V_mL
        return Solution(self.composition, volume=V_mL)
    
    def __add__(self, other: "Solution") -> "Solution":
        if not isinstance(other, Solution):
            raise NotImplemented
        if self.state.system().id() != other.state.system().id():
            raise ValueError("Cannot mix solutions from different chemical systems!")
        
        new_composition = {}
        new_volume = self.volume + other.volume

        for sp in _species_names():
            self_mol = self.state.speciesAmount(sp) if self.state.speciesAmount(sp) > ZERO else 0.0
            other_mol = other.state.speciesAmount(sp) if other.state.speciesAmount(sp) > ZERO else 0.0
            if (sp not in ['H2O']+solids) and (total_mol := self_mol + other_mol) > 0: 
                new_composition[sp] = 1000*float(total_mol)/new_volume
        
        return Solution(new_composition, volume=new_volume)
    
    def equilibrate_and_filter(self) -> tuple["Solution", Dict[str, float]]:

        state = _copy_state(self.state)
        solver.solve(state)

        precipitates = {}
        for solid in solids:
            mol = state.speciesAmount(solid)
            if mol > ZERO:
                precipitates[solid] = 1000*float(mol)
        
        new_composition = {}
        for sp in _species_names():
            mol = state.speciesAmount(sp)
            if (sp not in ['H2O']+solids) and mol > ZERO:
                new_composition[sp] = 1000*float(mol)/self.volume
        
        return Solution(new_composition, volume=self.volume), precipitates







            
            
