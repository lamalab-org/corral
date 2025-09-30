"""Test stoichiometry parsing functionality."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest


@dataclass
class Reaction:
    equation: str
    type: str
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, float]
    k_range: Optional[Tuple[float, float]] = None
    quantum_yield: Optional[Tuple[float, float]] = None
    
    @classmethod
    def from_dict(cls, rxn_dict: dict):
        equation = rxn_dict["equation"]
        
        if "->" not in equation:
            raise ValueError(f"Invalid equation format: {equation}")
        
        left, right = equation.split("->")
        reactants = [s.strip() for s in left.split("+")]
        products = [s.strip() for s in right.split("+")]
        
        ignored = {"hv", "H2O", "OH", "products", "H", "2 H"}
        stoich = {}
        
        def parse_species_with_coeff(species_str):
            parts = species_str.strip().split(' ', 1)
            if len(parts) == 2 and parts[0].isdigit():
                return int(parts[0]), parts[1]
            return 1, species_str
        
        for r in reactants:
            if r not in ignored:
                coeff, species = parse_species_with_coeff(r)
                stoich[species] = stoich.get(species, 0) - coeff
        
        for p in products:
            if p not in ignored:
                coeff, species = parse_species_with_coeff(p)
                stoich[species] = stoich.get(species, 0) + coeff
        
        return cls(
            equation=equation,
            type=rxn_dict["type"],
            reactants=reactants,
            products=products,
            stoichiometry=stoich,
            k_range=rxn_dict.get("k_range"),
            quantum_yield=rxn_dict.get("quantum_yield"),
        )


def test_simple_reaction_stoichiometry():
    """Test simple reaction stoichiometry parsing."""
    rxn = Reaction.from_dict({
        "equation": "A + B -> C",
        "type": "dark",
        "k_range": [1e3, 1e5]
    })
    
    expected = {"A": -1, "B": -1, "C": 1}
    assert rxn.stoichiometry == expected


def test_complex_reaction_with_coefficients():
    """Test complex reaction with coefficients."""
    rxn = Reaction.from_dict({
        "equation": "2 RuIII + H2O2 -> 2 RuII + O2 + 2 H",
        "type": "dark", 
        "k_range": [1e3, 1e5]
    })
    
    expected = {"RuIII": -2, "H2O2": -1, "RuII": 2, "O2": 1}
    assert rxn.stoichiometry == expected


def test_reaction_with_ignored_species():
    """Test reaction with species that should be ignored."""
    rxn = Reaction.from_dict({
        "equation": "RuII + hv -> RuII*",
        "type": "light",
        "quantum_yield": [0.8, 1.0]
    })
    
    expected = {"RuII": -1, "RuII*": 1}
    assert rxn.stoichiometry == expected


def test_invalid_equation_format():
    """Test that invalid equation format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid equation format"):
        Reaction.from_dict({
            "equation": "A + B = C",  # Missing ->
            "type": "dark"
        })


def test_reaction_attributes():
    """Test that all reaction attributes are properly set."""
    rxn_dict = {
        "equation": "A + B -> C + D",
        "type": "dark",
        "k_range": [1e3, 1e5],
        "quantum_yield": [0.5, 0.8]
    }
    
    rxn = Reaction.from_dict(rxn_dict)
    
    assert rxn.equation == "A + B -> C + D"
    assert rxn.type == "dark"
    assert rxn.reactants == ["A", "B"]
    assert rxn.products == ["C", "D"]
    assert rxn.k_range == [1e3, 1e5]
    assert rxn.quantum_yield == [0.5, 0.8]
    assert rxn.stoichiometry == {"A": -1, "B": -1, "C": 1, "D": 1}


def test_photochemical_reaction():
    """Test photochemical reaction parsing."""
    rxn = Reaction.from_dict({
        "equation": "RuII* + S2O8 -> RuIII + SO4_rad + SO4",
        "type": "dark",
        "k_range": [1e7, 1e9]
    })
    
    expected = {"RuII*": -1, "S2O8": -1, "RuIII": 1, "SO4_rad": 1, "SO4": 1}
    assert rxn.stoichiometry == expected