import re


def validate_species(species: list[str]) -> list[str]:
    """
    Ensure species names are valid chemical identifiers.
    Allows element symbols, numbers, parentheses, and charges.
    Examples: H2O, O2, Fe(OH)3, SO4^2-, NH4+, CH3-CH2-OH
    """
    cleaned = []
    chem_pattern = re.compile(r"^[A-Za-z0-9()^+\-·]+$")
    for s in species:
        if not chem_pattern.match(s):
            raise ValueError(f"Invalid species name: {s}")
        cleaned.append(s)
    return cleaned


def parse_reactions(reactions: list[str], species: list[str]) -> list[dict]:
    """Parse reaction strings like '2A + B -> C' into stoichiometry dicts."""
    parsed = []
    for rxn in reactions:
        if "->" not in rxn:
            raise ValueError(f"Invalid reaction format: {rxn}")
        lhs, rhs = rxn.split("->")
        reactants = _parse_side(lhs.strip(), species)
        products = _parse_side(rhs.strip(), species)
        parsed.append({"reactants": reactants, "products": products, "raw": rxn})
    return parsed


def _parse_side(side: str, species: list[str]) -> dict[str, int]:
    parts = [p.strip() for p in side.split("+")]
    coeffs = {}
    for p in parts:
        m = re.match(r"^(\d*)([A-Za-z0-9_]+)$", p)
        if not m:
            raise ValueError(f"Invalid term: {p}")
        coeff = int(m.group(1)) if m.group(1) else 1
        sp = m.group(2)
        if sp not in species:
            raise ValueError(f"Species {sp} not in species list")
        coeffs[sp] = coeffs.get(sp, 0) + coeff
    return coeffs


def build_stoichiometry_matrix(reactions: list[dict], species: list[str]):
    """Build stoichiometry matrix (n_species x n_reactions)."""
    import numpy as np

    S = np.zeros((len(species), len(reactions)), dtype=int)
    for j, rxn in enumerate(reactions):
        for sp, coeff in rxn["products"].items():
            S[species.index(sp), j] += coeff
        for sp, coeff in rxn["reactants"].items():
            S[species.index(sp), j] -= coeff
    return S
