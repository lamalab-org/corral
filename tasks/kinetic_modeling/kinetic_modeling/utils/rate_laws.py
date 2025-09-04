from typing import Any

from kinetic_modeling.utils.reaction_parser import parse_reactions


def generate_rate_law(reaction: str, mechanism: str = "elementary") -> dict[str, Any]:
    """
    Generate rate law expression for a reaction based on different mechanisms.

    Args:
        reaction (str): Reaction string (e.g., '[A] + 2 [B] > [C] + [D], k1')
        mechanism (str): Mechanism type ('elementary', 'michaelis_menten', 'langmuir_hinshelwood')

    Returns:
        dict: Dictionary containing rate expression, parameters, and units
    """
    if mechanism == "elementary":
        return _generate_elementary_rate_law(reaction)
    elif mechanism == "michaelis_menten":
        return _generate_michaelis_menten_rate_law(reaction)
    elif mechanism == "langmuir_hinshelwood":
        return _generate_langmuir_hinshelwood_rate_law(reaction)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def _generate_elementary_rate_law(reaction: str) -> dict[str, Any]:
    """
    Generate elementary rate law (rate = k * [A]^a * [B]^b * ...).

    Args:
        reaction (str): Reaction string (e.g., '[A] + 2 [B] > [C] + [D], k1')

    Returns:
        dict: Dictionary containing rate expression, parameters, and units
    """
    # Use the existing parse_reactions function for robust parsing
    parsed_reactions, _ = parse_reactions([reaction])

    if not parsed_reactions:
        raise ValueError(f"Could not parse reaction: {reaction}")

    reaction_data = parsed_reactions[0]
    reactants = reaction_data["reactants"]

    # Generate rate expression
    rate_terms = []
    parameters = ["k"]

    for species, stoich in reactants.items():
        if stoich == 1:
            rate_terms.append(f"{species}")
        else:
            rate_terms.append(f"{species}^{stoich}")

    expression = f"k * {' * '.join(rate_terms)}" if rate_terms else "k"

    return {
        "expression": expression,
        "parameters": parameters,
        "units": "M/s",
        "mechanism": "elementary",
    }


def _generate_michaelis_menten_rate_law(reaction: str) -> dict[str, Any]:
    """
    Generate Michaelis-Menten rate law (rate = Vmax * [S] / (Km + [S])).

    Args:
        reaction (str): Reaction string

    Returns:
        dict: Rate law data
    """
    # Use the existing parse_reactions function for robust parsing
    parsed_reactions, _ = parse_reactions([reaction])

    if not parsed_reactions:
        raise ValueError(f"Could not parse reaction: {reaction}")

    reaction_data = parsed_reactions[0]
    reactants = reaction_data["reactants"]

    if not reactants:
        raise ValueError("No reactants found for Michaelis-Menten kinetics")

    # First reactant is substrate
    substrate = next(iter(reactants.keys()))

    expression = f"Vmax * {substrate} / (Km + {substrate})"
    parameters = ["Vmax", "Km"]

    return {
        "expression": expression,
        "parameters": parameters,
        "units": "M/s",
        "mechanism": "michaelis_menten",
    }


def _generate_langmuir_hinshelwood_rate_law(reaction: str) -> dict[str, Any]:
    """
    Generate Langmuir-Hinshelwood rate law for surface reactions.

    Args:
        reaction (str): Reaction string

    Returns:
        dict: Rate law data
    """
    # Use the existing parse_reactions function for robust parsing
    parsed_reactions, _ = parse_reactions([reaction])

    if not parsed_reactions:
        raise ValueError(f"Could not parse reaction: {reaction}")

    reaction_data = parsed_reactions[0]
    reactants = reaction_data["reactants"]

    if len(reactants) < 2:
        raise ValueError("Langmuir-Hinshelwood requires at least 2 reactants")

    # For simplicity, assume two reactants A and B
    species_list = list(reactants.keys())
    A, B = species_list[0], species_list[1]

    # Rate = k * KA * KB * [A] * [B] / ((1 + KA * [A] + KB * [B])^2)
    expression = f"k * KA * KB * {A} * {B} / ((1 + KA * {A} + KB * {B})^2)"
    parameters = ["k", "KA", "KB"]

    return {
        "expression": expression,
        "parameters": parameters,
        "units": "M/s",
        "mechanism": "langmuir_hinshelwood",
    }
