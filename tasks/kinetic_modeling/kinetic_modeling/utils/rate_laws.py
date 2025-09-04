import re
from typing import Any


def generate_rate_law(reaction: str, mechanism: str = "elementary") -> dict[str, Any]:
    """
    Generate rate law expression for a reaction based on different mechanisms.

    Args:
        reaction (str): Reaction string (e.g., 'A + 2B -> C + D')
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
        reaction (str): Reaction string (e.g., 'A + 2B -> C + D')

    Returns:
        dict: Dictionary containing rate expression, parameters, and units
    """
    # Parse reaction to extract reactants and stoichiometry
    reactants_str = reaction.split("->")[0].strip()

    # Extract species and coefficients
    reactants = {}
    species_matches = re.findall(
        r"(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z][A-Za-z0-9]*)", reactants_str
    )

    for coeff_str, species in species_matches:
        coeff = float(coeff_str) if coeff_str else 1.0
        reactants[species] = coeff

    # Generate rate expression
    rate_terms = []
    parameters = ["k"]

    for species, stoich in reactants.items():
        if stoich == 1:
            rate_terms.append(f"[{species}]")
        else:
            rate_terms.append(f"[{species}]^{stoich}")

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
    # Extract substrate (first reactant)
    reactants_str = reaction.split("->")[0].strip()
    species_matches = re.findall(r"([A-Za-z][A-Za-z0-9]*)", reactants_str)

    if not species_matches:
        raise ValueError("No reactants found for Michaelis-Menten kinetics")

    substrate = species_matches[0]  # First reactant is substrate

    expression = f"Vmax * [{substrate}] / (Km + [{substrate}])"
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
    # Parse reactants
    reactants_str = reaction.split("->")[0].strip()
    species_matches = re.findall(r"([A-Za-z][A-Za-z0-9]*)", reactants_str)

    if len(species_matches) < 2:
        raise ValueError("Langmuir-Hinshelwood requires at least 2 reactants")

    # For simplicity, assume two reactants A and B
    A, B = species_matches[0], species_matches[1]

    # Rate = k * KA * KB * [A] * [B] / ((1 + KA * [A] + KB * [B])^2)
    expression = f"k * KA * KB * [{A}] * [{B}] / ((1 + KA * [{A}] + KB * [{B}])^2)"
    parameters = ["k", "KA", "KB"]

    return {
        "expression": expression,
        "parameters": parameters,
        "units": "M/s",
        "mechanism": "langmuir_hinshelwood",
    }


def calculate_rate(
    rate_law: dict[str, Any],
    concentrations: dict[str, float],
    parameters: dict[str, float],
) -> float:
    """
    Calculate reaction rate given rate law, concentrations, and parameters.

    Args:
        rate_law (dict): Rate law dictionary from generate_rate_law
        concentrations (dict): Current species concentrations
        parameters (dict): Rate law parameters
    Returns:
        float: Calculated reaction rate
    """
    # This is a simplified implementation
    # In practice, would need a more robust expression evaluator
    expression = rate_law["expression"]

    # Replace parameter names with values
    for param, value in parameters.items():
        expression = expression.replace(param, str(value))

    # Replace concentration terms
    for species, conc in concentrations.items():
        expression = expression.replace(f"[{species}]", str(conc))

    try:
        # Evaluate the expression (this is unsafe for production!)
        # In practice, use a proper expression parser
        return eval(expression.replace("^", "**"))
    except Exception as e:
        # Preserve the original exception context for easier debugging
        raise ValueError(f"Error evaluating rate expression: {e}") from e
