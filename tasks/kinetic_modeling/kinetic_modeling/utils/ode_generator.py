from collections.abc import Callable

import numpy as np
from loguru import logger
from scipy.integrate import odeint


def build_ode_system(
    parsed_reactions: list,
    species: list,
    rate_constants: dict,
    other_multipliers: dict | None = None,
) -> Callable:
    """
    Build the system of ordinary differential equations.

    Args:
        parsed_reactions (list): List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
        species (list): Sorted list of all unique chemical species
    rate_constants (dict): Dictionary mapping rate constant identifiers to values
    other_multipliers (dict, optional): Dictionary mapping other multipliers to values

    Returns:
        Callable: A function that computes the derivatives for each species.
    """
    if other_multipliers is None:
        other_multipliers = {}

    def ode_system(y: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for each species.

        Args:
            y (np.ndarray): Current concentrations of each species.

        Returns:
            np.ndarray: The computed derivatives for each species.
        """
        dydt = np.zeros(len(species))

        # Create a dictionary mapping species to their current concentrations
        conc = {spec: y[i] for i, spec in enumerate(species)}

        # Compute contribution from each reaction
        for reaction in parsed_reactions:
            # Start with base rate constant
            rate = rate_constants[reaction["rate_constant"]]

            # Apply other multipliers if present
            for multiplier in reaction["other_multipliers"]:
                mult = other_multipliers[multiplier]

                # If the multiplier is a function, resolve its arguments, call it, and multiply the rate
                if isinstance(mult, dict) and "function" in mult:
                    arguments = mult["arguments"]
                    # resolve sources into keyword args
                    kwargs = {}
                    for parameter, source in arguments.items():
                        if source in conc:
                            kwargs[parameter] = conc[source]
                        elif source in other_multipliers and not isinstance(
                            other_multipliers[source], dict
                        ):
                            kwargs[parameter] = other_multipliers[source]
                        elif source in rate_constants:
                            kwargs[parameter] = rate_constants[source]
                        else:
                            raise KeyError(
                                f"Cannot resolve argument source '{source}' for multiplier '{multiplier}'"
                            )
                    rate *= mult["function"](**kwargs)

                # If the multiplier is a number, multiply directly
                else:
                    rate *= mult

            # Calculate concentration-dependent rate
            for reactant, stoich in reaction["reactants"].items():
                rate *= conc[reactant] ** stoich

            # Update derivatives for reactants (consumption)
            for reactant, stoich in reaction["reactants"].items():
                idx = species.index(reactant)
                dydt[idx] -= stoich * rate

            # Update derivatives for products (production)
            for product, stoich in reaction["products"].items():
                idx = species.index(product)
                dydt[idx] += stoich * rate

        return dydt

    return ode_system


def solve_ode_system(
    parsed_reactions,
    species,
    rate_constants,
    initial_conditions,
    times,
    other_multipliers=None,
) -> np.ndarray:
    """
    Solve the system of ODEs.

    Args:
        parsed_reactions (list): List of dictionaries with keys 'reactants', 'products', 'rate_constant', and optional 'photon_flux', 'sigma'
        species (list): Sorted list of all unique chemical species
        rate_constants (dict): Dictionary mapping rate constant identifiers to values
        initial_conditions (dict): Dictionary mapping species to their initial concentrations
        times (np.ndarray): Time points at which to solve the ODEs
        other_multipliers (dict, optional): Dictionary mapping other multipliers to values

    Returns:
        np.ndarray: Solution array with shape (len(times), len(species)).
    """
    if other_multipliers is None:
        other_multipliers = {}

    y0 = np.zeros(len(species))  # Initial concentrations default to zero

    for spec, conc in initial_conditions.items():
        if spec in species:
            idx = species.index(spec)
            y0[idx] = conc
        else:
            logger.warning(f"Warning: {spec} not in species list")

    # Build ODE system
    ode_system = build_ode_system(
        parsed_reactions, species, rate_constants, other_multipliers
    )

    # Solve ODEs
    return odeint(ode_system, y0, times, rtol=1e-8, atol=1e-10, mxstep=5000)


def build_stoichiometry_matrix(parsed_reactions, species):
    """
    Build stoichiometry matrix from parsed reactions.

    Args:
        parsed_reactions (list): List of dictionaries with reaction data
        species (list): List of species names

    Returns:
        np.ndarray: Stoichiometry matrix where rows are species and columns are reactions
    """
    n_species = len(species)
    n_reactions = len(parsed_reactions)

    matrix = np.zeros((n_species, n_reactions))

    for j, reaction in enumerate(parsed_reactions):
        for i, spec in enumerate(species):
            # Net stoichiometry = products - reactants
            product_coeff = reaction["products"].get(spec, 0)
            reactant_coeff = reaction["reactants"].get(spec, 0)
            matrix[i, j] = product_coeff - reactant_coeff

    return matrix


def extract_parameters(parsed_rate_laws):
    """
    Extract parameter lists from rate law definitions.

    Args:
        parsed_rate_laws (list): List of parsed rate law dictionaries

    Returns:
        list: List of unique parameter names
    """
    parameters = set()

    for rate_law in parsed_rate_laws:
        if "parameters" in rate_law:
            parameters.update(rate_law["parameters"])

    return sorted(parameters)
