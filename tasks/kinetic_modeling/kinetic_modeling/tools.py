import json
import uuid
from typing import Any

from kinetic_modeling.utils.ode_generator import (
    build_ode_system,
    build_stoichiometry_matrix,
    extract_parameters,
)
from kinetic_modeling.utils.parameter_fitting import (
    calculate_confidence_intervals,
    fit_parameters,
    load_experimental_data,
)
from kinetic_modeling.utils.rate_laws import generate_rate_law
from kinetic_modeling.utils.reaction_parser import parse_reactions

from corral.base import Tool
from corral.utils import tool

# Module-level storage for reaction networks
# Tools do not have access to the env state which force us to do this.
_NETWORK_STORAGE = {}


@tool
def setup_reaction_network(reactions: list[str]) -> str:
    """Create reaction network from species and reaction descriptions.

    Args:
        species: List of chemical species (e.g., ['A', 'B', 'O2', 'H2O'])
        reactions: List of reaction strings (e.g., ['A + O2 -> B + H2O', '2A -> A2'])
    """
    network_id = str(uuid.uuid4())

    # Validate species and reactions
    parsed_reactions, sorted_species = parse_reactions(reactions)

    # Store network in environment state
    network_data = {
        "species": sorted_species,
        "reactions": parsed_reactions,
        "stoichiometry_matrix": build_stoichiometry_matrix(
            parsed_reactions, sorted_species
        ),
    }

    # Save to module-level storage (accessible globally)
    _NETWORK_STORAGE[network_id] = network_data

    return json.dumps(
        {
            "network_id": network_id,
            "status": "created",
            "n_species": len(sorted_species),
            "n_reactions": len(parsed_reactions),
        }
    )


@tool
def derive_rate_law(reaction: str, mechanism: str = "elementary") -> str:
    """Derive rate law expression for a reaction.

    Args:
        reaction: Reaction string (e.g., 'A + 2B -> C + D')
        mechanism: Mechanism type ('elementary', 'michaelis_menten', 'langmuir_hinshelwood')

    Returns:
        Rate law parameters and mathematical expression as JSON string
    """

    rate_law = generate_rate_law(reaction, mechanism)

    return json.dumps(
        {
            "reaction": reaction,
            "mechanism": mechanism,
            "rate_expression": rate_law["expression"],
            "parameters": rate_law["parameters"],
            "units": rate_law["units"],
        }
    )


def _get_network_storage():
    return _NETWORK_STORAGE


def get_network_by_id(network_id: str) -> dict[str, Any]:
    """
    Retrieve network by ID from module-level storage.

    Parameters:
    -----------
    network_id : str
        Network identifier

    Returns:
    --------
    dict
        Network data

    Raises:
    -------
    ValueError
        If network_id is not found in storage
    """
    network_storage = _get_network_storage()

    if network_id not in network_storage:
        raise ValueError(f"Network {network_id} not found in storage")

    return network_storage[network_id]


@tool
def generate_ode_system(network_id: str, rate_laws: list[str]) -> str:
    """Generate system of ODEs from reaction network and rate laws.

    Args:
        network_id: ID from setup_reaction_network
        rate_laws: List of rate law JSON strings from derive_rate_law

    Returns:
        ODE system as executable Python code string
    """
    # Retrieve network from environment state
    network = get_network_by_id(network_id)
    parsed_rate_laws = [json.loads(rl) for rl in rate_laws]

    # Extract rate constants from rate laws
    rate_constants = {}
    for rate_law in parsed_rate_laws:
        if "parameters" in rate_law:
            for param in rate_law["parameters"]:
                rate_constants[param] = 1.0  # Default value, will be fitted later

    ode_code = build_ode_system(
        parsed_reactions=network["reactions"],
        species=network["species"],
        rate_constants=rate_constants,
    )

    return json.dumps(
        {
            "network_id": network_id,
            "ode_system_code": ode_code,
            "n_equations": len(network["species"]),
            "parameters_needed": extract_parameters(parsed_rate_laws),
        }
    )


@tool
def fit_kinetic_parameters(
    ode_system: str, experiment: str, initial_params: str, bounds: str = "{}"
) -> str:
    """Fit kinetic parameters to experimental time-series data.

    Args:
        ode_system: JSON string from generate_ode_system containing Python code
        experiment: name of the experiment to fit
        initial_params: JSON string with parameter initial guesses {'k1': 0.1, 'k2': 0.05}
        bounds: JSON string with parameter bounds {'k1': [0, 10], 'k2': [0, 1]}

    Returns:
        Fitted parameters with confidence intervals and goodness-of-fit metrics
    """

    try:
        ode_data = json.loads(ode_system)
        initial_params_dict = json.loads(initial_params)
        bounds_dict = json.loads(bounds) if bounds != "{}" else {}

        # Load experimental data with improved error handling
        try:
            exp_data = load_experimental_data(experiment)
        except RuntimeError as e:
            # Return detailed error information about available experiments
            return json.dumps(
                {
                    "error": "Dataset loading failed",
                    "message": str(e),
                    "experiment_requested": experiment,
                    "success": False,
                }
            )

        # Perform fitting
        result = fit_parameters(
            ode_code=ode_data["ode_system_code"],
            exp_data=exp_data,
            initial_params=initial_params_dict,
            bounds=bounds_dict,
        )

        return json.dumps(
            {
                "fitted_params": result["params"],
                "param_errors": result["errors"],
                "goodness_of_fit": {
                    "r_squared": result["r_squared"],
                    "rmse": result["rmse"],
                    "aic": result["aic"],
                    "bic": result["bic"],
                },
                "convergence": result["success"],
            }
        )

    except Exception as e:
        # Handle any other errors that might occur during fitting
        return json.dumps(
            {"error": "Parameter fitting failed", "message": str(e), "success": False}
        )


@tool
def validate_model_fit(fitted_params: str, experimental_data: str) -> str:
    """Perform comprehensive validation of fitted kinetic model.

    Args:
        fitted_params: JSON string from fit_kinetic_parameters
        experimental_data: Path to original experimental data CSV
        ode_system: JSON string with ODE system definition

    Returns:
        Validation metrics, residual analysis, and prediction intervals
    """
    try:
        fitted_data = json.loads(fitted_params)

        # Load experimental data with improved error handling
        try:
            exp_data = load_experimental_data(experimental_data)
        except RuntimeError as e:
            # Return detailed error information about available experiments
            return json.dumps(
                {
                    "error": "Dataset loading failed",
                    "message": str(e),
                    "experiment_requested": experimental_data,
                    "success": False,
                }
            )

        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(
            fitted_data["fitted_params"],
            exp_data,
            param_errors=fitted_data.get("param_errors", None),
        )

        # Use actual goodness-of-fit metrics from fitted parameters
        goodness_of_fit = fitted_data.get("goodness_of_fit", {})

        validation_results = {
            "confidence_intervals": confidence_intervals,
            "goodness_of_fit": goodness_of_fit,
            "parameter_correlations": {},  # Not available in current implementation
            "residual_analysis": {
                "rmse": goodness_of_fit.get("rmse", "not_available"),
                "r_squared": goodness_of_fit.get("r_squared", "not_available"),
            },
            "model_selection": {
                "aic": goodness_of_fit.get("aic", "not_available"),
                "bic": goodness_of_fit.get("bic", "not_available"),
            },
            "convergence_status": fitted_data.get("convergence", "unknown"),
        }

        return json.dumps(validation_results)

    except Exception as e:
        # Handle any other errors that might occur during validation
        return json.dumps(
            {"error": "Model validation failed", "message": str(e), "success": False}
        )


@tool
def lookup_previous_works():
    return "This tool is not implemented yet."


def create_tools() -> dict[str, Tool]:
    """Create a dictionary of all available tools for the agent environment"""
    return {
        "setup_reaction_network": setup_reaction_network,
        "derive_rate_law": derive_rate_law,
        "generate_ode_system": generate_ode_system,
        "fit_kinetic_parameters": fit_kinetic_parameters,
        "validate_model_fit": validate_model_fit,
        "lookup_previous_works": lookup_previous_works,
    }
