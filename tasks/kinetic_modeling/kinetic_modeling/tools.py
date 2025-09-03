import json

from corral.utils import tool


@tool
def setup_reaction_network(species: list[str], reactions: list[str]) -> str:
    """Create reaction network from species and reaction descriptions.

    Args:
        species: List of chemical species (e.g., ['A', 'B', 'O2', 'H2O'])
        reactions: List of reaction strings (e.g., ['A + O2 -> B + H2O', '2A -> A2'])
    """
    import uuid

    from .utils.reaction_parser import (
        parse_reactions,
        validate_species,
    )

    network_id = str(uuid.uuid4())

    # Validate species and reactions
    validated_species = validate_species(species)
    parsed_reactions = parse_reactions(reactions, validated_species)

    # # Store network in environment state
    # network_data = {
    #     "species": validated_species,
    #     "reactions": parsed_reactions,
    #     "stoichiometry_matrix": build_stoichiometry_matrix(
    #         parsed_reactions, validated_species
    #     ),
    # }

    # Save to environment state (accessible via self.state in environment)
    return json.dumps(
        {
            "network_id": network_id,
            "status": "created",
            "n_species": len(validated_species),
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
    from .utils.rate_laws import generate_rate_law

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
    # network = get_network_by_id(network_id)
    parsed_rate_laws = [json.loads(rl) for rl in rate_laws]

    # Build ODE system code from network + rate laws
    # ode_code = build_ode_system(network, parsed_rate_laws)

    # Collect all parameters from rate laws
    parameters_needed = {
        pname: pval
        for rl in parsed_rate_laws
        for pname, pval in rl["parameters"].items()
    }

    return json.dumps(
        {
            "network_id": network_id,
            # "ode_system_code": ode_code,
            # "n_equations": len(network["species"]),
            "parameters_needed": parameters_needed,
        }
    )


# @tool
# def fit_kinetic_parameters(
#     ode_system: str, experimental_data: str, initial_params: str, bounds: str = "{}"
# ) -> str:
#     """Fit kinetic parameters to experimental time-series data.

#     Args:
#         ode_system: JSON string from generate_ode_system containing Python code
#         experimental_data: Path to CSV file with columns ['time', 'species1', 'species2', ...]
#         initial_params: JSON string with parameter initial guesses {'k1': 0.1, 'k2': 0.05}
#         bounds: JSON string with parameter bounds {'k1': [0, 10], 'k2': [0, 1]}

#     Returns:
#         Fitted parameters with confidence intervals and goodness-of-fit metrics
#     """
#     from .utils.parameter_fitting import fit_parameters

#     ode_data = json.loads(ode_system)
#     initial_params_dict = json.loads(initial_params)
#     bounds_dict = json.loads(bounds) if bounds != "{}" else {}

#     # Load experimental data
#     # exp_data = load_experimental_data(experimental_data)

#     # Perform fitting
#     result = fit_parameters(
#         ode_code=ode_data["ode_system_code"],
#         # exp_data=exp_data,
#         initial_params=initial_params_dict,
#         bounds=bounds_dict,
#     )

#     return json.dumps(
#         {
#             "fitted_params": result["params"],
#             "param_errors": result["errors"],
#             "goodness_of_fit": {
#                 "r_squared": result["r_squared"],
#                 "rmse": result["rmse"],
#                 "aic": result["aic"],
#                 "bic": result["bic"],
#             },
#             "convergence": result["success"],
#         }
#     )


@tool
def validate_model_fit(
    fitted_params: str, experimental_data: str, ode_system: str
) -> str:
    """Perform comprehensive validation of fitted kinetic model.

    Args:
        fitted_params: JSON string from fit_kinetic_parameters
        experimental_data: Path to original experimental data CSV
        ode_system: JSON string with ODE system definition

    Returns:
        Validation metrics, residual analysis, and prediction intervals
    """
    import json

    import numpy as np
    from scipy.integrate import solve_ivp

    from .utils.data_utils import load_experimental_data
    from .utils.parameter_fitting import build_ode_function

    # Load inputs
    params = json.loads(fitted_params)["fitted_params"]
    exp_data = load_experimental_data(experimental_data)
    ode_data = json.loads(ode_system)
    ode_code = ode_data["ode_system_code"]

    # Build ODE function from generated code
    f = build_ode_function(ode_code, params)

    # Initial conditions from first row of exp data
    y0 = np.array([v[0] for v in exp_data["concentrations"].values()])
    tspan = (exp_data["time"][0], exp_data["time"][-1])
    t_eval = exp_data["time"]

    sol = solve_ivp(f, tspan, y0, t_eval=t_eval)

    # Compute residuals
    sim = {sp: sol.y[i, :] for i, sp in enumerate(exp_data["concentrations"].keys())}
    residuals = {}
    for sp, obs in exp_data["concentrations"].items():
        residuals[sp] = (obs - sim[sp]).tolist()

    # Compute validation metrics
    metrics = {}
    for sp, obs in exp_data["concentrations"].items():
        pred = sim[sp]
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        metrics[sp] = {"r2": r2, "rmse": rmse}

    # Prediction intervals (very naive: ± std of residuals)
    intervals = {
        sp: {
            "lower": (sim[sp] - np.std(res)).tolist(),
            "upper": (sim[sp] + np.std(res)).tolist(),
        }
        for sp, res in residuals.items()
    }

    return json.dumps(
        {
            "metrics": metrics,
            "residuals": residuals,
            "prediction_intervals": intervals,
            "success": True,
        }
    )
