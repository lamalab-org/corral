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
from corral.utils import tool, vector_database_search

# Module-level storage for reaction networks
# Tools do not have access to the env state which force us to do this.
_NETWORK_STORAGE = {}


@tool
def setup_reaction_network(reactions: list[str]) -> str:
    """[BRIEF] Create a reaction network from species and reaction descriptions. [/BRIEF]

    [DETAILED] This tool sets up a reaction network by parsing the provided reactions, validating species, and constructing the stoichiometry matrix. The network is stored in a module-level storage for later retrieval. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need to create a structured reaction network for a reaction/s.
    - When you want to validate the reactions and species involved.
    - When you need to prepare the network for further analysis, such as rate law derivation or ODE system generation.
    [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a list of reaction strings ready. [/PREREQUISITE]
    2. [CURRENT] Call this tool with the list of reactions to set up the network. [/CURRENT]
    3. [FOLLOW_UP] Use the returned network ID in subsequent tools like `derive_rate_law` or `generate_ode_system`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It generates a unique network ID for the reaction network.
    - It parses the reaction strings to extract species and validate them.
    - It constructs the stoichiometry matrix based on the parsed reactions.
    - It stores the network data in a module-level dictionary for later access.
    - It returns a JSON string containing the network ID and summary information. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `setup_reaction_network(reactions=['A + O2 -> B + H2O', '2A -> A2'])`,
        `setup_reaction_network(reactions=['CH4 + 2O2 -> CO2 + 2H2O', 'CO + 0.5O2 -> CO2'])`,
        `setup_reaction_network(reactions=['N2 + 3H2 -> 2NH3', 'NH3 -> N2 + 3H2'])`,
        `setup_reaction_network(reactions=['C2H4 + H2 -> C2H6', 'C2H6 -> C2H4 + H2'])`,
        `setup_reaction_network(reactions=['A + B -> C', 'C -> A + B'])`,
    ]
    [/SYNTACTICAL]

    Args:
        reactions (list[str]):
            [BRIEF] List of reaction strings. [/BRIEF]
            [DETAILED] Each reaction string should be in the format 'reactants -> products', where reactants and products are combinations of species with optional stoichiometric coefficients. [/DETAILED]
            [SYNTACTICAL] String format of a reaction [/SYNTACTICAL]
            [EXAMPLES] "A + O2 -> B + H2O", "2A -> A2" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with network ID and summary. [/BRIEF]
            [DETAILED] The returned JSON string contains the unique network ID, the number of species, and the number of reactions in the network. This ID can be used in subsequent tools for further analysis. [/DETAILED]
            [EXAMPLES] '{"network_id": "123e4567-e89b-123e-4567-426614174000", "n_species": 4, "n_reactions": 2}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If any reaction string is malformed or contains invalid species. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if it encounters any issues while parsing the reaction strings, such as unrecognized species or incorrect formatting. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure all reaction strings are correctly formatted and all species are valid before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The tool does not currently handle reversible reactions; all reactions are treated as irreversible.
        - It assumes that species names are case-sensitive and must be consistent across all reactions.
        - The stoichiometry matrix is built assuming mass-action kinetics; other kinetic models are not considered at this stage.
    [/LIMITATIONS]
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
    """
    [BRIEF] Derive rate law expression for a reaction. [/BRIEF]

    [DETAILED] This tool derives the rate law expression for a given chemical reaction based on the specified reaction mechanism. It supports various mechanisms such as elementary, Michaelis-Menten, and Langmuir-Hinshelwood. The derived rate law includes the mathematical expression, parameters involved, and their units. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a specific reaction and need to determine its rate law.
    - When you want to understand how the reaction rate depends on the concentrations of reactants and products.
    - When preparing for kinetic modeling or simulation of reaction networks. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a reaction string and know the mechanism type. [/PREREQUISITE]
    2. [CURRENT] Call this tool with the reaction and mechanism to derive the rate law. [/CURRENT]
    3. [FOLLOW_UP] Use the derived rate law in subsequent tools like `generate_ode_system` or `fit_kinetic_parameters`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It takes a reaction string and a mechanism type as input.
    - It uses predefined templates and rules to derive the rate law expression.
    - It identifies the parameters involved in the rate law and their respective units.
    - It returns a JSON string containing the reaction, mechanism, rate expression, parameters, and units. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `derive_rate_law(reaction='A + B -> C', mechanism='elementary')`,
        `derive_rate_law(reaction='S + E -> ES -> P + E', mechanism='michaelis_menten')`,
        `derive_rate_law(reaction='A + B -> C', mechanism='langmuir_hinshelwood')`,
        `derive_rate_law(reaction='2A -> A2', mechanism='elementary')`,
        `derive_rate_law(reaction='CH4 + 2O2 -> CO2 + 2H2O', mechanism='elementary')`,
    ]
    [/SYNTACTICAL]

    Args:
        reaction (str):
            [BRIEF] Reaction string. [/BRIEF]
            [DETAILED] The reaction string should be in the format 'reactants -> products', where reactants and products are combinations of species with optional stoichiometric coefficients. [/DETAILED]
            [SYNTACTICAL] String format of a reaction [/SYNTACTICAL]
            [EXAMPLES] "A + B -> C", "S + E -> ES -> P + E" [/EXAMPLES]

        mechanism (str):
            [BRIEF] Mechanism type. [/BRIEF]
            [DETAILED] The mechanism type determines the form of the rate law. Supported mechanisms include 'elementary' for simple reactions, 'michaelis_menten' for enzyme-catalyzed reactions, and 'langmuir_hinshelwood' for surface reactions. [/DETAILED]
            [SYNTACTICAL] One of 'elementary', 'michaelis_menten', 'langmuir_hinshelwood' [/SYNTACTICAL]
            [EXAMPLES] "elementary", "michaelis_menten" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with rate law details. [/BRIEF]
            [DETAILED] The returned JSON string contains the original reaction, the mechanism type, the derived rate expression, a list of parameters involved in the rate law, and their units. This information is essential for kinetic modeling and simulation. [/DETAILED]
            [EXAMPLES] '{"reaction": "A + B -> C", "mechanism": "elementary", "rate_expression": "k1*[A]*[B]", "parameters": ["k1"], "units": {"k1": "L/(mol*s)"}}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the reaction string is malformed or the mechanism type is unsupported. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if it encounters issues while parsing the reaction string or if the specified mechanism is not recognized. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the reaction string is correctly formatted and the mechanism is one of the supported types before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The tool currently supports a limited set of mechanisms; more complex mechanisms may not be handled.
        - It assumes ideal behavior and does not account for factors like temperature or pressure variations.
        - The derived rate laws are based on standard kinetic models and may not capture all real-world complexities.
    [/LIMITATIONS]
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


def _get_network_storage() -> dict[str, Any]:
    return _NETWORK_STORAGE


def get_network_by_id(network_id: str) -> dict[str, Any]:
    """
    Retrieve network by ID from module-level storage.

    Args:
        network_id (str): Network identifier

    Returns:
        dict: Network data
    """
    network_storage = _get_network_storage()

    if network_id not in network_storage:
        raise ValueError(f"Network {network_id} not found in storage")

    return network_storage[network_id]


@tool
def generate_ode_system(network_id: str, rate_laws: list[str]) -> str:
    """
    [BRIEF] Generate ODE system from reaction network and rate laws. [/BRIEF]

    [DETAILED] This tool generates a system of ordinary differential equations (ODEs) based on a specified reaction network and associated rate laws. It retrieves the reaction network using the provided network ID, parses the rate laws, and constructs the ODE system in executable Python code. The generated ODE system can be used for kinetic modeling and simulation of the reaction dynamics. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have a defined reaction network and corresponding rate laws.
    - When you need to create a mathematical model of the reaction kinetics.
    - When preparing for parameter fitting or simulation of the reaction system. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a reaction network ID (use `setup_reaction_network`) and derived rate laws (`use derive_rate_laws`). [/PREREQUISITE]
    2. [CURRENT] Call this tool with the network ID and rate laws to generate the ODE system. [/CURRENT]
    3. [FOLLOW_UP] Use the generated ODE system in subsequent tools like `fit_kinetic_parameters`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It retrieves the reaction network from module-level storage using the provided ID.
    - It parses the rate laws from JSON strings to extract rate expressions and parameters.
    - It constructs the ODE system based on the stoichiometry of the reactions and the provided rate laws.
    - It returns a JSON string containing the network ID, the ODE system code, the number of equations, and the parameters needed for simulation. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `generate_ode_system(network_id='123e4567-e89b-123e-4567-426614174000', rate_laws=['{"reaction": "A + B -> C", "mechanism": "elementary", "rate_expression": "k1*[A]*[B]", "parameters": ["k1"], "units": {"k1": "L/(mol*s)"}}'])`,
        `generate_ode_system(network_id='223e4567-e89b-123e-4567-426614174001', rate_laws=['{"reaction": "S + E -> ES -> P + E", "mechanism": "michaelis_menten", "rate_expression": "(Vmax*[S])/(Km + [S])", "parameters": ["Vmax", "Km"], "units": {"Vmax": "mol/(L*s)", "Km": "mol/L"}}'])`,
        `generate_ode_system(network_id='323e4567-e89b-123e-4567-426614174002', rate_laws=['{"reaction": "A + B -> C", "mechanism": "langmuir_hinshelwood", "rate_expression": "(k1*[A]*[B])/(1 + K1*[A] + K2*[B])", "parameters": ["k1", "K1", "K2"], "units": {"k1": "L/(mol*s)", "K1": "L/mol", "K2": "L/mol"}}'])`,
        `generate_ode_system(network_id='423e4567-e89b-123e-4567-426614174003', rate_laws=['{"reaction": "2A -> A2", "mechanism": "elementary", "rate_expression": "k1*[A]^2", "parameters": ["k1"], "units": {"k1": "L^2/(mol^2*s)"}}'])`,
        `generate_ode_system(network_id='523e4567-e89b-123e-4567-426614174004', rate_laws=['{"reaction": "CH4 + 2O2 -> CO2 + 2H2O", "mechanism": "elementary", "rate_expression": "k1*[CH4]*[O2]^2", "parameters": ["k1"], "units": {"k1": "L^3/(mol^3*s)"}}'])`,
    ]
    [/SYNTACTICAL]

    Args:
        network_id (str):
            [BRIEF] Network ID. [/BRIEF]
            [DETAILED] The network ID should be obtained from a previous call to the `setup_reaction_network` tool. It uniquely identifies the reaction network to be used. [/DETAILED]
            [SYNTACTICAL] UUID string [/SYNTACTICAL]
            [EXAMPLES] "123e4567-e89b-123e-4567-426614174000" [/EXAMPLES]

        rate_laws (list[str]):
            [BRIEF] List of rate law JSON strings. [/BRIEF]
            [DETAILED] Each rate law should be provided as a JSON string, typically obtained from the `derive_rate_law` tool. The JSON should include the reaction, mechanism, rate expression, parameters, and units. [/DETAILED]
            [SYNTACTICAL] List of JSON strings [/SYNTACTICAL]
            [EXAMPLES] ['{"reaction": "A + B -> C", "mechanism": "elementary", "rate_expression": "k1*[A]*[B]", "parameters": ["k1"], "units": {"k1": "L/(mol*s)"}}'] [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with ODE system details. [/BRIEF]
            [DETAILED] The returned JSON string contains the network ID, the generated ODE system as executable Python code, the number of equations in the system, and a list of parameters needed for simulation. This information is crucial for kinetic modeling and parameter fitting. [/DETAILED]
            [EXAMPLES] '{"network_id": "123e4567-e89b-123e-4567-426614174000", "ode_system_code": "def ode_system(...)", "n_equations": 3, "parameters_needed": ["k1"]}' [/EXAMPLES]

    [RAISES] Exceptions:
        ValueError:
            [ERROR_WHEN] If the network ID is not found or if rate laws are malformed. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if it cannot find the specified network ID in storage or if it encounters issues while parsing the rate law JSON strings. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the network ID is valid and that all rate laws are correctly formatted JSON strings before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The tool assumes that the reaction network and rate laws are compatible; mismatches may lead to incorrect ODE systems.
        - It does not currently support dynamic changes to the reaction network; the network must be static once created.
        - The generated ODE system is in Python code format and may require further adaptation for use in specific simulation environments.
    [/LIMITATIONS]
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
    """
    [BRIEF] Fit kinetic parameters to experimental data. [/BRIEF]

    [DETAILED] This tool fits kinetic parameters of a reaction network to experimental time-series data. It takes an ODE system definition, experimental dataset name, initial parameter guesses, and optional bounds for the parameters. The fitting process uses optimization techniques to minimize the difference between the model predictions and the experimental data. The output includes the fitted parameters, confidence intervals, and goodness-of-fit metrics. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have an ODE system representing a reaction network and corresponding experimental data.
    - When you need to estimate kinetic parameters that best describe the observed reaction dynamics.
    - When preparing for model validation or further kinetic analysis. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a generated ODE system and experimental data available. You can prepare the system using `generate_ode_system`. [/PREREQUISITE]
    2. [CURRENT] Call this tool with the ODE system, experiment name, initial parameters, and bounds to fit the parameters. [/CURRENT]
    3. [FOLLOW_UP] Use the fitted parameters in subsequent tools like `validate_model_fit`. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It parses the ODE system and initial parameters from JSON strings.
    - It loads the experimental data based on the provided experiment name.
    - It performs parameter fitting using optimization algorithms to minimize the difference between model predictions and experimental observations.
    - It calculates confidence intervals and goodness-of-fit metrics for the fitted parameters.
    - It returns a JSON string containing the fitted parameters, their errors, goodness-of-fit metrics, and convergence status. [/CONTEXTUAL]


    [SYNTACTICAL] Usage examples:
    [
        `fit_kinetic_parameters(ode_system='{"ode_system_code": "(...)", "n_equations": 3, "parameters_needed": ["k1"]}', experiment='experiment_1', initial_params='{"k1": 0.1}', bounds='{"k1": [0, 10]}')`,
        `fit_kinetic_parameters(ode_system='{"ode_system_code": "(...)", "n_equations": 4, "parameters_needed": ["k1", "k2"]}', experiment='experiment_2', initial_params='{"k1": 0.05, "k2": 0.01}', bounds='{"k1": [0, 5], "k2": [0, 1]}')`,
        `fit_kinetic_parameters(ode_system='{"ode_system_code": "(...)", "n_equations": 2, "parameters_needed": ["k1"]}', experiment='experiment_3', initial_params='{"k1": 0.2}', bounds='{}')`,
        `fit_kinetic_parameters(ode_system='{"ode_system_code": "(...)", "n_equations": 5, "parameters_needed": ["k1", "k2", "k3"]}', experiment='experiment_4', initial_params='{"k1": 0.1, "k2": 0.05, "k3": 0.01}', bounds='{"k1": [0, 10], "k2": [0, 1], "k3": [0, 0.1]}')`,
        `fit_kinetic_parameters(ode_system='{"ode_system_code": "(...)", "n_equations": 3, "parameters_needed": ["k1", "k2"]}', experiment='experiment_5', initial_params='{"k1": 0.1, "k2": 0.1}', bounds='{"k1": [0, 10], "k2": [0, 10]}')`,
    ]
    [/SYNTACTICAL]

    Args:
        ode_system (str):
            [BRIEF] ODE system JSON string. [/BRIEF]
            [DETAILED] The ODE system should be provided as a JSON string, typically obtained from the `generate_ode_system` tool. It includes the ODE system code and the parameters needed for fitting. [/DETAILED]
            [SYNTACTICAL] JSON string [/SYNTACTICAL]
            [EXAMPLES] '{"ode_system_code": "def ode_system(...)", "n_equations": 3, "parameters_needed": ["k1"]}' [/EXAMPLES]

        experiment (str):
            [BRIEF] Experiment name. [/BRIEF]
            [DETAILED] The name of the experimental dataset to be used for fitting. This should correspond to a dataset that can be loaded by the `load_experimental_data` function. [/DETAILED]
            [SYNTACTICAL] String identifier [/SYNTACTICAL]
            [EXAMPLES] "experiment_1" [/EXAMPLES]

        initial_params (str):
            [BRIEF] Initial parameter guesses JSON string. [/BRIEF]
            [DETAILED] A JSON string containing initial guesses for the parameters to be fitted. The keys should match the parameter names in the ODE system, and the values should be numerical guesses. [/DETAILED]
            [SYNTACTICAL] JSON string [/SYNTACTICAL]
            [EXAMPLES] '{"k1": 0.1, "k2": 0.05}' [/EXAMPLES]

        bounds (str):
            [BRIEF] Parameter bounds JSON string. [/BRIEF]
            [DETAILED] A JSON string specifying bounds for the parameters to be fitted. The keys should match the parameter names, and the values should be lists or tuples with the lower and upper bounds. If no bounds are needed, an empty JSON object '{}' can be provided. [/DETAILED]
            [SYNTACTICAL] JSON string [/SYNTACTICAL]
            [EXAMPLES] '{"k1": [0, 10], "k2": [0, 1]}' or '{}' [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with fitting results. [/BRIEF]
            [DETAILED] The returned JSON string contains the fitted parameters, their confidence intervals, goodness-of-fit metrics (R-squared, RMSE, AIC, BIC), and convergence status. This information is essential for evaluating the quality of the fit and for further kinetic analysis. [/DETAILED]
            [EXAMPLES] '{"fitted_params": {"k1": 0.15}, "param_errors": {"k1": 0.01}, "goodness_of_fit": {"r_squared": 0.95, "rmse": 0.02, "aic": 10.5, "bic": 12.3}, "convergence": true}' [/EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
            [ERROR_WHEN] If the experimental dataset cannot be loaded. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a RuntimeError if it fails to load the specified experimental dataset, possibly due to an incorrect name or missing data. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify that the experiment name is correct and that the dataset is available before calling this tool. [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] If the ODE system or parameter JSON strings are malformed. [//ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if it encounters issues while parsing the ODE system or parameter JSON strings. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure that the ODE system and parameter strings are valid JSON before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The fitting process may be sensitive to the initial parameter guesses; poor guesses can lead to convergence issues or suboptimal fits.
        - The tool assumes that the experimental data is of sufficient quality and quantity for reliable parameter estimation.
        - It currently supports only certain types of optimization algorithms; more complex fitting scenarios may require custom implementations.
    [/LIMITATIONS]
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
    """
    [BRIEF] Validate fitted kinetic model against experimental data. [/BRIEF]

    [DETAILED] This tool validates a fitted kinetic model by comparing its predictions against experimental time-series data. It takes the fitted parameters, the name of the experiment, and the ODE system definition. The validation process includes calculating confidence intervals, goodness-of-fit metrics, and performing residual analysis. The output provides a comprehensive assessment of the model's performance and reliability. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you have fitted kinetic parameters and want to assess the quality of the fit.
    - When you need to validate the model against experimental observations.
    - When preparing for model refinement or reporting results. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have fitted parameters and access to the original experimental data. [/PREREQUISITE]
    2. [CURRENT] Call this tool with the fitted parameters, experimental data path, and ODE system to validate the model. [/CURRENT]
    3. [FOLLOW_UP] Use the validation results to inform further model development or analysis. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It parses the fitted parameters from a JSON string.
    - It loads the original experimental data from the specified path.
    - It calculates confidence intervals for the fitted parameters.
    - It computes goodness-of-fit metrics such as R-squared, RMSE, AIC, and BIC.
    - It performs residual analysis to evaluate the model's predictive accuracy.
    - It returns a JSON string containing the validation results, including confidence intervals, goodness-of-fit metrics, and convergence status. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `validate_model_fit(fitted_params='{"fitted_params": {"k1": 0.15}, "param_errors": {"k1": 0.01}, "goodness_of_fit": {"r_squared": 0.95, "rmse": 0.02, "aic": 10.5, "bic": 12.3}, "convergence": true}', experimental_data='path/to/experiment_1.csv')`,
        `validate_model_fit(fitted_params='{"fitted_params": {"k1": 0.05, "k2": 0.01}, "param_errors": {"k1": 0.005, "k2": 0.002}, "goodness_of_fit": {"r_squared": 0.90, "rmse": 0.03, "aic": 15.2, "bic": 18.4}, "convergence": true}', experimental_data='path/to/experiment_2.csv')`,
        `validate_model_fit(fitted_params='{"fitted_params": {"k1": 0.2}, "param_errors": {"k1": 0.02}, "goodness_of_fit": {"r_squared": 0.92, "rmse": 0.025, "aic": 12.8, "bic": 14.6}, "convergence": false}', experimental_data='path/to/experiment_3.csv')`,
        `validate_model_fit(fitted_params='{"fitted_params": {"k1": 0.1, "k2": 0.05, "k3": 0.01}, "param_errors": {"k1": 0.01, "k2": 0.005, "k3": 0.001}, "goodness_of_fit": {"r_squared": 0.88, "rmse": 0.04, "aic": 20.1, "bic": 22.5}, "convergence": true}', experimental_data='path/to/experiment_4.csv')`,
        `validate_model_fit(fitted_params='{"fitted_params": {"k1": 0.1, "k2": 0.1}, "param_errors": {"k1": 0.01, "k2": 0.01}, "goodness_of_fit": {"r_squared": 0.93, "rmse": 0.025, "aic": 11.5, "bic": 13.7}, "convergence": true}', experimental_data='path/to/experiment_5.csv')`,
    ]
    [/SYNTACTICAL]

    Args:
        fitted_params (str):
            [BRIEF] Fitted parameters JSON string. [/BRIEF]
            [DETAILED] The fitted parameters should be provided as a JSON string, typically obtained from the `fit_kinetic_parameters` tool. It includes the fitted parameters, their errors, goodness-of-fit metrics, and convergence status. [/DETAILED]
            [SYNTACTICAL] JSON string [/SYNTACTICAL]
            [EXAMPLES] '{"fitted_params": {"k1": 0.15}, "param_errors": {"k1": 0.01}, "goodness_of_fit": {"r_squared": 0.95, "rmse": 0.02, "aic": 10.5, "bic": 12.3}, "convergence": true}' [/EXAMPLES]

        experimental_data (str):
            [BRIEF] Path to experimental data CSV. [/BRIEF]
            [DETAILED] The path to the original experimental data file in CSV format. This file should contain time-series data that was used for fitting the kinetic parameters. [/DETAILED]
            [SYNTACTICAL] File path string [/SYNTACTICAL]
            [EXAMPLES] "path/to/experiment_1.csv" [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with validation results. [/BRIEF]
            [DETAILED] The returned JSON string contains the confidence intervals for the fitted parameters, goodness-of-fit metrics (R-squared, RMSE, AIC, BIC), residual analysis results, and convergence status. This information is crucial for assessing the reliability and accuracy of the fitted kinetic model. [/DETAILED]
            [EXAMPLES] '{"confidence_intervals": {"k1": [0.14, 0.16]}, "goodness_of_fit": {"r_squared": 0.95, "rmse": 0.02, "aic": 10.5, "bic": 12.3}, "residual_analysis": {"rmse": 0.02, "r_squared": 0.95}, "model_selection": {"aic": 10.5, "bic": 12.3}, "convergence_status": true}' [/EXAMPLES]

    [RAISES] Exceptions:

        RuntimeError:
            [ERROR_WHEN] If the experimental dataset cannot be loaded. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a RuntimeError if it fails to load the specified experimental dataset, possibly due to an incorrect path or missing file. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Verify that the experimental data path is correct and that the file is accessible before calling this tool. [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] If the fitted parameters or ODE system JSON strings are malformed. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if it encounters issues while parsing the fitted parameters or ODE system JSON strings. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure that the fitted parameters and ODE system strings are valid JSON before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The validation process assumes that the experimental data is of sufficient quality for reliable assessment of the model fit.
        - The tool currently does not support advanced statistical tests for model validation; it focuses on standard metrics and analyses.
        - It may not account for all sources of uncertainty in the experimental data or model predictions.
    [/LIMITATIONS]
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
def lookup_previous_works(query: str, top_k: int = 5) -> str:
    """
    [BRIEF] Search for relevant kinetic modeling research papers and information. [/BRIEF]

    [DETAILED] This tool searches through a vector database of kinetic modeling research papers to find the most relevant information based on the provided query. It uses semantic search to retrieve papers, methodologies, and findings that are most similar to the query. The tool returns detailed information about the retrieved papers including titles, abstracts, key findings, and methodologies that can inform kinetic modeling approaches. [/DETAILED]

    [PROCEDURAL] When to use this tool:
    - When you need background information about kinetic modeling approaches for specific reactions.
    - When looking for literature precedents for parameter estimation or model validation.
    - When seeking methodological guidance from published research.
    - When you want to understand what kinetic models have been used for similar systems. [/PROCEDURAL]

    [WORKFLOW_INTEGRATION] Typical workflow integration:
    1. [PREREQUISITE] Ensure you have a clear research question or topic to search for. [/PREREQUISITE]
    2. [CURRENT] Call this tool with a descriptive query to find relevant literature. [/CURRENT]
    3. [FOLLOW_UP] Use the retrieved information to inform your kinetic modeling approach, parameter estimation, or model validation. [/FOLLOW_UP] [/WORKFLOW_INTEGRATION]

    [CONTEXTUAL] How this tool works:
    - It takes a search query describing the kinetic modeling topic of interest.
    - It uses vector database search to find semantically similar research papers and methodologies.
    - It retrieves the most relevant papers based on similarity scores.
    - It returns structured information about each paper including abstracts, key findings, and methods.
    - The results can help guide kinetic modeling decisions and provide literature support. [/CONTEXTUAL]

    [SYNTACTICAL] Usage examples:
    [
        `lookup_previous_works(query='Michaelis-Menten kinetics enzyme catalysis', top_k=3)`,
        `lookup_previous_works(query='first-order reaction kinetics parameter estimation', top_k=5)`,
        `lookup_previous_works(query='Langmuir-Hinshelwood surface reaction mechanisms', top_k=4)`,
        `lookup_previous_works(query='ODE system fitting experimental data kinetics', top_k=5)`,
        `lookup_previous_works(query='reaction network analysis stoichiometry matrix', top_k=3)`,
    ]
    [/SYNTACTICAL]

    Args:
        query (str):
            [BRIEF] Search query for kinetic modeling literature. [/BRIEF]
            [DETAILED] A descriptive query that captures the kinetic modeling topic, reaction type, methodology, or specific aspect you want to research. The query should be specific enough to retrieve relevant papers but broad enough to capture related work. [/DETAILED]
            [SYNTACTICAL] Descriptive text string [/SYNTACTICAL]
            [EXAMPLES] "Michaelis-Menten kinetics enzyme catalysis", "first-order reaction kinetics parameter estimation" [/EXAMPLES]

        top_k (int):
            [BRIEF] Number of papers to retrieve. [/BRIEF]
            [DETAILED] The maximum number of relevant papers to return from the search. Higher values provide more comprehensive results but may include less relevant papers. Lower values focus on the most relevant results. [/DETAILED]
            [SYNTACTICAL] Positive integer [/SYNTACTICAL]
            [EXAMPLES] 3, 5, 10 [/EXAMPLES]

    Returns:
        str:
            [BRIEF] JSON string with retrieved literature information. [/BRIEF]
            [DETAILED] The returned JSON string contains a list of relevant papers, each with detailed information including title, abstract, authors, key findings, methodologies used, and similarity scores. This information can be used to inform kinetic modeling approaches and provide literature context for research. [/DETAILED]
            [EXAMPLES] '{"papers": [{"title": "Kinetic Analysis of...", "abstract": "This study...", "similarity_score": 0.85, "key_findings": "...", "methodology": "..."}], "query": "enzyme kinetics", "total_results": 3}' [/EXAMPLES]

    [RAISES] Exceptions:
        RuntimeError:
            [ERROR_WHEN] If the kinetic_papers vector database collection does not exist. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a RuntimeError if the vector database directory or the kinetic_papers collection cannot be found. This typically occurs when the database has not been created yet. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Ensure the kinetic_papers vector database has been created and is accessible in the vector_databases directory before calling this tool. [/ERROR_RECOVERY]

        ValueError:
            [ERROR_WHEN] If the query is empty or top_k is not a positive integer. [/ERROR_WHEN]
            [ERROR_DETAILS] The function will raise a ValueError if the query string is empty or if top_k is not a positive integer value. [/ERROR_DETAILS]
            [ERROR_RECOVERY] Provide a non-empty query string and ensure top_k is a positive integer before calling this tool. [/ERROR_RECOVERY]
    [/RAISES]

    [LIMITATIONS] Known Limitations:
        - The quality of results depends on the comprehensiveness and quality of the kinetic_papers database.
        - Search results are based on semantic similarity and may not capture all relevant methodological details.
        - The tool does not verify the scientific accuracy or current validity of the retrieved papers.
        - Results may be biased toward the types of papers included in the vector database.
    [/LIMITATIONS]
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Search the kinetic_papers vector database
        results = vector_database_search(
            query=query.strip(),
            collection_name="kinetic_papers",
            path="vector_databases",
            top_k=top_k,
        )

        # Format the results for kinetic modeling context
        formatted_papers = []
        for result in results:
            paper_info = {
                "id": result["id"],
                "title": result["metadata"].get("title", "Title not available"),
                "authors": result["metadata"].get("authors", "Authors not available"),
                "abstract": result["content"][:500] + "..."
                if len(result["content"]) > 500
                else result["content"],
                "full_content": result["content"],
                "similarity_score": round(result["similarity_score"], 3),
                "methodology": result["metadata"].get(
                    "methodology", "Methodology not specified"
                ),
                "key_findings": result["metadata"].get(
                    "key_findings", "Key findings not specified"
                ),
                "reaction_type": result["metadata"].get(
                    "reaction_type", "Reaction type not specified"
                ),
                "kinetic_model": result["metadata"].get(
                    "kinetic_model", "Kinetic model not specified"
                ),
            }
            formatted_papers.append(paper_info)

        response = {
            "query": query,
            "total_results": len(formatted_papers),
            "papers": formatted_papers,
            "search_metadata": {
                "collection": "kinetic_papers",
                "top_k_requested": top_k,
                "database_path": "vector_databases",
            },
        }

        return json.dumps(response, indent=2)

    except RuntimeError as e:
        # Handle database-related errors
        return json.dumps(
            {
                "error": "Database access failed",
                "message": str(e),
                "query": query,
                "success": False,
                "suggestion": "Ensure the kinetic_papers vector database has been created in the vector_databases directory",
            }
        )

    except ValueError as e:
        # Handle input validation errors
        return json.dumps(
            {
                "error": "Invalid input parameters",
                "message": str(e),
                "query": query,
                "success": False,
            }
        )

    except Exception as e:
        # Handle any other unexpected errors
        return json.dumps(
            {
                "error": "Literature search failed",
                "message": str(e),
                "query": query,
                "success": False,
            }
        )


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
