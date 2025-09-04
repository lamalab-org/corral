import ast
from types import SimpleNamespace
from typing import Any

import numpy as np
from datasets import get_dataset_infos, load_dataset
from kinetic_modeling.utils.ode_generator import solve_ode_system
from kinetic_modeling.utils.reaction_parser import parse_reactions
from loguru import logger
from scipy import stats
from scipy.optimize import differential_evolution, dual_annealing, minimize


class FittingModel:
    """
    Model class for organizing kinetic fitting with optimization methods.

    Args:
        reaction_network (list): List of reaction strings
    """

    def __init__(self, reaction_network: list[str]):
        self.reaction_network = reaction_network
        self.fixed_rate_constants: dict[str, float] = {}
        self.rate_constants_to_optimize: dict[str, tuple[float, float]] = {}
        self.data_to_be_fitted: dict[str, Any] = {}
        self.initial_conditions: dict[str, float] = {}
        self.other_multipliers: dict[str, Any] = {}
        self.times: dict[str, Any] = {}
        self.experiments: list[Any] = []
        self.loss_function = None
        self.x0 = None
        self.result = None

        self.parsed_reactions, self.species = parse_reactions(self.reaction_network)

    def optimize(self, workers: int = -1, disp: bool = True):
        """Optimize using differential evolution."""
        bounds = list(self.rate_constants_to_optimize.values())

        self.result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(self,),
            workers=workers,
            disp=disp,
            updating="deferred",
            x0=self.x0,
        )
        return self.result

    def optimize_dual_annealing(self):
        """Optimize using dual annealing."""
        bounds = list(self.rate_constants_to_optimize.values())

        self.result = dual_annealing(objective_function, bounds=bounds, args=(self,))
        return self.result

    def minimize_objective(self, x0: np.ndarray, method: str = "L-BFGS-B"):
        """Minimize using scipy minimize."""
        self.result = minimize(objective_function, method=method, x0=x0, args=(self,))
        return self.result


def objective_function(
    rate_constants_to_optimize: np.ndarray,
    model: FittingModel,
    return_full: bool = False,
) -> Any:
    """
    Calculate the objective function for kinetic model optimization.

    This function evaluates the total weighted error between model predictions and experimental
    data across all experiments. It solves the ODE system for each experiment using the provided
    rate constants and compares the results to experimental data using the specified loss function.

    Args:
        rate_constants_to_optimize (np.ndarray): Array of rate constant values to be optimized
        model (FittingModel): The kinetic model object containing reaction network and experimental data
        return_full (bool | str): Controls the return format

    Returns:
        float | tuple: The total weighted error across all experiments and species
            - If False, returns total error (float)
            - If True, returns (total_error, full_output) where full_output is a dict with detailed results
            - If "All", returns (total_error, full_output, time_series) where time_series contains full time series data
    """
    rate_constants = dict(
        zip(
            model.rate_constants_to_optimize.keys(),
            rate_constants_to_optimize,
            strict=False,
        )
    )
    rate_constants.update(model.fixed_rate_constants)

    total_error = 0.0
    full_output = {}
    time_series = {}

    for experiment_entry in model.experiments:
        if isinstance(experiment_entry, tuple):
            experiment, weight = experiment_entry
        else:
            experiment, weight = experiment_entry, 1.0

        experiment_name = getattr(
            experiment, "experiment_name", f"exp_{len(full_output)}"
        )
        full_output[experiment_name] = {}

        # Resolve experimental parameters
        initial_conditions = resolve_experiment_attributes(
            model.initial_conditions, experiment
        )
        other_multipliers = resolve_experiment_attributes(
            model.other_multipliers, experiment
        )
        times = resolve_experiment_attributes(model.times, experiment)
        data_to_be_fitted = resolve_experiment_attributes(
            model.data_to_be_fitted, experiment
        )

        # Solve the ODE system
        model_result = solve_ode_system(
            model.parsed_reactions,
            model.species,
            rate_constants,
            initial_conditions,
            times.get("times", times),
            other_multipliers,
        )
        time_series[experiment_name] = model_result

        # Calculate the error between the model and the data for each species
        experiment_error = 0.0

        for species, data in data_to_be_fitted.items():
            idx = model.species.index(species)
            model_data = model_result[:, idx]

            if model.loss_function:
                error, model_data_transformed = model.loss_function(
                    model_data, data, times=times.get("times", times)
                )
            else:
                error, model_data_transformed = square_loss_time_series(
                    model_data, data
                )

            full_output[experiment_name][species] = model_data_transformed
            experiment_error += error

        total_error += experiment_error * weight

    if return_full is True:
        return total_error, full_output
    elif return_full == "All":
        return total_error, full_output, time_series
    else:
        return total_error


def resolve_experiment_attributes(template_dict: dict, experiment: Any) -> dict:
    """
    Resolve attribute paths in a dictionary to actual values from an experiment.

    Args:
        template_dict (dict): Dictionary with string attribute paths as values
        experiment (object): Experiment instance containing the attributes to look up

    Returns:
        dict: New dictionary with resolved attribute values
    """
    result_dict = {}

    for key, value in template_dict.items():
        if isinstance(value, dict) and "function" not in value:
            result_dict[key] = resolve_experiment_attributes(value, experiment)
        elif isinstance(value, str):
            path_components = value.split(".")
            current_obj = experiment

            for component in path_components:
                if hasattr(current_obj, component):
                    current_obj = getattr(current_obj, component)
                else:
                    raise AttributeError(
                        f"'{type(current_obj).__name__}' object has no attribute '{component}'"
                    )

            result_dict[key] = current_obj
        else:
            result_dict[key] = value

    return result_dict


def square_loss_time_series(
    model_data: np.ndarray, experimental_data: dict | np.ndarray, **kwargs
) -> tuple[float, np.ndarray]:
    """
    Calculate the square loss between model data and experimental data for time series fitting.

    Args:
        model_data (ndarray): The model data to be compared against experimental data
        experimental_data (dict | ndarray): The experimental data to compare with
        **kwargs: Additional keyword arguments

    Returns:
        tuple (error: float, model_data: ndarray): The sum of squared differences and model data
    """
    logger.debug("Calculating square loss for time series.")
    logger.debug(f"Model data shape: {model_data.shape}")
    logger.debug(f"Experimental data shape: {experimental_data}")
    logger.debug(f"Additional kwargs: {kwargs}")
    if isinstance(experimental_data, dict):
        exp_y = np.array(experimental_data["y"])
    else:
        exp_y = np.array(experimental_data)

    model_array = np.array(model_data)
    error = np.sum((model_array - exp_y) ** 2)

    return error, model_data


def fit_parameters(
    ode_code: str, exp_data: dict, initial_params: dict, bounds: dict
) -> dict:
    """
    Unified interface for parameter fitting.

    Args:
        ode_code (str): The ODE code to fit.
        exp_data (dict): The experimental data to fit against, with keys:
            - "times": array of time points
            - "data": dict mapping species names to concentration data
            - "initial_conditions": dict mapping species to initial concentrations
        initial_params (dict): The initial parameter guesses.
        bounds (dict): The bounds for the parameters.
    """
    try:
        # Parse reaction network from ode_code if it's a list of reaction strings
        if isinstance(ode_code, list):
            reaction_network = ode_code
        else:
            # Assume ode_code contains reaction network in some parseable format
            # For now, try to safely evaluate it as a Python list using ast.literal_eval
            try:
                reaction_network = ast.literal_eval(ode_code)
                # Ensure the result is a list
                if not isinstance(reaction_network, list):
                    raise ValueError(
                        "ode_code must evaluate to a list of reaction strings"
                    )
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    "ode_code must be a list of reaction strings or a string representation of a list"
                ) from e

        # Create and configure the fitting model
        model = FittingModel(reaction_network)

        # Set up rate constants to optimize
        bounds_list = []
        for param_name in initial_params:
            if param_name in bounds:
                model.rate_constants_to_optimize[param_name] = bounds[param_name]
                bounds_list.append(bounds[param_name])
            else:
                # Default bounds if not specified
                default_bound = (0.0, 10.0)
                model.rate_constants_to_optimize[param_name] = default_bound
                bounds_list.append(default_bound)

        # Set up initial conditions
        model.initial_conditions = exp_data.get("initial_conditions", {})

        # Set up experimental data structure
        times = exp_data["times"]
        data_dict = exp_data["data"]

        # Create a simple experiment-like object
        experiment = SimpleNamespace()
        experiment.experiment_name = "experiment_1"
        experiment.times = times

        # Set up data to be fitted - reference experiment attributes
        model.data_to_be_fitted = {}
        for species_name, species_data in data_dict.items():
            if isinstance(species_data, dict):
                model.data_to_be_fitted[species_name] = species_data
            else:
                model.data_to_be_fitted[species_name] = {"y": species_data}

        # Set up times - reference experiment attributes
        model.times = {"times": "times"}

        # Add experiment attributes for resolver
        for species_name in model.data_to_be_fitted:
            setattr(experiment, species_name, model.data_to_be_fitted[species_name])

        # Add experiment to model
        model.experiments = [experiment]

        # Set default loss function if not specified
        if model.loss_function is None:
            model.loss_function = square_loss_time_series

        # Set initial guess
        model.x0 = list(initial_params.values())

        # Run optimization with simpler approach
        try:
            result = model.optimize(workers=1, disp=False)
        except Exception:
            # Fallback to dual annealing if differential evolution fails
            try:
                result = model.optimize_dual_annealing()
            except Exception:
                # Final fallback to simple minimize
                result = model.minimize_objective(model.x0)

        if not result.success:
            return {
                "params": initial_params,
                "errors": dict.fromkeys(initial_params, np.inf),
                "r_squared": 0.0,
                "rmse": np.inf,
                "aic": np.inf,
                "bic": np.inf,
                "success": False,
                "message": f"Optimization failed: {result.message}",
            }

        # Extract fitted parameters
        fitted_params = dict(
            zip(model.rate_constants_to_optimize.keys(), result.x, strict=True)
        )

        # Get model predictions for goodness-of-fit calculation
        error, model_results = objective_function(result.x, model, return_full=True)

        # Calculate goodness-of-fit metrics for each species and aggregate
        total_r_squared = 0.0
        total_rmse = 0.0
        n_total_points = 0
        n_params = len(fitted_params)

        for species_name, species_data in data_dict.items():
            if species_name in model_results["experiment_1"]:
                model_pred = np.array(model_results["experiment_1"][species_name])
                if isinstance(species_data, dict):
                    exp_data_array = np.array(species_data["y"])
                else:
                    exp_data_array = np.array(species_data)

                metrics = calculate_goodness_of_fit(
                    model_pred, exp_data_array, n_params
                )
                total_r_squared += metrics["r_squared"] * len(exp_data_array)
                total_rmse += metrics["rmse"] ** 2 * len(exp_data_array)
                n_total_points += len(exp_data_array)

        # Average metrics across all species and data points
        avg_r_squared = total_r_squared / n_total_points if n_total_points > 0 else 0.0
        avg_rmse = (
            np.sqrt(total_rmse / n_total_points) if n_total_points > 0 else np.inf
        )

        # Calculate AIC and BIC based on total error
        log_likelihood = (
            -0.5 * n_total_points * np.log(2 * np.pi * error / n_total_points)
            - 0.5 * error / (error / n_total_points)
            if error > 0
            else 0
        )
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_total_points) - 2 * log_likelihood

        # Calculate parameter errors (approximate using optimization result)
        param_errors = {}
        if hasattr(result, "hess_inv") and result.hess_inv is not None:
            # Use Hessian inverse for error estimation
            diagonal_elements = (
                np.diag(result.hess_inv)
                if hasattr(result.hess_inv, "diagonal")
                else result.hess_inv
            )
            param_errors = {
                param: np.sqrt(abs(diagonal_elements[i]))
                if i < len(diagonal_elements)
                else 0.01
                for i, param in enumerate(fitted_params.keys())
            }
        else:
            # Default error estimate (10% of parameter value)
            param_errors = {
                param: abs(value) * 0.1 for param, value in fitted_params.items()
            }

        return {
            "params": fitted_params,
            "errors": param_errors,
            "r_squared": avg_r_squared,
            "rmse": avg_rmse,
            "aic": aic,
            "bic": bic,
            "success": True,
            "optimization_result": result,
            "total_error": error,
        }

    except Exception as e:
        return {
            "params": initial_params,
            "errors": dict.fromkeys(initial_params, np.inf),
            "r_squared": 0.0,
            "rmse": np.inf,
            "aic": np.inf,
            "bic": np.inf,
            "success": False,
            "message": f"Parameter fitting failed: {e!s}",
        }


def calculate_confidence_intervals(
    fitted_params: dict,
    experimental_data: dict,
    alpha: float = 0.05,
    param_errors: dict | None = None,
) -> dict:
    """
    Calculate confidence intervals for fitted parameters using statistical methods.

    Args:
        fitted_params (dict): Fitted parameter values
        experimental_data (dict): Experimental data used for fitting
        alpha (float): Significance level (default 0.05 for 95% confidence)
        param_errors (dict, optional): Parameter standard errors from optimization

    Returns:
        dict: Confidence intervals for each parameter with statistical validity
    """
    confidence_intervals = {}

    # Extract data information to determine degrees of freedom
    total_data_points = 0
    if isinstance(experimental_data, dict):
        # Count total data points across all species/experiments
        for value in experimental_data.values():
            if isinstance(value, dict):
                if "y" in value:
                    # Handle direct data format
                    total_data_points += len(value["y"])
                elif "times" in value:
                    total_data_points += len(value["times"])
            elif hasattr(value, "__len__"):
                total_data_points += len(value)

    # If we can't determine data points, use a conservative estimate
    if total_data_points == 0:
        total_data_points = 50  # Conservative default

    # Calculate degrees of freedom (data points - number of parameters)
    n_params = len(fitted_params)
    degrees_of_freedom = max(total_data_points - n_params, 1)

    # Get t-critical value for the specified confidence level
    t_critical = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)

    for param, value in fitted_params.items():
        # Use provided parameter errors if available
        if param_errors and param in param_errors:
            std_error = param_errors[param]
        else:
            # Estimate standard error using asymptotic approximation
            # This is a rough estimate based on parameter magnitude
            std_error = abs(value) * 0.1 if abs(value) > 1e-10 else 1e-3

        # Calculate margin of error
        margin_error = t_critical * std_error

        # Calculate confidence bounds
        lower_bound = value - margin_error
        upper_bound = value + margin_error

        # For rate constants, ensure lower bound is non-negative
        if param.startswith("k") and lower_bound < 0:
            lower_bound = max(0.0, value - abs(value) * 0.5)

        confidence_intervals[param] = {
            "lower": lower_bound,
            "upper": upper_bound,
            "std_error": std_error,
            "margin_error": margin_error,
            "confidence_level": (1 - alpha) * 100,
            "degrees_of_freedom": degrees_of_freedom,
            "t_critical": t_critical,
        }

    return confidence_intervals


def load_experimental_data(config: str) -> dict:
    """
    Load experimental data from dataset.

    Args:
        config (str): Configuration name for the dataset

    Returns:
        dict: loaded experimental data (train split only)
    """
    try:
        experimental_data = load_dataset(
            "jablonkagroup/kinetic-experiments", name=config
        )
        # The dataset always has only the train split, so extract it directly
        return experimental_data["train"]
    except Exception as e:
        # Try to get available configurations to provide helpful error message
        try:
            available_configs = list(
                get_dataset_infos("jablonkagroup/kinetic-experiments").keys()
            )
            error_msg = (
                f"Failed to load data for the experiment '{config}'. "
                f"Available experiments are: {available_configs}. "
                f"Please use one of these experiment names. "
            )
        except Exception:
            error_msg = (
                f"Failed to load data for the experiment '{config}'. "
                f"Ensure the experiment name is correct. "
            )

        raise RuntimeError(error_msg) from e


def calculate_goodness_of_fit(
    model_data: np.ndarray, experimental_data: np.ndarray, n_params: int
) -> dict:
    """
    Calculate comprehensive goodness-of-fit metrics.

    Args:
        model_data (np.ndarray): Model predictions
        experimental_data (np.ndarray): Experimental observations
        n_params (int): Number of fitted parameters

    Returns:
        dict: Dictionary containing R², RMSE, AIC, BIC
    """
    model_data = np.array(model_data)
    exp_data = np.array(experimental_data)

    # Calculate residuals
    residuals = exp_data - model_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((exp_data - np.mean(exp_data)) ** 2)

    # R-squared
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # AIC and BIC
    n_data = len(exp_data)
    log_likelihood = -0.5 * n_data * np.log(
        2 * np.pi * ss_res / n_data
    ) - 0.5 * ss_res / (ss_res / n_data)
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n_data) - 2 * log_likelihood

    return {
        "r_squared": r_squared,
        "rmse": rmse,
        "aic": aic,
        "bic": bic,
        "log_likelihood": log_likelihood,
    }
