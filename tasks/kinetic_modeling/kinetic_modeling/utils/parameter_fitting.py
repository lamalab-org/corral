import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def fit_parameters(
    ode_code: str, exp_data: pd.DataFrame, initial_params: dict, bounds: dict
):
    # Dynamically exec ode_code
    local_dict = {}
    exec(ode_code, {}, local_dict)
    ode_system = local_dict["ode_system"]

    species = [c for c in exp_data.columns if c != "time"]

    def simulate(params_dict):
        y0 = [exp_data[s].iloc[0] for s in species]
        tspan = exp_data["time"].to_numpy()
        sol = solve_ivp(
            lambda t, y: ode_system(t, y, params_dict),
            [tspan[0], tspan[-1]],
            y0,
            t_eval=tspan,
        )
        return sol.y.T

    def loss(param_vals):
        params_dict = dict(zip(initial_params.keys(), param_vals, strict=False))
        pred = simulate(params_dict)
        obs = exp_data[species].to_numpy()
        return np.sum((obs - pred) ** 2)

    x0 = list(initial_params.values())
    bnds = [tuple(bounds.get(k, (0, np.inf))) for k in initial_params]
    result = minimize(loss, x0, bounds=bnds)

    params_fitted = dict(zip(initial_params.keys(), result.x, strict=False))
    errors = {k: None for k in initial_params}  # TODO: confidence intervals

    return {
        "params": params_fitted,
        "errors": errors,
        "r_squared": None,
        "rmse": None,
        "aic": None,
        "bic": None,
        "success": result.success,
    }


def calculate_confidence_intervals(result):
    # placeholder
    return {k: (0, 0) for k in result["params"]}
