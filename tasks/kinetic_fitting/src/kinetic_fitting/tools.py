"""Tools for kinetic model fitting and analysis."""

import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set matplotlib backend environment variables before any matplotlib imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""  # Disable display

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to prevent threading issues
import matplotlib.pyplot as plt

# Additional safety: explicitly configure pyplot for non-interactive use
plt.ioff()  # Turn off interactive mode

# Import threading to help with matplotlib threading issues
import threading

# Create a lock for matplotlib operations
_matplotlib_lock = threading.Lock()

import numpy as np
import io
import base64
import h5py
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import linregress

from corral.backend.tool import tool, Tool
from litellm import completion


def _get_persistent_output_dir() -> Path:
    """Get or create persistent output directory for plots and results."""
    # Use the task directory's parent to ensure persistence across runs
    output_dir = Path(__file__).parent.parent.parent / "persistent_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@dataclass
class Reaction:
    """Represents a chemical reaction in the kinetic network."""

    equation: str
    type: str
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, float]
    k_range: Optional[Tuple[float, float]] = None
    quantum_yield: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, rxn_dict: dict) -> "Reaction":
        """Parse reaction from dictionary representation."""
        equation = rxn_dict["equation"]

        if "->" not in equation:
            raise ValueError(
                f"Invalid equation format: '{equation}'. Must be a chemical equation with -> arrow (e.g. 'A + B -> C + D'), not a rate expression (e.g. 'k*A*B')."
            )

        left, right = equation.split("->")
        reactants = [s.strip() for s in left.split("+")]
        products = [s.strip() for s in right.split("+")]

        ignored = {"hv", "H2O", "OH", "products", "H"}
        stoich = {}

        def parse_species_with_coeff(species_str):
            """Parse '2 RuIII' -> (2, 'RuIII')"""
            parts = species_str.strip().split(" ", 1)
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


def load_experimental_data(data_path: str) -> Dict[str, Any]:
    """Load experimental data from HDF5 file."""
    import pandas as pd
    from dataclasses import dataclass, field
    from typing import Dict

    @dataclass
    class ExperimentMetadata:
        """Store experimental conditions and metadata"""

        experiment_name: str
        power_output: float
        ru_concentration: float
        oxidant_concentration: float
        buffer_concentration: float
        pH: float
        buffer_used: int
        annotations: str = ""
        color: str = "#ce1480"

    @dataclass
    class AnalysisMetadata:
        """Store analysis metadata"""

        p: np.ndarray
        max_rate: float
        max_rate_ydiff: float
        initial_state: np.ndarray
        matrix: str
        rate_constant: float
        rxn_start: int
        rxn_end: int
        residual: np.ndarray
        idx_for_fitting: int

    @dataclass
    class DataSets:
        """Store datasets"""

        data_corrected: np.ndarray

    @dataclass
    class TimeSeriesData:
        """Store time series data"""

        time_reaction: np.ndarray
        data_reaction: np.ndarray
        y_fit: np.ndarray
        baseline_y: np.ndarray
        lbc_fit_y: np.ndarray
        full_x_values: np.ndarray
        full_y_corrected: np.ndarray
        x_diff: np.ndarray
        y_diff: np.ndarray
        y_diff_smoothed: np.ndarray
        y_diff_fit: np.ndarray
        time_full: np.ndarray
        data_full: np.ndarray

    @dataclass
    class ExperimentalData:
        """Container for individual experiment's data"""

        time_series_data: TimeSeriesData
        experiment_metadata: ExperimentMetadata
        analysis_metadata: AnalysisMetadata
        datasets: DataSets

    @dataclass
    class ExperimentalDataset:
        experiments: Dict[str, "ExperimentalData"] = field(default_factory=dict)
        overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

        @classmethod
        def load_from_hdf5(cls, filename: str):
            """Load experiments from HDF5 file"""
            dataset = cls()

            try:
                dataset.overview_df = pd.read_hdf(filename, key="overview_df")
            except (KeyError, ValueError):
                dataset.overview_df = pd.DataFrame()

            with h5py.File(filename, "r") as f:
                for exp_name in f.keys():
                    if exp_name == "overview_df":  # Skip the overview_df group
                        continue
                    try:
                        # Load experimental data
                        time_series_dict = dict(f[f"{exp_name}/time_series_data"].attrs)
                        time_series_data = TimeSeriesData(**time_series_dict)

                        # Load metadata
                        exp_metadata_dict = dict(
                            f[f"{exp_name}/experiment_metadata"].attrs
                        )
                        experiment_metadata = ExperimentMetadata(**exp_metadata_dict)

                        analysis_metadata_dict = dict(
                            f[f"{exp_name}/analysis_metadata"].attrs
                        )
                        analysis_metadata = AnalysisMetadata(**analysis_metadata_dict)

                        datasets_dict = dict(f[f"{exp_name}/datasets"].attrs)
                        datasets = DataSets(**datasets_dict)

                        # Create ExperimentalData and add to dataset
                        experimental_data = ExperimentalData(
                            time_series_data,
                            experiment_metadata,
                            analysis_metadata,
                            datasets,
                        )
                        dataset.experiments[exp_name] = experimental_data

                    except Exception:
                        continue  # Skip experiments that can't be loaded

            return dataset

    try:
        # Load using proper data structure
        dataset = ExperimentalDataset.load_from_hdf5(data_path)

        data = {}
        for exp_name, exp_data in dataset.experiments.items():
            # Extract time and oxygen data from real experimental data
            time = exp_data.time_series_data.time_reaction
            oxygen = exp_data.time_series_data.data_reaction

            # Use overview data for metadata (more reliable units)
            if not dataset.overview_df.empty:
                overview_row = dataset.overview_df[
                    dataset.overview_df["Experiment"] == exp_name
                ]
                if not overview_row.empty:
                    row = overview_row.iloc[0]
                    standardized_metadata = {
                        "c_Ru": row.get("c([Ru(bpy(3]Cl2) [M]", 0)
                        * 1e6,  # Convert M to µM
                        "c_S2O8": row.get("c(Na2S2O8) [M]", 0) * 1e6,  # Convert M to µM
                        "power_output": row.get("Power output [W/m^2]", 1000),
                        "pH": row.get("pH [-]", 7.0),
                        "irradiance": row.get("Power output [W/m^2]", 1000),
                    }
                else:
                    # Fallback to experimental metadata (but may have unit issues)
                    standardized_metadata = {
                        "c_Ru": exp_data.experiment_metadata.ru_concentration,
                        "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                        "power_output": exp_data.experiment_metadata.power_output,
                        "pH": exp_data.experiment_metadata.pH,
                        "irradiance": exp_data.experiment_metadata.power_output,
                    }
            else:
                # Fallback to experimental metadata
                standardized_metadata = {
                    "c_Ru": exp_data.experiment_metadata.ru_concentration,
                    "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                    "power_output": exp_data.experiment_metadata.power_output,
                    "pH": exp_data.experiment_metadata.pH,
                    "irradiance": exp_data.experiment_metadata.power_output,
                }

            data[exp_name] = {
                "time": time.tolist() if hasattr(time, "tolist") else time,
                "oxygen": oxygen.tolist() if hasattr(oxygen, "tolist") else oxygen,
                "metadata": standardized_metadata,
            }

        if not data:
            # Fallback: Use overview data but with proper real time-series data if available
            if not dataset.overview_df.empty:
                print(
                    f"Experiments have empty metadata, trying overview data with real time-series if available"
                )

                # Try to get real time-series data by looking for the data in attributes
                with h5py.File(data_path, "r") as f:
                    for _, row in dataset.overview_df.iterrows():
                        exp_name = row["Experiment"]

                        # Extract metadata with proper unit conversions from overview
                        standardized_metadata = {
                            "c_Ru": row.get("c([Ru(bpy(3]Cl2) [M]", 0)
                            * 1e6,  # Convert M to µM
                            "c_S2O8": row.get("c(Na2S2O8) [M]", 0)
                            * 1e6,  # Convert M to µM
                            "power_output": row.get("Power output [W/m^2]", 1000),
                            "pH": row.get("pH [-]", 7.0),
                            "irradiance": row.get("Power output [W/m^2]", 1000),
                        }

                        # Try to get real time-series data from the HDF5 file
                        time = None
                        oxygen = None

                        if exp_name in f:
                            exp_group = f[exp_name]
                            # Try to find time-series data in attributes
                            if "time_series_data" in exp_group:
                                ts_group = exp_group["time_series_data"]
                                if (
                                    hasattr(ts_group, "attrs")
                                    and "time_reaction" in ts_group.attrs
                                ):
                                    time = ts_group.attrs["time_reaction"]
                                    oxygen = ts_group.attrs["data_reaction"]

                        # If we couldn't find real time-series, create minimal placeholder
                        if time is None or oxygen is None:
                            time = np.linspace(0, 300, 10)  # Minimal placeholder
                            oxygen = np.ones_like(time) * 0.1  # Minimal placeholder

                        data[exp_name] = {
                            "time": time.tolist() if hasattr(time, "tolist") else time,
                            "oxygen": oxygen.tolist()
                            if hasattr(oxygen, "tolist")
                            else oxygen,
                            "metadata": standardized_metadata,
                        }

                print(
                    f"Loaded {len(data)} experiments from overview dataframe with metadata"
                )
                return data
            raise ValueError("No experimental data could be loaded")

        print(
            f"Loaded {len(data)} experiments with real time-series data from HDF5 file"
        )
        return data

    except Exception as e:
        raise ValueError(f"Could not load experimental data: {e}")


def load_reaction_network(network_path: str) -> Dict[str, Any]:
    """Load reaction network from JSON file."""
    with open(network_path) as f:
        return json.load(f)


def save_reaction_network(network_path: str, network: Dict[str, Any]) -> None:
    """Save reaction network to JSON file."""
    with open(network_path, "w") as f:
        json.dump(network, f, indent=2)

    # Also save to persistent directory
    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"
    with open(persistent_network_path, "w") as f:
        json.dump(network, f, indent=2)


def create_ode_system(
    reaction_network: dict,
    experimental_conditions: dict,
) -> Tuple[Any, List[str], Dict[str, int]]:
    """Convert reaction network to ODE system for integration."""
    reactions = [Reaction.from_dict(r) for r in reaction_network["reactions"]]

    all_species = set()
    for rxn in reactions:
        all_species.update(rxn.stoichiometry.keys())

    species_list = sorted(all_species)
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    def ode_func(y: np.ndarray, t: float, params: dict, conditions: dict) -> np.ndarray:
        dydt = np.zeros_like(y)

        for i, rxn in enumerate(reactions):
            if rxn.type == "light":
                quantum_yield = params[f"qy_{i}"]
                irradiance = conditions.get("irradiance", 1000)
                photon_flux_factor = irradiance / 1000

                absorber = None
                for reactant in rxn.reactants:
                    if reactant != "hv" and reactant in species_idx:
                        absorber = reactant
                        break

                rate = (
                    quantum_yield * photon_flux_factor * y[species_idx[absorber]]
                    if absorber
                    else 0
                )

            else:
                k = params[f"k_{i}"]
                rate = k

                for reactant in rxn.reactants:
                    if reactant in species_idx:
                        if rxn.equation.startswith(f"2 {reactant}"):
                            rate *= y[species_idx[reactant]] ** 2
                        else:
                            rate *= y[species_idx[reactant]]
                    elif reactant in ["H2O", "H+"]:
                        # Water and H+ are in large excess, treat as constants
                        rate *= 1.0  # No concentration dependence

            for species, coeff in rxn.stoichiometry.items():
                if species in species_idx:
                    dydt[species_idx[species]] += coeff * rate

        return dydt

    return ode_func, species_list, species_idx


def fit_reaction_network(
    time_exp: np.ndarray,
    oxygen_exp: np.ndarray,
    reaction_network: dict,
    experimental_conditions: dict,
    maxiter: int = 100,  # Increased for better optimization
    reference_params: dict = None,  # Optional reference parameters to seed optimization
) -> dict:
    """Fit reaction network parameters to experimental oxygen evolution data."""
    ode_func, species_list, species_idx = create_ode_system(
        reaction_network, experimental_conditions
    )

    y0 = np.zeros(len(species_list))
    if "RuII" in species_idx:
        y0[species_idx["RuII"]] = experimental_conditions.get("c_Ru", 10)
    if "S2O8" in species_idx:
        y0[species_idx["S2O8"]] = experimental_conditions.get("c_S2O8", 6000)

    bounds = []
    param_names = []
    reactions = [Reaction.from_dict(r) for r in reaction_network["reactions"]]

    for i, rxn in enumerate(reactions):
        if rxn.type == "light":
            bounds.append(rxn.quantum_yield or (0.0, 1.0))
            param_names.append(f"qy_{i}")
        else:
            k_range = rxn.k_range or (1e-3, 1e10)
            bounds.append((np.log10(k_range[0]), np.log10(k_range[1])))
            param_names.append(f"k_{i}")

    def objective(params_log: np.ndarray) -> float:
        params_dict = {}
        for name, val in zip(param_names, params_log):
            if name.startswith("qy_"):
                params_dict[name] = val
            else:
                params_dict[name] = 10**val

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = odeint(
                    ode_func,
                    y0,
                    time_exp,
                    args=(params_dict, experimental_conditions),
                    rtol=1e-6,
                    atol=1e-8,
                )

            o2_pred = (
                y_pred[:, species_idx["O2"]]
                if "O2" in species_idx
                else np.zeros_like(time_exp)
            )

            # Multiple checks for poor solutions
            o2_range = np.max(o2_pred) - np.min(o2_pred)
            o2_max = np.max(o2_pred)
            o2_mean = np.mean(o2_pred)
            exp_range = np.max(oxygen_exp) - np.min(oxygen_exp)
            exp_max = np.max(oxygen_exp)

            # Very harsh penalties for different types of bad solutions
            penalty_factor = 0

            # Penalty 1: Essentially zero solutions (all values near zero)
            if o2_max < 0.01 * exp_max or o2_max < 0.05:
                penalty_factor += 1e10  # Massive penalty for zero solutions

            # Penalty 2: Flat line solutions (no dynamics)
            elif o2_range < 0.02 * exp_range or o2_range < 0.2:
                penalty_factor += 1e9  # Huge penalty for flat solutions

            # Penalty 3: Solutions that don't reach reasonable magnitude
            elif o2_max < 0.1 * exp_max:
                penalty_factor += 1e8  # Large penalty for too-small solutions

            # Penalty 4: Solutions with wrong trend (decreasing when should increase)
            if len(o2_pred) > 2:
                if o2_pred[-1] < o2_pred[1]:  # Final < early value
                    penalty_factor += 1e7

            if penalty_factor > 0:
                rss = penalty_factor
            else:
                # Normal RSS calculation only for reasonable solutions
                rss = np.sum((oxygen_exp - o2_pred) ** 2)

                # Add smaller penalties for fine-tuning
                if np.any(y_pred < -1e-6):
                    rss += 1e5  # Penalty for negative concentrations

                # Bonus for solutions that show proper growth curves
                if o2_pred[-1] > 2 * o2_pred[0] and o2_range > 0.5 * exp_range:
                    rss *= 0.9  # Small bonus for good growth behavior

            return rss

        except Exception:
            return 1e10

    try:
        # Optionally seed with reference parameters if provided
        init_population = None
        if reference_params is not None:
            # Convert reference parameters to optimization space and create seed
            seed_point = []
            for name in param_names:
                if name in reference_params:
                    if name.startswith("qy_"):
                        seed_point.append(reference_params[name])
                    else:
                        seed_point.append(np.log10(reference_params[name]))
                else:
                    # Use middle of bounds for unknown parameters
                    idx = param_names.index(name)
                    mid_val = (bounds[idx][0] + bounds[idx][1]) / 2
                    seed_point.append(mid_val)

            # Create population with seeded first individual
            init_population = np.random.random((15, len(bounds)))
            for i, (low, high) in enumerate(bounds):
                init_population[:, i] = low + init_population[:, i] * (high - low)

            # Replace first individual with seed if valid
            if len(seed_point) == len(bounds):
                init_population[0] = seed_point

        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=15,  # Increased population size
            seed=42,
            workers=1,
            updating="deferred",
            atol=1e-6,
            tol=0.01,  # Tightened tolerances
            init=init_population if reference_params is not None else "latinhypercube",
        )

        params_dict = {}
        for name, val in zip(param_names, result.x):
            if name.startswith("qy_"):
                params_dict[name] = val
            else:
                params_dict[name] = 10**val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_final = odeint(
                ode_func,
                y0,
                time_exp,
                args=(params_dict, experimental_conditions),
                rtol=1e-6,
                atol=1e-8,
            )

        o2_pred = (
            y_final[:, species_idx["O2"]]
            if "O2" in species_idx
            else np.zeros_like(time_exp)
        )

        # Check if this is a reasonable solution before returning success
        o2_range = np.max(o2_pred) - np.min(o2_pred)
        o2_max = np.max(o2_pred)
        exp_range = np.max(oxygen_exp) - np.min(oxygen_exp)
        exp_max = np.max(oxygen_exp)

        # Determine if solution is acceptable
        solution_acceptable = True
        failure_reason = ""

        if o2_max < 0.01 * exp_max or o2_max < 0.05:
            solution_acceptable = False
            failure_reason = "Solution essentially zero"
        elif o2_range < 0.02 * exp_range or o2_range < 0.2:
            solution_acceptable = False
            failure_reason = "Solution is flat line"
        elif o2_max < 0.1 * exp_max:
            solution_acceptable = False
            failure_reason = "Solution magnitude too small"
        elif result.fun > 1e7:  # If RSS is very high (penalty was applied)
            solution_acceptable = False
            failure_reason = "High RSS indicates penalty was applied"

        # Calculate R² only for reasonable solutions
        if solution_acceptable:
            residuals = oxygen_exp - o2_pred
            r2 = 1 - np.sum(residuals**2) / np.sum(
                (oxygen_exp - np.mean(oxygen_exp)) ** 2
            )

            # Additional R² check
            if r2 < -1.0:  # R² shouldn't be extremely negative
                solution_acceptable = False
                failure_reason = f"R² too negative: {r2:.3f}"
        else:
            r2 = -999.0  # Clearly bad R²

        # Save detailed parameter information to persistent directory
        persistent_dir = _get_persistent_output_dir()

        # Add network diagnosis
        network_diagnosis = _diagnose_network_issues(
            reaction_network, experimental_conditions
        )

        param_info = {
            "params": params_dict,
            "rss": result.fun,
            "r2": r2,
            "success": result.success and solution_acceptable,
            "failure_reason": failure_reason if not solution_acceptable else "",
            "o2_range": o2_range,
            "o2_max": o2_max,
            "exp_conditions": experimental_conditions,
            "species_list": species_list,
            "species_idx": species_idx,
            "optimization_result": {
                "converged": result.success,
                "nfev": getattr(result, "nfev", "N/A"),
                "nit": getattr(result, "nit", "N/A"),
                "message": getattr(result, "message", "N/A"),
            },
            "bounds_used": bounds,
            "param_names": param_names,
            "network_diagnosis": network_diagnosis,
            "reaction_network": reaction_network,
        }

        # Save to timestamped file
        import time

        timestamp = int(time.time())
        param_file = persistent_dir / f"fit_params_{timestamp}.json"
        with open(param_file, "w") as f:
            json.dump(param_info, f, indent=2, default=str)

        return {
            "params": params_dict,
            "rss": result.fun,
            "r2": r2,
            "y_pred": o2_pred.tolist(),
            "success": result.success and solution_acceptable,
            "species_list": species_list,
            "full_solution": y_final.tolist(),
            "species_idx": species_idx,
            "failure_reason": failure_reason if not solution_acceptable else "",
            "o2_range": o2_range,
            "o2_max": o2_max,
            "param_file": str(param_file),
        }

    except Exception as e:
        return {
            "params": {},
            "rss": 1e10,
            "r2": -999.0,
            "y_pred": np.zeros_like(time_exp).tolist(),
            "success": False,
            "species_list": species_list,
            "error": str(e),
            "failure_reason": f"Exception during fitting: {str(e)}",
            "o2_range": 0.0,
            "o2_max": 0.0,
        }


def _score_ru_trend(rates_vs_ru: dict) -> float:
    """Score Ru concentration trend (should increase then decrease)."""
    if len(rates_vs_ru) < 3:
        return 0.5

    concentrations = np.array(sorted(rates_vs_ru.keys()))
    rates = np.array([np.mean(rates_vs_ru[c]) for c in concentrations])
    max_idx = np.argmax(rates)

    position_score = 0.3 if max_idx in [0, len(rates) - 1] else 1.0
    increase_score = (
        float(np.all(np.diff(rates[: max_idx + 1]) >= 0)) if max_idx > 0 else 0.5
    )
    decrease_score = (
        float(np.all(np.diff(rates[max_idx:]) <= 0))
        if max_idx < len(rates) - 1
        else 0.5
    )

    return (position_score + increase_score + decrease_score) / 3


def _score_s2o8_trend(rates_vs_s2o8: dict) -> float:
    """Score S2O8 trend (monotonic increase with saturation)."""
    if len(rates_vs_s2o8) < 3:
        return 0.5

    concentrations = np.array(sorted(rates_vs_s2o8.keys()))
    rates = np.array([np.mean(rates_vs_s2o8[c]) for c in concentrations])

    monotonic_score = float(np.all(np.diff(rates) >= 0))

    if len(rates) >= 3:
        slopes = np.diff(rates)
        saturation_score = float(slopes[-1] < slopes[0])
    else:
        saturation_score = 0.5

    return (monotonic_score + saturation_score) / 2


def _score_irradiance_trend(rates_vs_irradiance: dict) -> float:
    """Score irradiance trend (should be linear)."""
    if len(rates_vs_irradiance) < 3:
        return 0.5

    concentrations = np.array(sorted(rates_vs_irradiance.keys()))
    rates = np.array([np.mean(rates_vs_irradiance[c]) for c in concentrations])

    _, _, r_value, _, _ = linregress(concentrations, rates)
    return max(0, r_value**2)


def _score_ph_trend(rates_vs_ph: dict) -> float:
    """Score pH trend (should have optimal range around pH 8-10 for Ru-based photocatalysis)."""
    if len(rates_vs_ph) < 2:
        return 0.5

    ph_values = np.array(sorted(rates_vs_ph.keys()))
    rates = np.array([np.mean(rates_vs_ph[ph]) for ph in ph_values])

    if len(ph_values) < 3:
        # For limited data, score based on whether higher pH gives higher rate
        if ph_values[-1] > ph_values[0]:
            return 0.8 if rates[-1] > rates[0] else 0.3
        return 0.5

    # Look for optimal pH in the alkaline range (8-10)
    max_rate_idx = np.argmax(rates)
    max_rate_ph = ph_values[max_rate_idx]

    # Score based on whether maximum is in optimal range
    if 8 <= max_rate_ph <= 10:
        optimal_score = 1.0
    elif 7 <= max_rate_ph <= 11:
        optimal_score = 0.7
    else:
        optimal_score = 0.3

    # Score based on general alkaline preference (higher pH > acidic pH)
    if len(ph_values[ph_values >= 7]) > 0 and len(ph_values[ph_values < 7]) > 0:
        alkaline_preference = (
            1.0
            if np.mean(rates[ph_values >= 7]) > np.mean(rates[ph_values < 7])
            else 0.5
        )
    else:
        alkaline_preference = 0.5

    return (optimal_score + alkaline_preference) / 2


def _create_fit_plot(
    time: np.ndarray,
    y_exp: np.ndarray,
    y_pred: np.ndarray,
    exp_name: str,
    metadata: dict,
    save_path: str = None,
) -> str:
    """Create fit visualization and return as base64 string.

    Args:
        time: Time points
        y_exp: Experimental data
        y_pred: Model predictions
        exp_name: Experiment name
        metadata: Experimental metadata
        save_path: Optional path to save plot file
    """
    with _matplotlib_lock:
        # Ensure matplotlib works properly in threads by forcing backend
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.ioff()

        residuals = y_exp - y_pred
        r2 = 1 - np.sum(residuals**2) / np.sum((y_exp - np.mean(y_exp)) ** 2)
        rmse = np.sqrt(np.mean(residuals**2))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

        ax1.scatter(
            time,
            y_exp,
            alpha=0.6,
            s=30,
            label="Experimental",
            color="#1f77b4",
            zorder=3,
        )
        ax1.plot(time, y_pred, linewidth=2.5, label="Model", color="#ff7f0e", zorder=2)
        ax1.set_ylabel("[O₂] (µM)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"{exp_name}\nR² = {r2:.4f}, RMSE = {rmse:.2f} µM",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend(fontsize=11, frameon=True, shadow=True)
        ax1.grid(alpha=0.3, linestyle="--")

        conditions = (
            f"[Ru] = {metadata.get('c_Ru', '?')} µM  |  "
            f"[S₂O₈²⁻] = {metadata.get('c_S2O8', '?')} µM  |  "
            f"Power = {metadata.get('irradiance', '?')} W/m²  |  "
            f"pH = {metadata.get('pH', '?')}"
        )
        ax1.text(
            0.02,
            0.98,
            conditions,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax2.scatter(time, residuals, alpha=0.6, s=20, color="#d62728", zorder=3)
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1.5, zorder=2)
        ax2.fill_between(time, residuals, 0, alpha=0.2, color="#d62728")
        ax2.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Residuals (µM)", fontsize=11)
        ax2.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()

        # Save to file if requested
        if save_path:
            plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")


def _create_phenomenological_plots(
    data: dict,
    save_dir: str = None,
) -> str:
    """Create phenomenological trend plots for [Ru], [S2O8], and irradiance."""
    with _matplotlib_lock:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.ioff()

        # Collect trend data
        trends = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}, "pH": {}}

        for exp_name, exp_data in data.items():
            meta = exp_data["metadata"]
            oxygen = np.array(exp_data["oxygen"])
            time = np.array(exp_data["time"])

            # Calculate maximum rate
            rates = np.gradient(oxygen, time)
            max_rate = np.max(rates)

            for param in ["c_Ru", "c_S2O8", "irradiance", "pH"]:
                param_val = meta.get(param)
                if param_val is not None:
                    if param_val not in trends[param]:
                        trends[param][param_val] = []
                    trends[param][param_val].append(max_rate)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # [Ru] trend plot
        if trends["c_Ru"]:
            concentrations = np.array(sorted(trends["c_Ru"].keys()))
            rates = [np.mean(trends["c_Ru"][c]) for c in concentrations]
            rates_std = [
                np.std(trends["c_Ru"][c]) if len(trends["c_Ru"][c]) > 1 else 0
                for c in concentrations
            ]

            axes[0, 0].errorbar(
                concentrations,
                rates,
                yerr=rates_std,
                marker="o",
                capsize=5,
                markersize=8,
                linewidth=2,
            )
            axes[0, 0].set_xlabel("[Ru(bpy)₃²⁺] (µM)", fontweight="bold")
            axes[0, 0].set_ylabel("Max O₂ Rate (µM/s)", fontweight="bold")
            axes[0, 0].set_title("Ru Concentration Dependence", fontweight="bold")
            axes[0, 0].grid(True, alpha=0.3)

        # [S2O8] trend plot
        if trends["c_S2O8"]:
            concentrations = np.array(sorted(trends["c_S2O8"].keys()))
            rates = [np.mean(trends["c_S2O8"][c]) for c in concentrations]
            rates_std = [
                np.std(trends["c_S2O8"][c]) if len(trends["c_S2O8"][c]) > 1 else 0
                for c in concentrations
            ]

            axes[0, 1].errorbar(
                concentrations,
                rates,
                yerr=rates_std,
                marker="s",
                capsize=5,
                markersize=8,
                linewidth=2,
                color="orange",
            )
            axes[0, 1].set_xlabel("[S₂O₈²⁻] (µM)", fontweight="bold")
            axes[0, 1].set_ylabel("Max O₂ Rate (µM/s)", fontweight="bold")
            axes[0, 1].set_title(
                "Persulfate Concentration Dependence", fontweight="bold"
            )
            axes[0, 1].grid(True, alpha=0.3)

        # Irradiance trend plot
        if trends["irradiance"]:
            irradiances = np.array(sorted(trends["irradiance"].keys()))
            rates = [np.mean(trends["irradiance"][c]) for c in irradiances]
            rates_std = [
                np.std(trends["irradiance"][c])
                if len(trends["irradiance"][c]) > 1
                else 0
                for c in irradiances
            ]

            axes[1, 0].errorbar(
                irradiances,
                rates,
                yerr=rates_std,
                marker="^",
                capsize=5,
                markersize=8,
                linewidth=2,
                color="green",
            )
            axes[1, 0].set_xlabel("Irradiance (W/m²)", fontweight="bold")
            axes[1, 0].set_ylabel("Max O₂ Rate (µM/s)", fontweight="bold")
            axes[1, 0].set_title("Irradiance Dependence", fontweight="bold")
            axes[1, 0].grid(True, alpha=0.3)

        # pH trend plot
        if trends["pH"]:
            ph_values = np.array(sorted(trends["pH"].keys()))
            rates = [np.mean(trends["pH"][ph]) for ph in ph_values]
            rates_std = [
                np.std(trends["pH"][ph]) if len(trends["pH"][ph]) > 1 else 0
                for ph in ph_values
            ]

            axes[1, 1].errorbar(
                ph_values,
                rates,
                yerr=rates_std,
                marker="D",
                capsize=5,
                markersize=8,
                linewidth=2,
                color="purple",
            )
            axes[1, 1].set_xlabel("pH", fontweight="bold")
            axes[1, 1].set_ylabel("Max O₂ Rate (µM/s)", fontweight="bold")
            axes[1, 1].set_title("pH Dependence", fontweight="bold")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_dir:
            plot_path = Path(save_dir) / "phenomenological_trends.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")

        # Also save to persistent directory
        persistent_dir = _get_persistent_output_dir()
        persistent_path = persistent_dir / "phenomenological_trends.png"
        plt.savefig(persistent_path, dpi=150, bbox_inches="tight")

        plt.close()

        return f"Phenomenological trend plots saved to {persistent_path}"


def _diagnose_network_issues(network: dict, sample_conditions: dict) -> str:
    """Diagnose potential issues with the reaction network that could cause poor fits."""
    issues = []

    reactions = network.get("reactions", [])
    if len(reactions) == 0:
        issues.append("No reactions in network")
        return "CRITICAL: " + "; ".join(issues)

    # Check for disconnected species
    all_species = set()
    products_formed = set()
    reactants_consumed = set()

    for rxn in reactions:
        try:
            reaction_obj = Reaction.from_dict(rxn)
            all_species.update(reaction_obj.stoichiometry.keys())
            reactants_consumed.update(reaction_obj.reactants)
            products_formed.update(reaction_obj.products)
        except Exception as e:
            issues.append(f"Invalid reaction: {rxn.get('equation', 'unknown')}")

    # Check if O2 can be formed
    if "O2" not in products_formed:
        issues.append("O2 is not produced by any reaction")

    # Check if there's a path from initial species to O2
    initial_species = set()
    if "RuII" in all_species:
        initial_species.add("RuII")
    if "S2O8" in all_species:
        initial_species.add("S2O8")

    if not initial_species:
        issues.append("No clear initial species (RuII, S2O8) found")

    # Check for reasonable parameter ranges
    light_reactions = [r for r in reactions if r.get("type") == "light"]
    dark_reactions = [r for r in reactions if r.get("type") == "dark"]

    if len(light_reactions) == 0:
        issues.append("No light reactions - photocatalysis requires light activation")

    if len(dark_reactions) == 0:
        issues.append("No dark reactions - need thermal steps for chemistry")

    # Check quantum yield ranges
    for rxn in light_reactions:
        qy_range = rxn.get("quantum_yield")
        if qy_range and (qy_range[0] >= qy_range[1] or qy_range[1] > 1.0):
            issues.append(
                f"Invalid quantum yield range {qy_range} in {rxn.get('equation')}"
            )

    # Check rate constant ranges
    for rxn in dark_reactions:
        k_range = rxn.get("k_range")
        if k_range and k_range[0] >= k_range[1]:
            issues.append(f"Invalid k_range {k_range} in {rxn.get('equation')}")

    if not issues:
        return "Network structure appears reasonable"

    return "ISSUES FOUND: " + "; ".join(issues)


# File-based tool functions


@tool
def describe_experimental_data(data_path: str) -> str:
    """List available experimental datasets with metadata.

    Args:
        data_path: Path to experimental data HDF5 file
    """
    data = load_experimental_data(data_path)

    lines = ["Available experimental datasets:\n"]

    for exp_name, exp_data in list(data.items())[:10]:
        meta = exp_data["metadata"]
        time = np.array(exp_data["time"])
        oxygen = np.array(exp_data["oxygen"])

        lines.extend(
            [
                f"{exp_name}:",
                f"  [Ru(bpy)3Cl2]: {meta.get('c_Ru', 'N/A')} µM",
                f"  [Na2S2O8]: {meta.get('c_S2O8', 'N/A')} µM",
                f"  Power/Irradiance: {meta.get('irradiance', 'N/A')} W/m²",
                f"  pH: {meta.get('pH', 'N/A')}",
                f"  Data points: {len(time)}",
                f"  Time range: {time.min():.1f} - {time.max():.1f} s",
                f"  O2 range: {oxygen.min():.3f} - {oxygen.max():.3f} µM\n",
            ]
        )

    if len(data) > 10:
        lines.append(f"... and {len(data) - 10} more experiments")

    return "\n".join(lines)


@tool
def get_current_network(network_path: str) -> str:
    """Get the current reaction network.

    Args:
        network_path: Path to reaction network JSON file
    """
    try:
        network = load_reaction_network(network_path)
    except FileNotFoundError:
        # Create default network if it doesn't exist
        network = initialize_default_network()
        save_reaction_network(network_path, network)
    except Exception as e:
        return f"Error loading network: {e}"

    lines = [f"Current reaction network ({len(network['reactions'])} reactions):\n"]

    for i, rxn in enumerate(network["reactions"]):
        lines.append(f"{i:2d}: {rxn['equation']} ({rxn['type']})")
        if rxn["type"] == "light" and "quantum_yield" in rxn:
            lines.append(f"     QY range: {rxn['quantum_yield']}")
        elif "k_range" in rxn:
            lines.append(f"     k range: {rxn['k_range']}")
        lines.append("")

    return "\n".join(lines)


@tool
def fit_single_experiment(
    data_path: str, network_path: str, results_path: str, exp_name: str
) -> str:
    """Fit current reaction network to a single experiment.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        results_path: Path to save fit results JSON file
        exp_name: Name of the experiment to fit
    """
    try:
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        exp_data = data[exp_name]

        result = fit_reaction_network(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            network,
            exp_data["metadata"],
        )

        if not result.get("success", False):
            failure_reason = result.get(
                "failure_reason", result.get("error", "Unknown error")
            )
            rss = result.get("rss", "N/A")
            r2 = result.get("r2", "N/A")
            o2_max = result.get("o2_max", "N/A")
            o2_range = result.get("o2_range", "N/A")

            return (
                f"Fit failed for {exp_name}:\n"
                f"  Reason: {failure_reason}\n"
                f"  RSS: {rss}\n"
                f"  R²: {r2}\n"
                f"  O₂ max: {o2_max}\n"
                f"  O₂ range: {o2_range}\n"
                f"  This indicates the model is producing flat/zero solutions.\n"
                f"  Consider: 1) Adding more reactions, 2) Adjusting parameter ranges, 3) Checking reaction network connectivity"
            )

        # Load existing results or create new
        try:
            with open(results_path) as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = {"experiments": {}}

        all_results["experiments"][exp_name] = {
            "rss": result["rss"],
            "r2": result["r2"],
            "params": result["params"],
            "y_pred": result["y_pred"],
        }

        # Create and save fit plot to persistent directory
        persistent_dir = _get_persistent_output_dir()
        plot_filename = persistent_dir / f"fit_plot_{exp_name}.png"
        _create_fit_plot(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            np.array(result["y_pred"]),
            exp_name,
            exp_data["metadata"],
            save_path=str(plot_filename),
        )

        # Save results both to workspace and persistent directory
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Also save to persistent directory
        persistent_results_path = persistent_dir / "fit_results.json"
        with open(persistent_results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Report parameter file location
        param_file = result.get("param_file", "N/A")

        return (
            f"Fit results for {exp_name}:\n"
            f"  R² = {result['r2']:.4f}\n"
            f"  RMSE = {np.sqrt(result['rss'] / len(result['y_pred'])):.4f} µM\n"
            f"  RSS = {result['rss']:.2f}\n"
            f"  O₂ max: {result.get('o2_max', 'N/A'):.4f} µM\n"
            f"  O₂ range: {result.get('o2_range', 'N/A'):.4f} µM\n"
            f"  Plot saved to: {plot_filename}\n"
            f"  Parameters saved to: {param_file}\n"
            f"  Persistent directory: {persistent_dir}"
        )

    except Exception as e:
        return f"Error fitting experiment: {e}"


@tool
def fit_all_experiments(data_path: str, network_path: str, results_path: str) -> str:
    """Fit current reaction network to all experiments.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        results_path: Path to save fit results JSON file
    """
    try:
        data = load_experimental_data(data_path)
        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        all_results = {"experiments": {}}
        successful = 0
        total_rss = 0.0
        plots_saved = []

        for i, (exp_name, exp_data) in enumerate(data.items()):
            result = fit_reaction_network(
                np.array(exp_data["time"]),
                np.array(exp_data["oxygen"]),
                network,
                exp_data["metadata"],
            )

            all_results["experiments"][exp_name] = {
                "rss": result["rss"],
                "r2": result["r2"],
                "success": result.get("success", False),
                "params": result.get("params", {}),
                "y_pred": result.get("y_pred", []),
            }

            # Save plots for all successful experiments
            if (
                result.get("success", False)
                and all_results["experiments"][exp_name]["r2"] is not None
            ):
                persistent_dir = _get_persistent_output_dir()
                plot_filename = persistent_dir / f"fit_plot_{exp_name}.png"
                _create_fit_plot(
                    np.array(exp_data["time"]),
                    np.array(exp_data["oxygen"]),
                    np.array(result["y_pred"]),
                    exp_name,
                    exp_data["metadata"],
                    save_path=str(plot_filename),
                )
                plots_saved.append(str(plot_filename))

            if result.get("success", False):
                successful += 1
                total_rss += result["rss"]

        avg_rss = total_rss / successful if successful > 0 else 1e10

        all_results["summary"] = {
            "total_experiments": len(data),
            "successful_fits": successful,
            "total_rss": total_rss,
            "avg_rss": avg_rss,
        }

        # Save results both to workspace and persistent directory
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Also save to persistent directory
        persistent_dir = _get_persistent_output_dir()
        persistent_results_path = persistent_dir / "fit_results.json"
        with open(persistent_results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Save a comprehensive parameter summary for debugging
        param_summary = {
            "network_used": network,
            "summary_stats": all_results["summary"],
            "successful_experiments": {},
            "failed_experiments": {},
            "timestamp": int(time.time()),
        }

        for exp_name, exp_result in all_results["experiments"].items():
            if exp_result.get("success", False):
                param_summary["successful_experiments"][exp_name] = exp_result
            else:
                param_summary["failed_experiments"][exp_name] = exp_result

        param_summary_file = persistent_dir / "parameter_summary.json"
        with open(param_summary_file, "w") as f:
            json.dump(param_summary, f, indent=2, default=str)

        plot_info = f"\n  Plots saved: {', '.join(plots_saved)}" if plots_saved else ""

        return (
            f"Fitted all experiments:\n"
            f"  Successful: {successful}/{len(data)}\n"
            f"  Total RSS: {total_rss:.2f}\n"
            f"  Average RSS: {avg_rss:.2f}\n"
            f"  All plots and results saved to: {persistent_dir}\n"
            f"  Parameter summary saved to: {param_summary_file}"
            f"{plot_info}"
        )

    except Exception as e:
        return f"Error fitting experiments: {e}"


@tool
def evaluate_phenomenological_trends(
    data_path: str, network_path: str, results_path: str
) -> str:
    """Evaluate how well model reproduces experimental trends.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        results_path: Path to results JSON file (updated with trend scores)
    """
    try:
        data = load_experimental_data(data_path)
        trends = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}, "pH": {}}
        trends_original = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}, "pH": {}}

        reaction_network = load_reaction_network(network_path)

        predictions_for_plotting = {}

        for exp_name, exp_data in list(data.items()):
            meta = exp_data["metadata"]
            oxygen = np.array(exp_data["oxygen"])
            time = np.array(exp_data["time"])
            predictions = fit_reaction_network(
                time, oxygen, reaction_network, meta, maxiter=200
            )

            y_pred = predictions["y_pred"]

            # Store the predictions in a format similar to the original data for plotting
            predictions_for_plotting[exp_name] = {
                "metadata": meta,
                "time": time,
                "oxygen": y_pred,
            }

            # Calculate maximum rate from predictions
            rates = np.gradient(y_pred, time)
            max_rate = np.max(rates)
            # Calculate the same for the original data to see how well trends match
            rates_original = np.gradient(oxygen, time)
            max_rate_original = np.max(rates_original)

            for param in ["c_Ru", "c_S2O8", "irradiance", "pH"]:
                param_val = meta.get(param)
                if param_val is not None:
                    if param_val not in trends[param]:
                        trends[param][param_val] = []
                    trends[param][param_val].append(max_rate)

            for param in ["c_Ru", "c_S2O8", "irradiance", "pH"]:
                param_val = meta.get(param)
                if param_val is not None:
                    if param_val not in trends_original[param]:
                        trends_original[param][param_val] = []
                    trends_original[param][param_val].append(max_rate_original)

        ru_score = _score_ru_trend(trends["c_Ru"])
        s2o8_score = _score_s2o8_trend(trends["c_S2O8"])
        irr_score = _score_irradiance_trend(trends["irradiance"])
        ph_score = _score_ph_trend(trends["pH"])

        overall_score = (
            0.3 * ru_score + 0.25 * s2o8_score + 0.25 * irr_score + 0.2 * ph_score
        )

        ru_score_original = _score_ru_trend(trends_original["c_Ru"])
        s2o8_score_original = _score_s2o8_trend(trends_original["c_S2O8"])
        irr_score_original = _score_irradiance_trend(trends_original["irradiance"])
        ph_score_original = _score_ph_trend(trends_original["pH"])

        # Load existing results and update with trend scores
        try:
            with open(results_path) as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = {}

        all_results["phenomenological_trends"] = {
            "ru_score": ru_score,
            "s2o8_score": s2o8_score,
            "irradiance_score": irr_score,
            "pH_score": ph_score,
            "overall_score": overall_score,
        }

        # Update best score
        if "best_phenomenological_score" not in all_results:
            all_results["best_phenomenological_score"] = 0.0

        if overall_score > all_results["best_phenomenological_score"]:
            all_results["best_phenomenological_score"] = overall_score

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Create phenomenological trend plots using the predicted data
        plot_info = _create_phenomenological_plots(predictions_for_plotting)

        return (
            f"Phenomenological trend scores:\n"
            f"  Ru concentration trend: {ru_score:.3f}\n"
            f"  S2O8 concentration trend: {s2o8_score:.3f}\n"
            f"  Irradiance trend: {irr_score:.3f}\n"
            f"  pH trend: {ph_score:.3f}\n"
            f"  Overall score: {overall_score:.3f}\n"
            f"  Best score so far: {all_results['best_phenomenological_score']:.3f}\n\n"
            f"  Trend fitting MAE between the fitted data, and the original data: "
            f"  Ru concentration trend error: {np.sum(np.abs(ru_score - ru_score_original))}"
            f"  S2O8 concentration trend error: {np.sum(np.abs(s2o8_score - s2o8_score_original))}"
            f"  Irradiance trend error: {np.sum(np.abs(irr_score - irr_score_original))}"
            f"  pH trend error: {np.sum(np.abs(ph_score - ph_score_original))}"
            f"{plot_info}"
        )

    except Exception as e:
        return f"Error evaluating trends: {e}"


@tool
def modify_reaction_network(
    network_path: str,
    add_reactions: Optional[str] = None,
    remove_reactions: Optional[str] = None,
    modify_k_ranges: Optional[str] = None,
    modify_quantum_yields: Optional[str] = None,
) -> str:
    """Modify the current reaction network.

    IMPORTANT: Use CHEMICAL EQUATIONS with -> arrows, NOT rate expressions!

    Args:
        network_path: Path to reaction network JSON file
        add_reactions: JSON string of list of reaction dictionaries to add.
            Each reaction MUST have this exact format:
            {
                "equation": "A + B -> C + D",        // Chemical equation with -> arrow (REQUIRED)
                "type": "light" or "dark",           // Reaction type (REQUIRED)
                "k_range": [min, max],               // For dark reactions only
                "quantum_yield": [min, max],         // For light reactions only (0-1)
                "description": "..."                 // Optional description
            }

            CORRECT examples:
            - Light reaction: '[{"equation": "RuII + hv -> RuII_ex", "type": "light", "quantum_yield": [0.1, 1.0]}]'
            - Dark reaction: '[{"equation": "RuII_ex + S2O8 -> RuIII + SO4", "type": "dark", "k_range": [1e6, 1e9]}]'
            - Dimerization: '[{"equation": "RuIII + RuIII -> Ru2_dim", "type": "dark", "k_range": [1e5, 1e8]}]'

            WRONG examples (DO NOT USE):
            - "k1*S0" (this is a rate expression, not a chemical equation)
            - "k1*C0" (this is a rate expression, not a chemical equation)
            - "k1*y0" (this is a rate expression, not a chemical equation)
            - "A -> B" without proper species names

        remove_reactions: JSON string of list of reaction indices to remove
        modify_k_ranges: JSON string of dict {index: [new_min, new_max]}
        modify_quantum_yields: JSON string of dict {index: [new_min, new_max]}
    """
    try:
        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        changes = []

        if remove_reactions:
            try:
                indices = json.loads(remove_reactions)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format in remove_reactions parameter: {e}"
            for idx in sorted(indices, reverse=True):
                if 0 <= idx < len(network["reactions"]):
                    removed = network["reactions"].pop(idx)
                    changes.append(f"Removed reaction {idx}: {removed['equation']}")

        if modify_k_ranges:
            try:
                k_ranges = json.loads(modify_k_ranges)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format in modify_k_ranges parameter: {e}"
            for idx, new_range in k_ranges.items():
                idx = int(idx)
                if 0 <= idx < len(network["reactions"]):
                    network["reactions"][idx]["k_range"] = tuple(new_range)
                    changes.append(f"Modified k_range for reaction {idx}")

        if modify_quantum_yields:
            try:
                qy_ranges = json.loads(modify_quantum_yields)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format in modify_quantum_yields parameter: {e}"
            for idx, new_range in qy_ranges.items():
                idx = int(idx)
                if 0 <= idx < len(network["reactions"]):
                    qy_range = (max(0, new_range[0]), min(1, new_range[1]))
                    network["reactions"][idx]["quantum_yield"] = qy_range
                    changes.append(f"Modified quantum_yield for reaction {idx}")

        if add_reactions:
            try:
                new_rxns = json.loads(add_reactions)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format in add_reactions parameter: {e}"

            for rxn in new_rxns:
                # Validate required fields
                if "equation" not in rxn:
                    return "Error: Each reaction must have an 'equation' field"
                if "type" not in rxn:
                    return "Error: Each reaction must have a 'type' field ('light' or 'dark')"

                # Validate equation format early
                if "->" not in rxn["equation"]:
                    return f"Error: Invalid equation format '{rxn['equation']}'. Must be chemical equation with -> arrow (e.g. 'RuII + S2O8 -> RuIII + SO4'), not rate expression (e.g. 'k*S*Ru')."

                network["reactions"].append(rxn)
                changes.append(f"Added reaction: {rxn['equation']}")

        # Save modified network
        save_reaction_network(network_path, network)

        return (
            f"Modified reaction network:\n"
            + "\n".join(f"  - {c}" for c in changes)
            + f"\n\nTotal reactions now: {len(network['reactions'])}"
        )

    except Exception as e:
        return f"Error modifying network: {e}"


@tool
def analyze_fit_with_vision(
    data_path: str,
    network_path: str,
    exp_name: str,
    model: str = "gpt-4o",
) -> str:
    """Use vision model to analyze fit quality from plot.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to current reaction network JSON
        exp_name: Name of experiment to analyze
        model: Vision model to use
    """
    try:
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        exp_data = data[exp_name]

        result = fit_reaction_network(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            network,
            exp_data["metadata"],
        )

        if not result.get("success"):
            return "Cannot analyze: fit failed"

        # Create plot and save to file
        plot_filename = f"fit_plot_{exp_name}.png"
        plot_b64 = _create_fit_plot(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            np.array(result["y_pred"]),
            exp_name,
            exp_data["metadata"],
            save_path=plot_filename,
        )

        # Call vision model
        response = completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{plot_b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this photocatalytic O₂ evolution fit:\n\n"
                                "1. How well does the model capture the induction period?\n"
                                "2. How well does it match the growth phase slope and curvature?\n"
                                "3. How well does it capture the plateau/maximum?\n"
                                "4. Are there systematic deviations?\n"
                                "5. Where is the fit worst (early/middle/late times)?\n"
                                "6. What specific model improvements would you suggest?\n\n"
                                "Be specific and quantitative."
                            ),
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Vision analysis failed: {e}"


@tool
def summarize_recent_fits(results_path: str) -> str:
    """Provide a summary of recent fit attempts for debugging poor fits.

    Args:
        results_path: Path to results JSON file
    """
    try:
        persistent_dir = _get_persistent_output_dir()

        # Look for recent parameter files
        param_files = list(persistent_dir.glob("fit_params_*.json"))
        param_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        if not param_files:
            return "No recent parameter files found in persistent directory"

        # Analyze the most recent fits
        summary_lines = ["Recent fit analysis:\n"]

        for param_file in param_files[:5]:  # Last 5 fits
            try:
                with open(param_file) as f:
                    param_data = json.load(f)

                timestamp = param_file.stem.split("_")[-1]
                success = param_data.get("success", False)
                rss = param_data.get("rss", "N/A")
                r2 = param_data.get("r2", "N/A")
                o2_max = param_data.get("o2_max", "N/A")
                o2_range = param_data.get("o2_range", "N/A")
                failure_reason = param_data.get("failure_reason", "")
                network_diagnosis = param_data.get("network_diagnosis", "No diagnosis")

                summary_lines.extend(
                    [
                        f"Fit {timestamp}:",
                        f"  Success: {success}",
                        f"  RSS: {rss}",
                        f"  R²: {r2}",
                        f"  O₂ max: {o2_max}",
                        f"  O₂ range: {o2_range}",
                        f"  Failure: {failure_reason}",
                        f"  Network: {network_diagnosis}",
                        "",
                    ]
                )

            except Exception as e:
                summary_lines.append(f"Error reading {param_file}: {e}")

        # Check for parameter summary
        param_summary_file = persistent_dir / "parameter_summary.json"
        if param_summary_file.exists():
            try:
                with open(param_summary_file) as f:
                    summary_data = json.load(f)

                total_exp = summary_data.get("summary_stats", {}).get(
                    "total_experiments", 0
                )
                successful = summary_data.get("summary_stats", {}).get(
                    "successful_fits", 0
                )

                summary_lines.extend(
                    [
                        f"Overall batch results:",
                        f"  Total experiments: {total_exp}",
                        f"  Successful fits: {successful}",
                        f"  Success rate: {successful / total_exp * 100:.1f}%"
                        if total_exp > 0
                        else "  Success rate: 0%",
                        "",
                    ]
                )

            except Exception as e:
                summary_lines.append(f"Error reading parameter summary: {e}")

        summary_lines.append(f"All parameter files available in: {persistent_dir}")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"Error analyzing recent fits: {e}"


@tool
def test_ode_system_with_reference_params(
    data_path: str,
    network_path: str,
    exp_name: str,
    reference_params_json: str = '{"qy_0": 1.0, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 0.005, "k_5": 0.003}',
) -> str:
    """Test the ODE system with reference parameters to debug flat line issues.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        exp_name: Name of experiment to test
        reference_params_json: JSON string of reference parameters (default: literature values)
    """
    try:
        # Load data and network
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        exp_data = data[exp_name]
        time_exp = np.array(exp_data["time"])
        oxygen_exp = np.array(exp_data["oxygen"])

        # Create ODE system
        ode_func, species_list, species_idx = create_ode_system(
            network, exp_data["metadata"]
        )

        # Parse reference parameters from input
        try:
            reference_params = json.loads(reference_params_json)
        except json.JSONDecodeError as e:
            return f"Error parsing reference parameters JSON: {e}"

        # Set initial conditions
        y0 = np.zeros(len(species_list))
        if "RuII" in species_idx:
            y0[species_idx["RuII"]] = exp_data["metadata"].get("c_Ru", 10)
        if "S2O8" in species_idx:
            y0[species_idx["S2O8"]] = exp_data["metadata"].get("c_S2O8", 6000)

        # Test ODE integration
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_test = odeint(
                    ode_func,
                    y0,
                    time_exp,
                    args=(reference_params, exp_data["metadata"]),
                    rtol=1e-6,
                    atol=1e-8,
                )

            o2_pred = (
                y_test[:, species_idx["O2"]]
                if "O2" in species_idx
                else np.zeros_like(time_exp)
            )

            # Calculate metrics
            o2_range = np.max(o2_pred) - np.min(o2_pred)
            o2_max = np.max(o2_pred)
            exp_range = np.max(oxygen_exp) - np.min(oxygen_exp)
            exp_max = np.max(oxygen_exp)
            rss = np.sum((oxygen_exp - o2_pred) ** 2)

            # Save diagnostic plot
            persistent_dir = _get_persistent_output_dir()
            test_plot_path = persistent_dir / f"ode_test_{exp_name}.png"
            _create_fit_plot(
                time_exp,
                oxygen_exp,
                o2_pred,
                f"ODE Test - {exp_name}",
                exp_data["metadata"],
                save_path=str(test_plot_path),
            )

            # Detailed analysis
            analysis = [
                f"ODE System Test Results for {exp_name}:",
                f"  Species found: {species_list}",
                f"  Initial conditions: {dict(zip(species_list, y0))}",
                f"  Reference parameters used: {reference_params}",
                f"  ",
                f"  Results:",
                f"    O₂ predicted range: {o2_range:.6f} µM",
                f"    O₂ predicted max: {o2_max:.6f} µM",
                f"    Experimental range: {exp_range:.3f} µM",
                f"    Experimental max: {exp_max:.3f} µM",
                f"    RSS: {rss:.2f}",
                f"    Ratio pred/exp range: {o2_range / exp_range:.6f}"
                if exp_range > 0
                else "    Ratio: undefined",
                f"  ",
                f"  Diagnosis:",
            ]

            if o2_max < 0.001:
                analysis.append(
                    "    CRITICAL: Essentially zero O₂ production - ODE system not working"
                )
            elif o2_range < 0.01 * exp_range:
                analysis.append(
                    "    CRITICAL: Flat line output - parameters/network issue"
                )
            elif o2_max < 0.1 * exp_max:
                analysis.append(
                    "    Issue: O₂ production too low - parameter scaling problem"
                )
            else:
                analysis.append("    Good: O₂ production in reasonable range")

            analysis.extend(
                [
                    f"  ",
                    f"  Test plot saved to: {test_plot_path}",
                    f"  Final concentrations: {dict(zip(species_list, y_test[-1]))}",
                ]
            )

            return "\n".join(analysis)

        except Exception as e:
            return f"ODE integration failed: {e}"

    except Exception as e:
        return f"Error in ODE system test: {e}"


@tool
def fit_with_reference_start(
    data_path: str,
    network_path: str,
    results_path: str,
    exp_name: str,
    reference_params_json: str = '{"qy_0": 1.0, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 0.005, "k_5": 0.003}',
) -> str:
    """Fit experiment using reference parameters as starting point to debug optimization.

    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        results_path: Path to save results JSON file
        exp_name: Name of experiment to fit
        reference_params_json: JSON string of reference parameters to use as seed
    """
    try:
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            network = initialize_default_network()
            save_reaction_network(network_path, network)

        exp_data = data[exp_name]

        # Parse reference parameters
        try:
            reference_params = json.loads(reference_params_json)
        except json.JSONDecodeError as e:
            return f"Error parsing reference parameters JSON: {e}"

        # Fit with reference starting point
        result = fit_reaction_network(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            network,
            exp_data["metadata"],
            reference_params=reference_params,  # Use reference parameters as seed
        )

        # Save detailed results
        persistent_dir = _get_persistent_output_dir()
        ref_result_file = persistent_dir / f"reference_fit_{exp_name}.json"
        with open(ref_result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Create comparison plot
        if result.get("success", False):
            plot_filename = persistent_dir / f"reference_fit_plot_{exp_name}.png"
            _create_fit_plot(
                np.array(exp_data["time"]),
                np.array(exp_data["oxygen"]),
                np.array(result["y_pred"]),
                f"Reference Start Fit - {exp_name}",
                exp_data["metadata"],
                save_path=str(plot_filename),
            )

            return (
                f"Reference-seeded fit for {exp_name}:\n"
                f"  Success: {result['success']}\n"
                f"  R² = {result['r2']:.4f}\n"
                f"  RSS = {result['rss']:.2f}\n"
                f"  O₂ max: {result.get('o2_max', 'N/A'):.4f} µM\n"
                f"  O₂ range: {result.get('o2_range', 'N/A'):.4f} µM\n"
                f"  Parameters: {result['params']}\n"
                f"  Plot: {plot_filename}\n"
                f"  Details: {ref_result_file}"
            )
        else:
            failure_reason = result.get("failure_reason", "Unknown")
            return (
                f"Reference-seeded fit FAILED for {exp_name}:\n"
                f"  Reason: {failure_reason}\n"
                f"  RSS: {result.get('rss', 'N/A')}\n"
                f"  O₂ max: {result.get('o2_max', 'N/A')}\n"
                f"  O₂ range: {result.get('o2_range', 'N/A')}\n"
                f"  Details saved to: {ref_result_file}"
            )

    except Exception as e:
        return f"Error in reference fit: {e}"


def initialize_default_network() -> Dict[str, Any]:
    """Initialize with Akhtar network structure for agent discovery."""
    return {
        "reactions": [
            {
                "equation": "RuII + hv -> RuII*",
                "type": "light",
                "quantum_yield": [0.8, 1.0],
                "description": "Photoexcitation of Ru(II) catalyst",
            },
            {
                "equation": "RuII* + S2O8 -> RuIII + SO4_rad + SO4",
                "type": "dark",
                "k_range": [1e7, 1e9],
                "description": "Excited Ru oxidation by persulfate",
            },
            {
                "equation": "RuII + SO4_rad -> RuIII + SO4",
                "type": "dark",
                "k_range": [1e8, 1e10],
                "description": "Ru oxidation by sulfate radical",
            },
            {
                "equation": "RuIII + OH -> RuII + OH_rad",
                "type": "dark",
                "k_range": [1e3, 1e5],
                "description": "Ru(III) reduction by hydroxide",
            },
            {
                "equation": "2 OH_rad -> H2O2",
                "type": "dark",
                "k_range": [1e9, 1e10],
                "description": "OH radical dimerization",
            },
            {
                "equation": "2 RuIII + H2O2 -> 2 RuII + O2 + 2 H",
                "type": "dark",
                "k_range": [1e3, 1e5],
                "description": "O2 evolution from H2O2",
            },
            {
                "equation": "RuIII + hv -> RuIII*",
                "type": "light",
                "quantum_yield": [0.8, 1.0],
                "description": "Photoexcitation of Ru(III)",
            },
            {
                "equation": "RuIII* + S2O8 -> RuIV_intermediate",
                "type": "dark",
                "k_range": [1e7, 1e9],
                "description": "Formation of Ru(IV) intermediate",
            },
            {
                "equation": "2 RuIV_intermediate -> Ru_Dimer_active",
                "type": "dark",
                "k_range": [1e5, 1e7],
                "description": "Active dimer formation",
            },
            {
                "equation": "RuIV_intermediate + RuIV_intermediate -> Ru_oligomer_inactive",
                "type": "dark",
                "k_range": [1e6, 1e8],
                "description": "Inactive oligomer formation",
            },
            {
                "equation": "OH_rad + RuII -> decomposed_Ru",
                "type": "dark",
                "k_range": [1e8, 1e10],
                "description": "Catalyst decomposition",
            },
        ],
        "metadata": {
            "created_by": "initialize_default_network",
            "description": "Akhtar network structure for agent to discover improved kinetic pathways",
            "version": "4.0",
        },
    }


def setup_working_directory(work_dir: str, data_path: str) -> None:
    """Set up working directory with default files."""
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # Create default network if it doesn't exist
    network_path = work_path / "current_network.json"
    if not network_path.exists():
        default_network = initialize_default_network()
        save_reaction_network(str(network_path), default_network)

    # Create empty results file if it doesn't exist
    results_path = work_path / "fit_results.json"
    if not results_path.exists():
        with open(results_path, "w") as f:
            json.dump({"experiments": {}, "best_phenomenological_score": 0.0}, f)


def create_tools() -> Dict[str, Tool]:
    """Create and return all available tools."""
    return {
        "describe_experimental_data": describe_experimental_data,
        "get_current_network": get_current_network,
        "fit_single_experiment": fit_single_experiment,
        "fit_all_experiments": fit_all_experiments,
        "evaluate_phenomenological_trends": evaluate_phenomenological_trends,
        "modify_reaction_network": modify_reaction_network,
        "analyze_fit_with_vision": analyze_fit_with_vision,
    }
