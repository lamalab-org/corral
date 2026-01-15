"""Tools for kinetic model fitting and analysis."""

import base64
import copy
import glob
import io
import json
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from litellm import completion
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import linregress

from corral.backend.tool import Tool, tool
from kinetic_fitting.phenomenological_trends import generate_full_qualitative_assessment

# Set matplotlib backend environment variables before any matplotlib imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["DISPLAY"] = ""  # Disable display

matplotlib.use("Agg")  # Use non-GUI backend to prevent threading issues

# Additional safety: explicitly configure pyplot for non-interactive use
plt.ioff()  # Turn off interactive mode

# Import threading to help with matplotlib threading issues

# Create a lock for matplotlib operations
_matplotlib_lock = threading.Lock()


def _get_persistent_output_dir() -> Path:
    """Get or create persistent output directory for plots and results."""
    # Use the task directory's parent to ensure persistence across runs
    output_dir = Path(os.environ["DIRECTORY"])
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
        photon_flux: Optional[float] = None
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
                        # Handle potential missing fields gracefully
                        valid_fields = ExperimentMetadata.__annotations__.keys()
                        filtered_metadata = {
                            k: v
                            for k, v in exp_metadata_dict.items()
                            if k in valid_fields
                        }
                        experiment_metadata = ExperimentMetadata(**filtered_metadata)

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
                        # Try to get photon_flux from overview or fallback to experiment metadata
                        "photon_flux": row.get(
                            "photon_flux", exp_data.experiment_metadata.photon_flux
                        ),
                    }
                else:
                    # Fallback to experimental metadata (but may have unit issues)
                    standardized_metadata = {
                        "c_Ru": exp_data.experiment_metadata.ru_concentration,
                        "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                        "power_output": exp_data.experiment_metadata.power_output,
                        "pH": exp_data.experiment_metadata.pH,
                        "irradiance": exp_data.experiment_metadata.power_output,
                        "photon_flux": exp_data.experiment_metadata.photon_flux,
                    }
            else:
                # Fallback to experimental metadata
                standardized_metadata = {
                    "c_Ru": exp_data.experiment_metadata.ru_concentration,
                    "c_S2O8": exp_data.experiment_metadata.oxidant_concentration,
                    "power_output": exp_data.experiment_metadata.power_output,
                    "pH": exp_data.experiment_metadata.pH,
                    "irradiance": exp_data.experiment_metadata.power_output,
                    "photon_flux": exp_data.experiment_metadata.photon_flux,
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
                            "photon_flux": row.get("photon_flux", None),
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

    time_stamp = str(int(time.time()))

    with open(network_path, "w") as f:
        json.dump(network, f, indent=2)

    # Also save to persistent directory
    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"

    persistent_network_path = persistent_dir / f"current_network_{time_stamp}.json"

    with open(persistent_network_path, "w") as f:
        json.dump(network, f, indent=2)

    persistent_network_path = persistent_dir / "current_network.json"

    with open(persistent_network_path, "w") as f:
        json.dump(network, f, indent=2)


def create_ode_system(
    reaction_network: dict,
    experimental_conditions: dict,
) -> Tuple[Any, List[str], Dict[str, int]]:
    """
    Convert reaction network to ODE system for integration.
    UPDATED: Matches 'Script 2' physics with competitive absorption (Lambert-Beer).
    UPDATED: Uses photon_flux if available for correct scaling.
    """
    reactions = [Reaction.from_dict(r) for r in reaction_network["reactions"]]

    all_species = set()
    for rxn in reactions:
        all_species.update(rxn.stoichiometry.keys())

    species_list = sorted(all_species)
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    # Constants from Script 2 / HTE setup
    PATHLENGTH = 2.25  # cm
    EPSILON_RU_II = 8500.0  # M^-1 cm^-1
    EPSILON_RU_III = 540.0  # M^-1 cm^-1
    AVOGADRO_NUMBER = 6.022e23
    VOLUME_L = PATHLENGTH * 1e-3  # Assuming 1 cm^2 area

    def ode_func(y: np.ndarray, t: float, params: dict, conditions: dict) -> np.ndarray:
        dydt = np.zeros_like(y)

        # 1. Calculate Photochemistry Physics (Lambert-Beer & Competition)
        # Convert uM (y) to M for absorbance calculation
        c_ru_ii_M = y[species_idx["RuII"]] * 1e-6 if "RuII" in species_idx else 0
        c_ru_iii_M = y[species_idx["RuIII"]] * 1e-6 if "RuIII" in species_idx else 0

        # Total Absorbance (A_tot)
        absorbance_tot = (
            (c_ru_ii_M * EPSILON_RU_II) + (c_ru_iii_M * EPSILON_RU_III)
        ) * PATHLENGTH

        # Prevent division by zero if solution is perfectly clear
        if absorbance_tot < 1e-9:
            absorptance_factor = 0
            fraction_ru_ii = 0
            fraction_ru_iii = 0
        else:
            # Fraction of incident light absorbed (1 - 10^-A)
            absorptance_factor = 1 - 10 ** (-absorbance_tot)

            # Apportion photons based on contribution to absorbance
            fraction_ru_ii = (c_ru_ii_M * EPSILON_RU_II * PATHLENGTH) / absorbance_tot
            fraction_ru_iii = (
                c_ru_iii_M * EPSILON_RU_III * PATHLENGTH
            ) / absorbance_tot

        # Incident Photon Flux Calculation
        # We need the Volumetric Photon Flux in uM/s to match the ODE units.

        if conditions.get("photon_flux") is not None:
            # Case 1: Use provided photon_flux (photons/s)
            # Convert photons/s -> mol/s -> M/s -> uM/s
            photon_flux_photons_s = conditions["photon_flux"]
            photon_flux_mol_s = photon_flux_photons_s / AVOGADRO_NUMBER
            incident_flux_M_s = photon_flux_mol_s / VOLUME_L
            incident_flux_uM_s = incident_flux_M_s * 1e6
            incident_flux = incident_flux_uM_s

        else:
            # Case 2: Fallback to Irradiance (W/m^2)
            # Estimate photon flux assuming 450nm light
            irradiance_W_m2 = conditions.get("irradiance", 1000)
            irradiance_W_cm2 = irradiance_W_m2 * 1e-4

            # Energy of one 450nm photon in Joules
            # E = h*c/lambda = 6.626e-34 * 3e8 / 450e-9
            E_photon_J = 4.41e-19

            photon_flux_approx = irradiance_W_cm2 / E_photon_J  # photons/s/cm^2

            # Convert to Volumetric Flux (Flux_vol = Flux_area / Pathlength)
            # Note: This assumes the irradiance is incident on the face of the cuvette
            photon_flux_vol_photons_s_cm3 = photon_flux_approx / PATHLENGTH

            # Convert to uM/s
            # photons/s/cm3 -> mol/s/cm3 -> mol/s/L -> M/s -> uM/s
            photon_flux_vol_mol_s_cm3 = photon_flux_vol_photons_s_cm3 / AVOGADRO_NUMBER
            photon_flux_vol_M_s = photon_flux_vol_mol_s_cm3 * 1000  # 1000 cm3 = 1 L
            incident_flux_uM_s = photon_flux_vol_M_s * 1e6

            incident_flux = incident_flux_uM_s

        for i, rxn in enumerate(reactions):
            rate = 0.0

            if rxn.type == "light":
                # Light Reaction Rate = Flux * Absorptance * Fraction_Species * Quantum_Yield

                # Identify the absorber based on reactants
                is_ru_ii_absorber = "RuII" in rxn.reactants
                is_ru_iii_absorber = "RuIII" in rxn.reactants

                absorbed_flux = 0.0
                if is_ru_ii_absorber:
                    absorbed_flux = incident_flux * absorptance_factor * fraction_ru_ii
                elif is_ru_iii_absorber:
                    absorbed_flux = incident_flux * absorptance_factor * fraction_ru_iii

                # Get quantum yield (or fitting parameter)
                qy = params.get(
                    f"qy_{i}", rxn.quantum_yield[0] if rxn.quantum_yield else 0.1
                )

                rate = absorbed_flux * qy

            else:
                # Dark reactions (standard mass action)
                k = params[f"k_{i}"]
                rate = k

                for reactant in rxn.reactants:
                    if reactant in species_idx:
                        # Handle stoichiometry: 2 A -> ... means rate propto [A]^2
                        stoich_coeff = 0
                        # Count occurrence in reactants list (handled by Reaction class,
                        # but simple check here for "2 RuIII")
                        if rxn.equation.startswith(f"2 {reactant}"):
                            rate *= y[species_idx[reactant]] ** 2
                        else:
                            rate *= y[species_idx[reactant]]
                    elif reactant in ["H2O", "H+"]:
                        rate *= 1.0

            # Apply rate to stoichiometry
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
    maxiter: int = 300,
    reference_params: dict = None,
) -> dict:
    """
    Fit reaction network parameters.
    UPDATED: Fits 'Reaction Rate' (derivative) rather than cumulative O2.
    UPDATED: Fixes Excited State Decay (k ~ 1.5e6) automatically.
    """
    ode_func, species_list, species_idx = create_ode_system(
        reaction_network, experimental_conditions
    )

    # 1. Setup Initial Conditions
    y0 = np.zeros(len(species_list))
    if "RuII" in species_idx:
        y0[species_idx["RuII"]] = experimental_conditions.get("c_Ru", 10)
    if "S2O8" in species_idx:
        y0[species_idx["S2O8"]] = experimental_conditions.get("c_S2O8", 6000)

    # 2. Setup Parameters & Fixed Physics
    bounds = []
    param_names = []
    reactions = [Reaction.from_dict(r) for r in reaction_network["reactions"]]

    # Identify fixed parameters (Script 2 fixes k8 = 1/650ns)
    fixed_params = {}

    for i, rxn in enumerate(reactions):
        if rxn.type == "light":
            bounds.append(rxn.quantum_yield or (0.0, 1.0))
            param_names.append(f"qy_{i}")
        else:
            k_range = rxn.k_range or (1e-3, 1e10)
            bounds.append((np.log10(k_range[0]), np.log10(k_range[1])))
            param_names.append(f"k_{i}")

    # 3. Calculate Experimental Rate (Derivative Fitting)
    # Calculate d[O2]/dt from experimental data
    # Smoothing is often required for experimental derivatives, but we use gradient for now
    from scipy.signal import savgol_filter

    y_diff = np.diff(oxygen_exp) / np.diff(time_exp)
    savgol_window_factor = 5
    rate_exp = savgol_filter(
        oxygen_exp,
        window_length=int(len(y_diff) / savgol_window_factor),
        polyorder=3,
        delta=time_exp[1] - time_exp[0],
    )

    def objective(params_log: np.ndarray) -> float:
        params_dict = fixed_params.copy()

        # Unpack optimized parameters
        for name, val in zip(param_names, params_log, strict=True):
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

            # Get predicted O2
            o2_pred = (
                y_pred[:, species_idx["O2"]]
                if "O2" in species_idx
                else np.zeros_like(time_exp)
            )

            # CALCULATE PREDICTED RATE (Derivative)
            rate_pred = np.gradient(o2_pred, time_exp)

            # RSS on RATES, not concentrations
            # This prioritizes the "shape" and "burst" over the final yield
            rss = np.sum((rate_exp - rate_pred) ** 2)

            # Sanity penalties
            if np.max(o2_pred) < 0.05:
                rss += 1e10  # Dead reaction penalty

            return rss

        except Exception:
            return 1e15

    # Optimization Routine (Same as before, just using new objective)
    try:
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
            popsize=15,
            seed=42,
            workers=1,
            updating="deferred",
            disp=False,  # cleaner output
            atol=1e-6,
            tol=0.01,
        )

        # Reconstruct full params
        final_params = fixed_params.copy()
        for name, val in zip(param_names, result.x, strict=False):
            if name.startswith("qy_"):
                final_params[name] = val
            else:
                final_params[name] = 10**val

        # Generate final curves for return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_final = odeint(
                ode_func,
                y0,
                time_exp,
                args=(final_params, experimental_conditions),
                rtol=1e-6,
                atol=1e-8,
            )

        o2_pred = (
            y_final[:, species_idx["O2"]]
            if "O2" in species_idx
            else np.zeros_like(time_exp)
        )

        # Calculate R2 based on RATES, matching the optimization target
        rate_pred = np.gradient(o2_pred, time_exp)
        residuals = rate_exp - rate_pred
        r2 = 1 - np.sum(residuals**2) / np.sum((rate_exp - np.mean(rate_exp)) ** 2)

        return {
            "params": final_params,
            "rss": result.fun,
            "r2": r2,
            "y_pred": o2_pred.tolist(),  # Return conc for plotting
            "rate_pred": rate_pred.tolist(),  # Optional: return rates
            "success": result.success,
            "species_list": species_list,
            "full_solution": y_final.tolist(),
            "species_idx": species_idx,
            "failure_reason": "" if result.success else "Optimization failed",
            "o2_range": np.max(o2_pred) - np.min(o2_pred),
            "o2_max": np.max(o2_pred),
        }

    except Exception as e:
        return {
            "params": {},
            "rss": 1e10,
            "r2": -999.0,
            "y_pred": np.zeros_like(time_exp).tolist(),
            "success": False,
            "error": str(e),
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
        return 0.0

    concentrations = np.array(sorted(rates_vs_s2o8.keys()))
    rates = np.array([np.mean(rates_vs_s2o8[c]) for c in concentrations])

    # Binary score: 1.0 if rate at highest concentration > rate at lowest concentration
    if rates[-1] > rates[0]:
        return 1.0
    return 0.0


def _score_irradiance_trend(rates_vs_irradiance: dict) -> float:
    """Score irradiance trend (should be linear)."""
    if len(rates_vs_irradiance) < 3:
        return 0.5

    concentrations = np.array(sorted(rates_vs_irradiance.keys()))
    rates = np.array([np.mean(rates_vs_irradiance[c]) for c in concentrations])

    _, _, r_value, _, _ = linregress(concentrations, rates)
    return max(0, r_value**2)  # pyright: ignore[reportOperatorIssue]


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
    save_path: str | None = None,
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
        import matplotlib as mpl  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415

        mpl.use("Agg", force=True)

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
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
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
    trends,
    trends_original,
):
    """
    Creates and saves plots comparing model-predicted trends with original experimental trends.

    Args:
        trends (dict): Dictionary of trends calculated from model predictions.
        trends_original (dict): Dictionary of trends calculated from original data.

    Returns:
        str: A message indicating that the plot has been saved.
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Comparison of Phenomenological Trends: Model vs. Experimental Data",
        fontsize=16,
    )
    axes = axes.flatten()

    param_map = {
        "c_Ru": "Ru concentration (M)",
        "c_S2O8": "S2O8 concentration (M)",
        "irradiance": "Irradiance (W/m^2)",
        "pH": "pH",
    }

    # Iterate over the parameters and plot the data
    for i, param in enumerate(["c_Ru", "c_S2O8", "irradiance", "pH"]):
        ax = axes[i]

        # --- Process and plot original experimental data ---
        original_trend = trends_original.get(param, {})
        if original_trend:
            # Sort by the parameter value (concentration, pH, etc.)
            sorted_original_items = sorted(original_trend.items())
            x_original = [item[0] for item in sorted_original_items]
            # Calculate mean of rates for each parameter value
            y_original = [np.mean(item[1]) for item in sorted_original_items]
            ax.plot(
                x_original,
                y_original,
                "o-",
                label="Experimental Data",
                color="blue",
                markersize=8,
            )

        # --- Process and plot model predicted data ---
        predicted_trend = trends.get(param, {})
        if predicted_trend:
            sorted_predicted_items = sorted(predicted_trend.items())
            x_predicted = [item[0] for item in sorted_predicted_items]
            y_predicted = [np.mean(item[1]) for item in sorted_predicted_items]
            ax.plot(
                x_predicted,
                y_predicted,
                "s--",
                label="Model Prediction",
                color="red",
                markersize=6,
            )

        ax.set_xlabel(param_map.get(param, param), fontsize=12)
        ax.set_ylabel("Maximum O₂ Evolution Rate", fontsize=12)
        ax.set_title(f"Trend for {param_map.get(param, param)}", fontsize=14)
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    timestamp = int(time.time())
    plt.savefig(
        _get_persistent_output_dir() / Path(f"phenomenological_trends_{timestamp}.png")
    )
    plt.close()
    return (
        f"\nPhenomenological trend plots saved to: '{_get_persistent_output_dir() / Path(f'phenomenological_trends_{timestamp}.png')}'",
        Path(f"phenomenological_trends_{timestamp}.png"),
    )


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
        except Exception:
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

    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"

    try:
        network = load_reaction_network(persistent_network_path)
    except FileNotFoundError:
        # Create default network if it doesn't exist
        network = initialize_default_network()
        save_reaction_network(persistent_network_path, network)
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
def fit_single_experiment(data_path: str, results_path: str, exp_name: str) -> str:
    """Fit current reaction network to a single experiment.

    Args:
        data_path: Path to experimental data HDF5 file
        results_path: Path to save fit results JSON file
        exp_name: Name of the experiment to fit
    """
    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"

    try:
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(persistent_network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(persistent_network_path, network)

        exp_data = data[exp_name]

        result = fit_reaction_network(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            network,
            exp_data["metadata"],
        )
        # import pdb; pdb.set_trace()
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
            with open(results_path) as f:  # noqa: PTH123
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
        # with open(results_path, "w") as f:
        #     json.dump(all_results, f, indent=2)
        import time  # noqa: PLC0415

        fit_results_file = persistent_dir / f"fit_results.json.{int(time.time())}"

        with open(fit_results_file, "w") as f:
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
            f"  Fit results saved to: {fit_results_file}\n"
            f"  Persistent directory: {persistent_dir}"
        )

    except Exception as e:
        return f"Error fitting experiment: {e}"


def fit_reaction_network_global(
    data: dict[str, Any],
    reaction_network: dict,
    maxiter: int = 100,
) -> dict:
    """
    Global fit using Rate-based loss function.
    """
    # Use valid conditions for setup
    sample_key = next(iter(data))
    ode_func, species_list, species_idx = create_ode_system(
        reaction_network, data[sample_key]["metadata"]
    )

    reactions = [Reaction.from_dict(r) for r in reaction_network["reactions"]]
    bounds = []
    param_names = []
    fixed_params = {}

    for i, rxn in enumerate(reactions):
        # Fix Decay Constant again
        if (
            rxn.type == "dark"
            and "RuII_ex" in rxn.reactants
            and "RuII" in rxn.products
            and len(rxn.reactants) == 1
        ):
            decay_k = 1.0 / (650e-9)
            fixed_params[f"k_{i}"] = decay_k
            continue

        if rxn.type == "light":
            bounds.append(rxn.quantum_yield or (0.0, 1.0))
            param_names.append(f"qy_{i}")
        else:
            k_range = rxn.k_range or (1e-3, 1e10)
            bounds.append((np.log10(k_range[0]), np.log10(k_range[1])))
            param_names.append(f"k_{i}")

    def global_objective(params_log: np.ndarray) -> float:
        params_dict = fixed_params.copy()
        for name, val in zip(param_names, params_log, strict=False):
            if name.startswith("qy_"):
                params_dict[name] = val
            else:
                params_dict[name] = 10**val

        total_rss = 0
        for exp_name, exp_data in data.items():
            meta = exp_data["metadata"]
            time_exp = np.array(exp_data["time"])
            oxygen_exp = np.array(exp_data["oxygen"])

            # Calculate experimental Rate
            rate_exp = np.gradient(oxygen_exp, time_exp)

            y0 = np.zeros(len(species_list))
            if "RuII" in species_idx:
                y0[species_idx["RuII"]] = meta.get("c_Ru", 10)
            if "S2O8" in species_idx:
                y0[species_idx["S2O8"]] = meta.get("c_S2O8", 6000)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = odeint(
                        ode_func,
                        y0,
                        time_exp,
                        args=(params_dict, meta),
                        rtol=1e-6,
                        atol=1e-8,
                    )
                o2_pred = (
                    y_pred[:, species_idx["O2"]]
                    if "O2" in species_idx
                    else np.zeros_like(time_exp)
                )

                # Calculate Predicted Rate
                rate_pred = np.gradient(o2_pred, time_exp)

                # Rate-based RSS
                total_rss += np.sum((rate_exp - rate_pred) ** 2)

            except Exception:
                total_rss += 1e15
        return total_rss

    result = differential_evolution(
        global_objective, bounds, maxiter=maxiter, popsize=15, seed=42, disp=True
    )

    best_params = fixed_params.copy()
    for name, val in zip(param_names, result.x, strict=False):
        if name.startswith("qy_"):
            best_params[name] = val
        else:
            best_params[name] = 10**val

    return {"params": best_params, "success": result.success, "total_rss": result.fun}


@tool
def fit_all_experiments(data_path: str, results_path: str) -> str:
    """Fit current reaction network globally to all experiments.

    Args:
        data_path: Path to experimental data HDF5 file
        results_path: Path to save fit results JSON file
    """

    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"
    try:
        data = load_experimental_data(data_path)
        network = load_reaction_network(persistent_network_path)

        # Perform global fit (one model for all)
        print(f"Starting global fit for {len(data)} experiments...")
        global_result = fit_reaction_network_global(data, network, maxiter=150)

        if not global_result["success"]:
            return "Global optimization failed to converge."

        global_params = global_result["params"]
        ode_func, species_list, species_idx = create_ode_system(network, {})

        all_results = {"experiments": {}, "global_params": global_params}
        total_rss = 0.0
        plots_saved = []

        # Generate predictions for each experiment using the global parameters
        for exp_name, exp_data in data.items():
            meta = exp_data["metadata"]
            time_exp = np.array(exp_data["time"])
            oxygen_exp = np.array(exp_data["oxygen"])

            y0 = np.zeros(len(species_list))
            if "RuII" in species_idx:
                y0[species_idx["RuII"]] = meta.get("c_Ru", 10)
            if "S2O8" in species_idx:
                y0[species_idx["S2O8"]] = meta.get("c_S2O8", 6000)

            y_pred_full = odeint(ode_func, y0, time_exp, args=(global_params, meta))
            o2_pred = (
                y_pred_full[:, species_idx["O2"]]
                if "O2" in species_idx
                else np.zeros_like(time_exp)
            )

            rss = np.sum((oxygen_exp - o2_pred) ** 2)
            r2 = 1 - rss / np.sum((oxygen_exp - np.mean(oxygen_exp)) ** 2)

            all_results["experiments"][exp_name] = {
                "rss": rss,
                "r2": r2,
                "success": True,
                "params": global_params,
                "y_pred": o2_pred.tolist(),
            }

            # Save individual fit plots
            persistent_dir = _get_persistent_output_dir()
            plot_filename = persistent_dir / f"fit_plot_{exp_name}.png"
            _create_fit_plot(
                time_exp,
                oxygen_exp,
                o2_pred,
                exp_name,
                meta,
                save_path=str(plot_filename),
            )
            plots_saved.append(str(plot_filename))
            total_rss += rss

        all_results["summary"] = {
            "total_experiments": len(data),
            "total_rss": total_rss,
            "avg_rss": total_rss / len(data),
        }

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        with open(results_path + "." + str(int(time.time())), "w") as f:
            json.dump(all_results, f, indent=2)

        time_stamp = str(int(time.time()))
        output_file = (
            _get_persistent_output_dir() / f"{Path(results_path).stem}.{time_stamp}"
        )
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        return (
            f"Fitted all experiments globally (one model for all):\n"
            f"  Global Parameters: {global_params}\n"
            f"  Total RSS: {total_rss:.2f}\n"
            f"  Average RSS: {all_results['summary']['avg_rss']:.2f}\n"
            f"  Plots and results saved to: {output_file}"
        )
    except Exception as e:
        return f"Error fitting experiments: {e}"


def _calculate_mae(
    pred_dict: Dict[float, List[float]], exp_dict: Dict[float, List[float]]
) -> Optional[float]:
    """Calculates the Mean Absolute Error between predicted and experimental trend means."""
    errors = []
    # We iterate over parameter values present in both datasets
    for val in exp_dict:  # noqa: PLC0206
        if val in pred_dict and len(exp_dict[val]) > 0 and len(pred_dict[val]) > 0:
            avg_exp = np.mean(exp_dict[val])
            avg_pred = np.mean(pred_dict[val])
            errors.append(abs(avg_exp - avg_pred))

    return np.mean(errors) if errors else None  # pyright: ignore[reportReturnType]


def _calculate_magnitude_score(
    pred_dict: Dict[float, List[float]],
    exp_dict: Dict[float, List[float]],
    threshold: float = 0.1,
) -> float:
    """
    Calculates a score (0.0 to 1.0) based on how well the magnitude matches.
    If model prediction is < 10% of experiment (flat line), score is 0.
    """
    model_means = []
    exp_means = []

    # Align keys to ensure we compare apples to apples
    for key in exp_dict:  # noqa: PLC0206
        if key in pred_dict and exp_dict[key] and pred_dict[key]:
            model_means.append(np.mean(pred_dict[key]))
            exp_means.append(np.mean(exp_dict[key]))

    if not model_means or not exp_means:
        return 0.0

    avg_model = np.mean(model_means)
    avg_exp = np.mean(exp_means)

    if avg_exp == 0:
        return 0.0

    ratio = avg_model / avg_exp

    # PENALTY CLIFF:
    # If the model predicts less than 10% of the signal, it's a failure.
    if ratio < threshold:
        return 0.0

    # If it's within range, score based on log-distance (to handle orders of magnitude)
    # or simple ratio. Simple ratio is better for linear plots.
    # We cap at 1.0. If model over-predicts, we penalize inversely.
    if ratio <= 1.0:
        return ratio
    else:
        return 1.0 / ratio


@tool
def evaluate_phenomenological_trends(data_path: str) -> str:
    """Evaluate how well a single global model reproduces experimental trends.

    Args:
        data_path: Path to experimental data HDF5 file
    """
    try:
        data = load_experimental_data(data_path)
        reaction_network = load_reaction_network(
            _get_persistent_output_dir() / "current_network.json"
        )

        # 1. Perform Global Fit
        print("Performing global fit for trend evaluation...")
        global_result = fit_reaction_network_global(data, reaction_network, maxiter=5)
        global_params = global_result["params"]

        ode_func, species_list, species_idx = create_ode_system(reaction_network, {})

        # 2. Calculate Trends
        trends = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}, "pH": {}}
        trends_original = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}, "pH": {}}

        for exp_name, exp_data in data.items():
            meta = exp_data["metadata"]
            oxygen = np.array(exp_data["oxygen"])
            time = np.array(exp_data["time"])

            y0 = np.zeros(len(species_list))
            if "RuII" in species_idx:
                y0[species_idx["RuII"]] = meta.get("c_Ru", 10)
            if "S2O8" in species_idx:
                y0[species_idx["S2O8"]] = meta.get("c_S2O8", 6000)

            y_pred_full = odeint(ode_func, y0, time, args=(global_params, meta))
            y_pred = (
                y_pred_full[:, species_idx["O2"]]
                if "O2" in species_idx
                else np.zeros_like(time)
            )

            # Calculate maximum rates
            rates = np.gradient(y_pred, time)
            max_rate = np.max(rates)

            rates_original = np.gradient(oxygen, time)
            max_rate_original = np.max(rates_original)

            for param in ["c_Ru", "c_S2O8", "irradiance", "pH"]:
                param_val = meta.get(param)
                if param_val is not None:
                    trends[param].setdefault(param_val, []).append(max_rate)
                    trends_original[param].setdefault(param_val, []).append(
                        max_rate_original
                    )

        # 3. Scoring with MAGNITUDE PENALTY
        # Calculate shape scores (qualitative)
        ru_shape = _score_ru_trend(trends["c_Ru"])
        s2o8_shape = _score_s2o8_trend(trends["c_S2O8"])
        irr_shape = _score_irradiance_trend(trends["irradiance"])
        ph_shape = _score_ph_trend(trends["pH"])

        # Calculate magnitude scores (quantitative)
        ru_mag = _calculate_magnitude_score(trends["c_Ru"], trends_original["c_Ru"])
        s2o8_mag = _calculate_magnitude_score(
            trends["c_S2O8"], trends_original["c_S2O8"]
        )
        irr_mag = _calculate_magnitude_score(
            trends["irradiance"], trends_original["irradiance"]
        )
        ph_mag = _calculate_magnitude_score(trends["pH"], trends_original["pH"])

        # Combine scores: Shape * Magnitude
        # If magnitude is 0 (flat line), the total score becomes 0.
        ru_score = ru_shape * ru_mag
        s2o8_score = s2o8_shape * s2o8_mag
        irr_score = irr_shape * irr_mag
        ph_score = ph_shape * ph_mag

        overall_score = (
            0.3 * ru_score + 0.25 * s2o8_score + 0.25 * irr_score + 0.2 * ph_score
        )

        maes = {p: _calculate_mae(trends[p], trends_original[p]) for p in trends}
        mae_report = "\n".join(
            [
                f"  - {p} MAE: {val:.4f} µM/s"
                if val is not None
                else f"  - {p} MAE: N/A"
                for p, val in maes.items()
            ]
        )

        # Get the best score from previous result files
        best_previous_score = _get_best_previous_phenomenological_score()

        # Determine the actual best score
        if overall_score > best_previous_score:
            best_phenomenological_score = overall_score
            is_new_best = True
        else:
            best_phenomenological_score = best_previous_score
            is_new_best = False

        fitted_network = copy.deepcopy(reaction_network)
        for i, rxn in enumerate(fitted_network["reactions"]):
            if rxn.get("type") == "light":
                param_key = f"qy_{i}"
                if param_key in global_params:
                    rxn["fitted_quantum_yield"] = global_params[param_key]
            else:
                param_key = f"k_{i}"
                if param_key in global_params:
                    rxn["fitted_k"] = global_params[param_key]

        # Build results dict
        all_results = {
            "phenomenological_trends": {
                "ru_score": ru_score,
                "s2o8_score": s2o8_score,
                "irradiance_score": irr_score,
                "pH_score": ph_score,
                "overall_score": overall_score,
                "global_params": global_params,
                "components": {
                    "ru_shape": ru_shape,
                    "ru_mag": ru_mag,
                    "s2o8_shape": s2o8_shape,
                    "s2o8_mag": s2o8_mag,
                },
            },
            "network": fitted_network,  # Now contains fitted_k and fitted_quantum_yield
            "best_phenomenological_score": best_phenomenological_score,
        }

        import time as timer

        filename = f"{_get_persistent_output_dir().as_posix()}/phenomenologic_result.json.{int(timer.time())}"

        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)

        plot_info = _create_phenomenological_plots(trends, trends_original)

        best_score_status = (
            "NEW BEST!" if is_new_best else "(best remains from previous run)"
        )
        qualitative_trend_evaluation = generate_full_qualitative_assessment(
            trends, trends_original
        )

        return (
            f"Phenomenological trend scores saved at filename: {filename}\n"
            f"Phenomenological trend scores (Global Model):\n"
            f"  Overall score: {overall_score:.3f}\n"
            f"  Ru trend: {ru_score:.3f} (Shape: {ru_shape:.2f}, Mag: {ru_mag:.2f})\n"
            f"  S2O8 trend: {s2o8_score:.3f} (Shape: {s2o8_shape:.2f}, Mag: {s2o8_mag:.2f})\n"
            f"  Irr trend: {irr_score:.3f} (Shape: {irr_shape:.2f}, Mag: {irr_mag:.2f})\n"
            f"  pH trend: {ph_score:.3f} (Shape: {ph_shape:.2f}, Mag: {ph_mag:.2f})\n"
            f"Mean Absolute Errors (MAE) in Max Rate:\n{mae_report}\n\n"
            f"  Global Params used: {global_params}\n"
            f"  *** Best Phenomenological Score: {best_phenomenological_score:.3f} {best_score_status} ***\n"
            f"{plot_info}"
            f"{qualitative_trend_evaluation}"
        )

    except Exception as e:
        return f"Error evaluating trends: {e}"


def _get_best_previous_phenomenological_score() -> float:
    """Find the best phenomenological score from all previous result files.

    Scans all phenomenologic_result.json.* files, extracts timestamps,
    and returns the best score found across all files.

    Returns:
        The best phenomenological score found, or 0.0 if no files exist.
    """
    output_dir = _get_persistent_output_dir()
    pattern = f"{output_dir.as_posix()}/phenomenologic_result.json.*"
    result_files = glob.glob(pattern)

    if not result_files:
        return 0.0

    best_score = 0.0

    # Extract timestamps and sort files by timestamp (most recent last)
    files_with_timestamps = []
    for filepath in result_files:
        # Extract timestamp from filename (e.g., phenomenologic_result.json.1234567890)
        match = re.search(r"phenomenologic_result\.json\.(\d+)$", filepath)
        if match:
            timestamp = int(match.group(1))
            files_with_timestamps.append((timestamp, filepath))

    # Sort by timestamp
    files_with_timestamps.sort(key=lambda x: x[0])

    # Check all files to find the best score
    for timestamp, filepath in files_with_timestamps:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Check for best_phenomenological_score in the file
            if "best_phenomenological_score" in data:
                score = data["best_phenomenological_score"]
                if score > best_score:
                    best_score = score

            # Also check the overall_score in phenomenological_trends
            if "phenomenological_trends" in data:
                overall = data["phenomenological_trends"].get("overall_score", 0.0)
                if overall > best_score:
                    best_score = overall

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue

    return best_score


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
    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"
    try:
        try:
            network = load_reaction_network(persistent_network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(persistent_network_path, network)

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
        save_reaction_network(persistent_network_path, network)

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
    exp_name: str,
    model: str = "claude-sonnet-4-5",
) -> str:
    """Use vision model to analyze fit quality from plot.

    Args:
        data_path: Path to experimental data HDF5 file
        exp_name: Name of experiment to analyze
        model: Vision model to use
    """

    persistent_dir = _get_persistent_output_dir()
    persistent_network_path = persistent_dir / "current_network.json"
    try:
        data = load_experimental_data(data_path)
        if exp_name not in data:
            return f"Error: Experiment {exp_name} not found"

        try:
            network = load_reaction_network(persistent_network_path)
        except FileNotFoundError:
            # Create default network if it doesn't exist
            network = initialize_default_network()
            save_reaction_network(persistent_network_path, network)

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
                "equation": "RuII + hv -> RuII_ex",
                "type": "light",
                "quantum_yield": [0.1, 1.0],
                "description": "Photoexcitation of Ru catalyst (k1)",
            },
            {
                "equation": "RuII_ex -> RuII",
                "type": "dark",
                "k_range": [1538461.5384615384, 1538461.5384615385],
                "description": "Excited state decay (k8= 1/650ns)",
            },
            {
                "equation": "RuII_ex + S2O8 -> RuIII + SO4",
                "type": "dark",
                "k_range": [1.0, 60.0],
                "description": "Oxidative quenching (k7)",
            },
            {
                "equation": "RuIII + H2O + hv -> H2O2 + RuII + H+",
                "type": "light",
                "quantum_yield": [0.1, 1.0],
                "description": "Light-driven water oxidation (k2)",
            },
            {
                "equation": "RuIII + RuIII -> Ru_Dimer",
                "type": "dark",
                "k_range": [0.001, 0.1],
                "description": "Dimer formation (k3)",
            },
            {
                "equation": "H2O2 -> O2",
                "type": "dark",
                "k_range": [0.001, 0.5],
                "description": "H2O2 decomposition to O2 (k5)",
            },
            {
                "equation": "RuIII -> Inactive",
                "type": "dark",
                "k_range": [0.001, 0.5],
                "description": "RuIII deactivation (k6)",
            },
            {
                "equation": "RuIII + RuIII + Ru_Dimer -> Ru_Dimer + Ru_Dimer",
                "type": "dark",
                "k_range": [0.001, 0.1],
                "description": "Autocatalytic dimer formation (k4)",
            },
        ],
        "metadata": {
            "created_by": "reaction_network_conversion",
            "description": "Reaction network for Ru-catalyzed photochemical water oxidation with dimer formation",
            "version": "2.0",
        },
    }


def setup_working_directory(work_dir: str, data_path: str) -> None:
    """Set up working directory with default files."""
    work_path = Path(_get_persistent_output_dir())
    work_path.mkdir(parents=True, exist_ok=True)

    # Create default network if it doesn't exist
    network_path = work_path / "current_network.json"
    if not network_path.exists():
        default_network = initialize_default_network()
        save_reaction_network(str(network_path), default_network)

    # Create empty results file if it doesn't exist
    # results_path = work_path / Path("fit_results.json")
    # if not results_path.exists():
    #     with open(results_path, "w") as f:
    #         json.dump({"experiments": {}, "best_phenomenological_score": 0.0}, f)


def _find_latest_phenomenological_plot() -> Optional[Path]:
    """Find the most recent phenomenological trends plot in the output directory."""
    output_dir = _get_persistent_output_dir()
    pattern = f"{output_dir.as_posix()}/phenomenological_trends_*.png"
    plot_files = glob.glob(pattern)

    if not plot_files:
        return None

    files_with_timestamps = []
    for filepath in plot_files:
        match = re.search(r"phenomenological_trends_(\d+)\.png$", filepath)
        if match:
            timestamp = int(match.group(1))
            files_with_timestamps.append((timestamp, Path(filepath)))

    if not files_with_timestamps:
        return None

    files_with_timestamps.sort(key=lambda x: x[0], reverse=True)
    return files_with_timestamps[0][1]


@tool
def analyse_last_phenomenological_trends_image_with_vision(
    model: str = "gpt-4o",
) -> str:
    """Analyze the latest phenomenological trends plot using a vision model.

    Args:
        model: Vision model to use for analysis
    """
    try:
        latest_plot = _find_latest_phenomenological_plot()

        with open(latest_plot, "rb") as f:
            plot_b64 = base64.b64encode(f.read()).decode("utf-8")

        prompt = """Analyze this 4-panel comparison of model (red dashed) vs experiment (blue solid).

        Top-left (Ru concentration):
        - Does the model show a maximum? At what concentration?
        - Does the model capture inhibition at high [Ru]?

        Top-right (S2O8 concentration):
        - Does the model saturate or keep increasing?
        - Where does saturation begin vs experiment?

        Bottom-left (Irradiance):
        - Is the relationship linear for both?
        - Does the slope match?

        Bottom-right (pH):
        - Where is the experimental optimum pH?
        - Does the model show pH dependence at all?

        General:
        - Which parameter shows the worst agreement?
        - Is there a systematic offset (model always below/above)?
        - At what parameter ranges do the largest deviations occur (low, mid, high)?"""

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
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return f"Plot: {latest_plot.name}\n\n{response.choices[0].message.content}"

    except Exception as e:
        return f"Vision analysis failed: {e}"


def create_tools() -> dict[str, Tool]:
    """Create and return all available tools."""
    return {
        "describe_experimental_data": describe_experimental_data,
        "get_current_network": get_current_network,
        "fit_single_experiment": fit_single_experiment,
        "fit_all_experiments": fit_all_experiments,
        "evaluate_phenomenological_trends": evaluate_phenomenological_trends,
        "modify_reaction_network": modify_reaction_network,
        "analyze_fit_with_vision": analyze_fit_with_vision,
        "analyse_last_phenomenological_trends_image_with_vision": analyse_last_phenomenological_trends_image_with_vision,
    }  # type: ignore
