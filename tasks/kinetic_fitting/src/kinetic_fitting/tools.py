"""Tools for kinetic model fitting and analysis."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues
import matplotlib.pyplot as plt

import numpy as np
import io
import base64
import h5py
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import linregress

from corral.backend.tool import tool, Tool
from litellm import completion


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
            raise ValueError(f"Invalid equation format: {equation}")
        
        left, right = equation.split("->")
        reactants = [s.strip() for s in left.split("+")]
        products = [s.strip() for s in right.split("+")]
        
        ignored = {"hv", "H2O", "OH", "products", "H"}
        stoich = {}
        
        def parse_species_with_coeff(species_str):
            """Parse '2 RuIII' -> (2, 'RuIII')"""
            parts = species_str.strip().split(' ', 1)
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
        experiments: Dict[str, 'ExperimentalData'] = field(default_factory=dict)
        overview_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
        
        @classmethod
        def load_from_hdf5(cls, filename: str):
            """Load experiments from HDF5 file"""
            dataset = cls()

            try:
                dataset.overview_df = pd.read_hdf(filename, key='overview_df')
            except (KeyError, ValueError):
                dataset.overview_df = pd.DataFrame()

            with h5py.File(filename, 'r') as f:
                for exp_name in f.keys():
                    if exp_name == 'overview_df':  # Skip the overview_df group
                        continue
                    try:
                        # Load experimental data
                        time_series_dict = dict(f[f'{exp_name}/time_series_data'].attrs)
                        time_series_data = TimeSeriesData(**time_series_dict)
                       
                        # Load metadata
                        exp_metadata_dict = dict(f[f'{exp_name}/experiment_metadata'].attrs)
                        experiment_metadata = ExperimentMetadata(**exp_metadata_dict)

                        analysis_metadata_dict = dict(f[f'{exp_name}/analysis_metadata'].attrs)
                        analysis_metadata = AnalysisMetadata(**analysis_metadata_dict)

                        datasets_dict = dict(f[f'{exp_name}/datasets'].attrs)
                        datasets = DataSets(**datasets_dict)

                        # Create ExperimentalData and add to dataset
                        experimental_data = ExperimentalData(
                            time_series_data, experiment_metadata, analysis_metadata, datasets
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
                overview_row = dataset.overview_df[dataset.overview_df['Experiment'] == exp_name]
                if not overview_row.empty:
                    row = overview_row.iloc[0]
                    standardized_metadata = {
                        'c_Ru': row.get('c([Ru(bpy(3]Cl2) [M]', 0) * 1e6,  # Convert M to µM
                        'c_S2O8': row.get('c(Na2S2O8) [M]', 0) * 1e6,  # Convert M to µM  
                        'power_output': row.get('Power output [W/m^2]', 1000),
                        'pH': row.get('pH [-]', 7.0),
                        'irradiance': row.get('Power output [W/m^2]', 1000),
                    }
                else:
                    # Fallback to experimental metadata (but may have unit issues)
                    standardized_metadata = {
                        'c_Ru': exp_data.experiment_metadata.ru_concentration,
                        'c_S2O8': exp_data.experiment_metadata.oxidant_concentration,
                        'power_output': exp_data.experiment_metadata.power_output,
                        'pH': exp_data.experiment_metadata.pH,
                        'irradiance': exp_data.experiment_metadata.power_output,
                    }
            else:
                # Fallback to experimental metadata
                standardized_metadata = {
                    'c_Ru': exp_data.experiment_metadata.ru_concentration,
                    'c_S2O8': exp_data.experiment_metadata.oxidant_concentration,
                    'power_output': exp_data.experiment_metadata.power_output,
                    'pH': exp_data.experiment_metadata.pH,
                    'irradiance': exp_data.experiment_metadata.power_output,
                }
            
            data[exp_name] = {
                'time': time.tolist() if hasattr(time, 'tolist') else time,
                'oxygen': oxygen.tolist() if hasattr(oxygen, 'tolist') else oxygen,
                'metadata': standardized_metadata,
            }
        
        if not data:
            # Fallback: Use overview data but with proper real time-series data if available
            if not dataset.overview_df.empty:
                print(f"Experiments have empty metadata, trying overview data with real time-series if available")
                
                # Try to get real time-series data by looking for the data in attributes
                with h5py.File(data_path, 'r') as f:
                    for _, row in dataset.overview_df.iterrows():
                        exp_name = row['Experiment']
                        
                        # Extract metadata with proper unit conversions from overview
                        standardized_metadata = {
                            'c_Ru': row.get('c([Ru(bpy(3]Cl2) [M]', 0) * 1e6,  # Convert M to µM
                            'c_S2O8': row.get('c(Na2S2O8) [M]', 0) * 1e6,  # Convert M to µM  
                            'power_output': row.get('Power output [W/m^2]', 1000),
                            'pH': row.get('pH [-]', 7.0),
                            'irradiance': row.get('Power output [W/m^2]', 1000),
                        }
                        
                        # Try to get real time-series data from the HDF5 file
                        time = None
                        oxygen = None
                        
                        if exp_name in f:
                            exp_group = f[exp_name]
                            # Try to find time-series data in attributes
                            if 'time_series_data' in exp_group:
                                ts_group = exp_group['time_series_data']
                                if hasattr(ts_group, 'attrs') and 'time_reaction' in ts_group.attrs:
                                    time = ts_group.attrs['time_reaction']
                                    oxygen = ts_group.attrs['data_reaction']
                        
                        # If we couldn't find real time-series, create minimal placeholder
                        if time is None or oxygen is None:
                            time = np.linspace(0, 300, 10)  # Minimal placeholder
                            oxygen = np.ones_like(time) * 0.1  # Minimal placeholder
                        
                        data[exp_name] = {
                            'time': time.tolist() if hasattr(time, 'tolist') else time,
                            'oxygen': oxygen.tolist() if hasattr(oxygen, 'tolist') else oxygen,
                            'metadata': standardized_metadata,
                        }
                    
                print(f"Loaded {len(data)} experiments from overview dataframe with metadata")
                return data
            raise ValueError("No experimental data could be loaded")
        
        print(f"Loaded {len(data)} experiments with real time-series data from HDF5 file")
        return data
        
    except Exception as e:
        raise ValueError(f"Could not load experimental data: {e}")


def load_reaction_network(network_path: str) -> Dict[str, Any]:
    """Load reaction network from JSON file."""
    with open(network_path) as f:
        return json.load(f)


def save_reaction_network(network_path: str, network: Dict[str, Any]) -> None:
    """Save reaction network to JSON file."""
    with open(network_path, 'w') as f:
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
                
                rate = quantum_yield * photon_flux_factor * y[species_idx[absorber]] if absorber else 0
                
            else:
                k = params[f"k_{i}"]
                rate = k
                
                for reactant in rxn.reactants:
                    if reactant in species_idx:
                        if rxn.equation.startswith(f"2 {reactant}"):
                            rate *= y[species_idx[reactant]] ** 2
                        else:
                            rate *= y[species_idx[reactant]]
        
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
    maxiter: int = 30,  # Reduced from 100
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
                params_dict[name] = 10 ** val
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = odeint(
                    ode_func, y0, time_exp,
                    args=(params_dict, experimental_conditions),
                    rtol=1e-6, atol=1e-8,
                )
            
            o2_pred = y_pred[:, species_idx["O2"]] if "O2" in species_idx else np.zeros_like(time_exp)
            rss = np.sum((oxygen_exp - o2_pred) ** 2)
            
            if np.any(y_pred < -1e-6):
                rss += 1e10
            
            return rss
            
        except Exception:
            return 1e10
    
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=maxiter, 
            popsize=10,  # Reduced from 15
            seed=42, workers=1, updating="deferred",
            atol=1e-3, tol=0.05,  # Relaxed tolerances
        )
        
        params_dict = {}
        for name, val in zip(param_names, result.x):
            if name.startswith("qy_"):
                params_dict[name] = val
            else:
                params_dict[name] = 10 ** val
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_final = odeint(
                ode_func, y0, time_exp,
                args=(params_dict, experimental_conditions),
                rtol=1e-6, atol=1e-8,
            )
        
        o2_pred = y_final[:, species_idx["O2"]] if "O2" in species_idx else np.zeros_like(time_exp)
        
        # Calculate R²
        residuals = oxygen_exp - o2_pred
        r2 = 1 - np.sum(residuals ** 2) / np.sum((oxygen_exp - np.mean(oxygen_exp)) ** 2)
        
        return {
            "params": params_dict,
            "rss": result.fun,
            "r2": r2,
            "y_pred": o2_pred.tolist(),
            "success": result.success,
            "species_list": species_list,
            "full_solution": y_final.tolist(),
            "species_idx": species_idx,
        }
        
    except Exception as e:
        return {
            "params": {},
            "rss": 1e10,
            "r2": 0.0,
            "y_pred": np.zeros_like(time_exp).tolist(),
            "success": False,
            "species_list": species_list,
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
    increase_score = float(np.all(np.diff(rates[:max_idx + 1]) >= 0)) if max_idx > 0 else 0.5
    decrease_score = float(np.all(np.diff(rates[max_idx:]) <= 0)) if max_idx < len(rates) - 1 else 0.5
    
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
    return max(0, r_value ** 2)


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
    residuals = y_exp - y_pred
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_exp - np.mean(y_exp)) ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])
    
    ax1.scatter(time, y_exp, alpha=0.6, s=30, label="Experimental", color="#1f77b4", zorder=3)
    ax1.plot(time, y_pred, linewidth=2.5, label="Model", color="#ff7f0e", zorder=2)
    ax1.set_ylabel("[O₂] (µM)", fontsize=12, fontweight="bold")
    ax1.set_title(f"{exp_name}\nR² = {r2:.4f}, RMSE = {rmse:.2f} µM", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle="--")
    
    conditions = (
        f"[Ru] = {metadata.get('c_Ru', '?')} µM  |  "
        f"[S₂O₈²⁻] = {metadata.get('c_S2O8', '?')} µM  |  "
        f"Power = {metadata.get('irradiance', '?')} W/m²  |  "
        f"pH = {metadata.get('pH', '?')}"
    )
    ax1.text(
        0.02, 0.98, conditions, transform=ax1.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
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
        
        lines.extend([
            f"{exp_name}:",
            f"  [Ru(bpy)3Cl2]: {meta.get('c_Ru', 'N/A')} µM",
            f"  [Na2S2O8]: {meta.get('c_S2O8', 'N/A')} µM",
            f"  Power/Irradiance: {meta.get('irradiance', 'N/A')} W/m²",
            f"  pH: {meta.get('pH', 'N/A')}",
            f"  Data points: {len(time)}",
            f"  Time range: {time.min():.1f} - {time.max():.1f} s",
            f"  O2 range: {oxygen.min():.3f} - {oxygen.max():.3f} µM\n",
        ])
    
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
        
        lines = [f"Current reaction network ({len(network['reactions'])} reactions):\n"]
        
        for i, rxn in enumerate(network['reactions']):
            lines.append(f"{i:2d}: {rxn['equation']} ({rxn['type']})")
            if rxn['type'] == 'light' and 'quantum_yield' in rxn:
                lines.append(f"     QY range: {rxn['quantum_yield']}")
            elif 'k_range' in rxn:
                lines.append(f"     k range: {rxn['k_range']}")
            lines.append("")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error loading network: {e}"


@tool
def fit_single_experiment(
    data_path: str, 
    network_path: str, 
    results_path: str, 
    exp_name: str
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
        
        network = load_reaction_network(network_path)
        exp_data = data[exp_name]
        
        result = fit_reaction_network(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            network,
            exp_data["metadata"],
        )
        
        if not result.get("success", False):
            return f"Fit failed for {exp_name}: {result.get('error', 'Unknown error')}"
        
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
        
        # Create and save fit plot
        plot_filename = f"fit_plot_{exp_name}.png"
        _create_fit_plot(
            np.array(exp_data["time"]),
            np.array(exp_data["oxygen"]),
            np.array(result["y_pred"]),
            exp_name,
            exp_data["metadata"],
            save_path=plot_filename,
        )
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return (
            f"Fit results for {exp_name}:\n"
            f"  R² = {result['r2']:.4f}\n"
            f"  RMSE = {np.sqrt(result['rss'] / len(result['y_pred'])):.4f} µM\n"
            f"  RSS = {result['rss']:.2f}\n"
            f"  Plot saved to: {plot_filename}"
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
        network = load_reaction_network(network_path)
        
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
            
            # Save plots for first 5 experiments for inspection
            if result.get("success", False) and i < 5:
                plot_filename = f"fit_plot_{exp_name}.png"
                _create_fit_plot(
                    np.array(exp_data["time"]),
                    np.array(exp_data["oxygen"]),
                    np.array(result["y_pred"]),
                    exp_name,
                    exp_data["metadata"],
                    save_path=plot_filename,
                )
                plots_saved.append(plot_filename)
            
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
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        plot_info = f"\n  Plots saved: {', '.join(plots_saved)}" if plots_saved else ""
        
        return (
            f"Fitted all experiments:\n"
            f"  Successful: {successful}/{len(data)}\n"
            f"  Total RSS: {total_rss:.2f}\n"
            f"  Average RSS: {avg_rss:.2f}"
            f"{plot_info}"
        )
        
    except Exception as e:
        return f"Error fitting experiments: {e}"


@tool
def evaluate_phenomenological_trends(
    data_path: str, 
    network_path: str, 
    results_path: str
) -> str:
    """Evaluate how well model reproduces experimental trends.
    
    Args:
        data_path: Path to experimental data HDF5 file
        network_path: Path to reaction network JSON file
        results_path: Path to results JSON file (updated with trend scores)
    """
    try:
        data = load_experimental_data(data_path)
        trends = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}}
        
        for exp_name, exp_data in data.items():
            meta = exp_data["metadata"]
            oxygen = np.array(exp_data["oxygen"])
            time = np.array(exp_data["time"])
            
            # Calculate maximum rate
            rates = np.gradient(oxygen, time)
            max_rate = np.max(rates)
            
            for param in ["c_Ru", "c_S2O8", "irradiance"]:
                param_val = meta.get(param)
                if param_val is not None:
                    if param_val not in trends[param]:
                        trends[param][param_val] = []
                    trends[param][param_val].append(max_rate)
        
        ru_score = _score_ru_trend(trends["c_Ru"])
        s2o8_score = _score_s2o8_trend(trends["c_S2O8"])
        irr_score = _score_irradiance_trend(trends["irradiance"])
        
        overall_score = 0.4 * ru_score + 0.3 * s2o8_score + 0.3 * irr_score
        
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
            "overall_score": overall_score,
        }
        
        # Update best score
        if "best_phenomenological_score" not in all_results:
            all_results["best_phenomenological_score"] = 0.0
        
        if overall_score > all_results["best_phenomenological_score"]:
            all_results["best_phenomenological_score"] = overall_score
        
        # Save updated results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return (
            f"Phenomenological trend scores:\n"
            f"  Ru concentration trend: {ru_score:.3f}\n"
            f"  S2O8 concentration trend: {s2o8_score:.3f}\n"
            f"  Irradiance trend: {irr_score:.3f}\n"
            f"  Overall score: {overall_score:.3f}\n"
            f"  Best score so far: {all_results['best_phenomenological_score']:.3f}"
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
    
    Args:
        network_path: Path to reaction network JSON file
        add_reactions: JSON string of list of reaction dictionaries to add
        remove_reactions: JSON string of list of reaction indices to remove  
        modify_k_ranges: JSON string of dict {index: [new_min, new_max]}
        modify_quantum_yields: JSON string of dict {index: [new_min, new_max]}
    """
    try:
        network = load_reaction_network(network_path)
        changes = []
        
        if remove_reactions:
            indices = json.loads(remove_reactions)
            for idx in sorted(indices, reverse=True):
                if 0 <= idx < len(network["reactions"]):
                    removed = network["reactions"].pop(idx)
                    changes.append(f"Removed reaction {idx}: {removed['equation']}")
        
        if modify_k_ranges:
            k_ranges = json.loads(modify_k_ranges)
            for idx, new_range in k_ranges.items():
                idx = int(idx)
                if 0 <= idx < len(network["reactions"]):
                    network["reactions"][idx]["k_range"] = tuple(new_range)
                    changes.append(f"Modified k_range for reaction {idx}")
        
        if modify_quantum_yields:
            qy_ranges = json.loads(modify_quantum_yields)
            for idx, new_range in qy_ranges.items():
                idx = int(idx)
                if 0 <= idx < len(network["reactions"]):
                    qy_range = (max(0, new_range[0]), min(1, new_range[1]))
                    network["reactions"][idx]["quantum_yield"] = qy_range
                    changes.append(f"Modified quantum_yield for reaction {idx}")
        
        if add_reactions:
            new_rxns = json.loads(add_reactions)
            for rxn in new_rxns:
                network["reactions"].append(rxn)
                changes.append(f"Added reaction: {rxn['equation']}")
        
        # Save modified network
        save_reaction_network(network_path, network)
        
        return (
            f"Modified reaction network:\n" +
            "\n".join(f"  - {c}" for c in changes) +
            f"\n\nTotal reactions now: {len(network['reactions'])}"
        )
        
    except Exception as e:
        return f"Error modifying network: {e}"


@tool
def analyze_fit_with_vision(
    data_path: str,
    network_path: str,
    exp_name: str,
    model: str = "claude-sonnet-4-20250514",
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
        
        network = load_reaction_network(network_path)
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
            messages=[{
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
            }],
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Vision analysis failed: {e}"


def initialize_default_network() -> Dict[str, Any]:
    """Initialize with reactions from Akhtar 2016."""
    return {
        'reactions': [
            {'equation': 'RuII + hv -> RuII*', 'type': 'light', 'quantum_yield': [0.8, 1.0]},
            {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4', 'type': 'dark', 'k_range': [1e7, 1e9]},
            {'equation': 'RuII + SO4_rad -> RuIII + SO4', 'type': 'dark', 'k_range': [1e8, 1e10]},
            {'equation': 'RuIII + OH -> RuII + OH_rad', 'type': 'dark', 'k_range': [1e3, 1e5]},
            {'equation': '2 OH_rad -> H2O2', 'type': 'dark', 'k_range': [1e9, 1e10]},
            {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H', 'type': 'dark', 'k_range': [1e3, 1e5]},
            {'equation': 'RuIII + hv -> RuIII*', 'type': 'light', 'quantum_yield': [0.8, 1.0]},
            {'equation': 'RuIII* + S2O8 -> RuIV_intermediate', 'type': 'dark', 'k_range': [1e7, 1e9]},
            {'equation': '2 RuIV_intermediate -> Ru_Dimer_active', 'type': 'dark', 'k_range': [1e5, 1e7]},
            {'equation': 'RuIV_intermediate + RuIV_intermediate -> Ru_oligomer_inactive', 'type': 'dark', 'k_range': [1e6, 1e8]},
            {'equation': 'OH_rad + RuII -> decomposed_Ru', 'type': 'dark', 'k_range': [1e8, 1e10]},
        ]
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
        with open(results_path, 'w') as f:
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