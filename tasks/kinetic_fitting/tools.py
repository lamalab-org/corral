"""Tools for kinetic model fitting and analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from scipy.stats import linregress
import matplotlib.pyplot as plt
import io
import base64

from corral.backend.tool import tool


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
        
        for r in reactants:
            if r not in ignored:
                stoich[r] = stoich.get(r, 0) - 1
        
        for p in products:
            if p not in ignored:
                stoich[p] = stoich.get(p, 0) + 1
        
        return cls(
            equation=equation,
            type=rxn_dict["type"],
            reactants=reactants,
            products=products,
            stoichiometry=stoich,
            k_range=rxn_dict.get("k_range"),
            quantum_yield=rxn_dict.get("quantum_yield"),
        )


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
    maxiter: int = 100,
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
            objective, bounds, maxiter=maxiter, popsize=15,
            seed=42, workers=1, updating="deferred",
            atol=1e-4, tol=0.01,
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
        
        return {
            "params": params_dict,
            "rss": result.fun,
            "y_pred": o2_pred,
            "success": result.success,
            "species_list": species_list,
            "full_solution": y_final,
            "species_idx": species_idx,
        }
        
    except Exception as e:
        return {
            "params": {},
            "rss": 1e10,
            "y_pred": np.zeros_like(time_exp),
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
) -> str:
    """Create fit visualization and return as base64 string."""
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
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode("utf-8")



@tool
def describe_experimental_data(env) -> str:
    """List available experimental datasets with metadata.
    
    Args:
        env: The kinetic fitting environment
    """
    lines = ["Available experimental datasets:\n"]
    
    for exp_name, data in list(env.experimental_data.items())[:10]:
        meta = data["metadata"]
        lines.extend([
            f"{exp_name}:",
            f"  [Ru(bpy)3Cl2]: {meta.get('c_Ru', 'N/A')} µM",
            f"  [Na2S2O8]: {meta.get('c_S2O8', 'N/A')} µM",
            f"  Power/Irradiance: {meta.get('irradiance', 'N/A')} W/m²",
            f"  pH: {meta.get('pH', 'N/A')}",
            f"  Data points: {len(data['time'])}",
            f"  Time range: {data['time'].min():.1f} - {data['time'].max():.1f} s",
            f"  O2 range: {data['oxygen'].min():.3f} - {data['oxygen'].max():.3f} µM\n",
        ])
    
    if len(env.experimental_data) > 10:
        lines.append(f"... and {len(env.experimental_data) - 10} more experiments")
    
    return "\n".join(lines)


@tool
def fit_single_experiment(env, exp_name: str) -> str:
    """Fit current reaction network to a single experiment.
    
    Args:
        env: The kinetic fitting environment
        exp_name: Name of the experiment to fit
    """
    if exp_name not in env.experimental_data:
        return f"Error: Experiment {exp_name} not found"
    
    data = env.experimental_data[exp_name]
    result = fit_reaction_network(
        data["time"], data["oxygen"],
        env.current_reaction_network, data["metadata"]
    )
    
    if not result.get("success", False):
        return f"Fit failed for {exp_name}: {result.get('error', 'Unknown error')}"
    
    y_exp = data["oxygen"]
    y_pred = result["y_pred"]
    residuals = y_exp - y_pred
    
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_exp - np.mean(y_exp)) ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Store result in environment
    env.fit_history.append({
        "exp_name": exp_name,
        "rss": result["rss"],
        "r2": r2,
    })
    
    return (
        f"Fit results for {exp_name}:\n"
        f"  R² = {r2:.4f}\n"
        f"  RMSE = {rmse:.4f} µM\n"
        f"  RSS = {result['rss']:.2f}"
    )


@tool
def fit_all_experiments(env) -> str:
    """Fit current reaction network to all experiments.
    
    Args:
        env: The kinetic fitting environment
    """
    results = []
    for exp_name in env.experimental_data.keys():
        data = env.experimental_data[exp_name]
        result = fit_reaction_network(
            data["time"], data["oxygen"],
            env.current_reaction_network, data["metadata"]
        )
        results.append({
            "exp_name": exp_name,
            "success": result.get("success", False),
            "rss": result.get("rss", 1e10),
        })
    
    successful = [r for r in results if r["success"]]
    total_rss = sum(r["rss"] for r in successful)
    avg_rss = total_rss / len(successful) if successful else 1e10
    
    # Update environment
    env.fit_history.append({
        "type": "all_experiments",
        "total_rss": total_rss,
        "avg_rss": avg_rss,
        "num_successful": len(successful),
    })
    
    return (
        f"Fitted all experiments:\n"
        f"  Successful: {len(successful)}/{len(results)}\n"
        f"  Total RSS: {total_rss:.2f}\n"
        f"  Average RSS: {avg_rss:.2f}"
    )


@tool
def evaluate_phenomenological_trends(env) -> str:
    """Evaluate how well model reproduces experimental trends.
    
    Args:
        env: The kinetic fitting environment
    """
    trends = {"c_Ru": {}, "c_S2O8": {}, "irradiance": {}}
    
    for exp_name, data in env.experimental_data.items():
        meta = data["metadata"]
        rates = np.gradient(data["oxygen"], data["time"])
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
    
    # Update best score
    if overall_score > env.best_score:
        env.best_score = overall_score
    
    return (
        f"Phenomenological trend scores:\n"
        f"  Ru concentration trend: {ru_score:.3f}\n"
        f"  S2O8 concentration trend: {s2o8_score:.3f}\n"
        f"  Irradiance trend: {irr_score:.3f}\n"
        f"  Overall score: {overall_score:.3f}\n"
        f"  Best score so far: {env.best_score:.3f}"
    )


@tool
def modify_reaction_network(
    env,
    add_reactions: Optional[List[Dict]] = None,
    remove_reactions: Optional[List[int]] = None,
    modify_k_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    modify_quantum_yields: Optional[Dict[int, Tuple[float, float]]] = None,
) -> str:
    """Modify the current reaction network.
    
    Args:
        env: The kinetic fitting environment
        add_reactions: List of reaction dictionaries to add
        remove_reactions: List of reaction indices to remove
        modify_k_ranges: Dict of {index: (new_min, new_max)} for rate constants
        modify_quantum_yields: Dict of {index: (new_min, new_max)} for quantum yields
    """
    new_network = {"reactions": list(env.current_reaction_network["reactions"])}
    changes = []
    
    if remove_reactions:
        for idx in sorted(remove_reactions, reverse=True):
            if 0 <= idx < len(new_network["reactions"]):
                removed = new_network["reactions"].pop(idx)
                changes.append(f"Removed reaction {idx}: {removed['equation']}")
    
    if modify_k_ranges:
        for idx, new_range in modify_k_ranges.items():
            if 0 <= idx < len(new_network["reactions"]):
                new_network["reactions"][idx]["k_range"] = tuple(new_range)
                changes.append(f"Modified k_range for reaction {idx}")
    
    if modify_quantum_yields:
        for idx, new_range in modify_quantum_yields.items():
            if 0 <= idx < len(new_network["reactions"]):
                qy_range = (max(0, new_range[0]), min(1, new_range[1]))
                new_network["reactions"][idx]["quantum_yield"] = qy_range
                changes.append(f"Modified quantum_yield for reaction {idx}")
    
    if add_reactions:
        for rxn in add_reactions:
            new_network["reactions"].append(rxn)
            changes.append(f"Added reaction: {rxn['equation']}")
    
    env.current_reaction_network = new_network
    
    return (
        f"Modified reaction network:\n" +
        "\n".join(f"  - {c}" for c in changes) +
        f"\n\nTotal reactions now: {len(new_network['reactions'])}"
    )


@tool
def analyze_fit_with_vision(env, exp_name: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Use vision model to analyze fit quality from plot.
    
    Args:
        env: The kinetic fitting environment
        exp_name: Name of experiment to analyze
        model: Vision model to use (default: claude-sonnet-4-20250514)
    """
    if exp_name not in env.experimental_data:
        return f"Error: Experiment {exp_name} not found"
    
    data = env.experimental_data[exp_name]
    result = fit_reaction_network(
        data["time"], data["oxygen"],
        env.current_reaction_network, data["metadata"]
    )
    
    if not result.get("success", False):
        return f"Cannot analyze: fit failed for {exp_name}"
    
    plot_b64 = _create_fit_plot(
        result["time"],
        data["oxygen"],
        result["y_pred"],
        exp_name,
        data["metadata"],
    )
    
    try:
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
                            "Analyze this kinetic fit for photocatalytic O₂ evolution.\n\n"
                            "The plot shows experimental data (blue points) vs model prediction (orange line), "
                            "with residuals below.\n\n"
                            "Evaluate:\n"
                            "1. How well does the model capture the induction period?\n"
                            "2. How well does it match the growth phase slope and curvature?\n"
                            "3. How well does it capture the plateau/maximum?\n"
                            "4. Are there systematic deviations?\n"
                            "5. Where is the fit worst (early/middle/late)?\n"
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