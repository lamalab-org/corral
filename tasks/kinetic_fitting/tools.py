import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
import warnings
  from scipy.stats import linregress

@dataclass
class Reaction:
    """Represents a single reaction in the network."""
    equation: str
    type: str  # 'light' or 'dark'
    reactants: List[str]
    products: List[str]
    stoichiometry: Dict[str, float]
    k_range: Optional[Tuple[float, float]] = None
    quantum_yield: Optional[Tuple[float, float]] = None
    
    @classmethod
    def from_dict(cls, rxn_dict: dict):
        """Parse reaction from dictionary."""
        equation = rxn_dict['equation']
        
        # Split on arrow
        if '->' not in equation:
            raise ValueError(f"Invalid equation format: {equation}")
        
        left, right = equation.split('->')
        
        # Parse reactants and products
        reactants = [s.strip() for s in left.split('+')]
        products = [s.strip() for s in right.split('+')]
        
        # Build stoichiometry dictionary
        # Assume stoichiometry of 1 for now (can be extended)
        stoich = {}
        for r in reactants:
            if r != 'hv' and r != 'H2O' and r != 'OH':  # Ignore photons, water, hydroxide
                stoich[r] = -1
        for p in products:
            if p != 'products' and p != 'H' and p != 'H2O':  # Ignore generic products
                stoich[p] = 1
        
        return cls(
            equation=equation,
            type=rxn_dict['type'],
            reactants=reactants,
            products=products,
            stoichiometry=stoich,
            k_range=rxn_dict.get('k_range'),
            quantum_yield=rxn_dict.get('quantum_yield')
        )


def create_ode_system(reaction_network: dict, experimental_conditions: dict):
    """
    Convert reaction network to ODE system.
    
    Returns:
        ode_func: Function dydt = f(y, t, params, conditions)
        species_list: List of species names
        species_idx: Dict mapping species name to index
    """
    reactions = [Reaction.from_dict(r) for r in reaction_network['reactions']]
    
    # Identify all species involved
    all_species = set()
    for rxn in reactions:
        all_species.update(rxn.stoichiometry.keys())
    
    species_list = sorted(list(all_species))
    species_idx = {sp: i for i, sp in enumerate(species_list)}
    
    def ode_func(y, t, params, conditions):
        """
        ODE function for scipy.integrate.odeint
        
        Args:
            y: concentration vector
            t: time
            params: dict of parameters (rate constants/quantum yields)
            conditions: experimental conditions
        """
        dydt = np.zeros_like(y)
        
        for i, rxn in enumerate(reactions):
            if rxn.type == 'light':
                # Light-driven reaction
                # rate = quantum_yield * photon_flux * [absorbing_species]
                quantum_yield = params[f'qy_{i}']
                
                # Get irradiance (W/m^2) and convert to effective rate
                # This is a simplification - proper conversion would need absorption
                irradiance = conditions.get('irradiance', 1000)
                photon_flux_factor = irradiance / 1000  # Normalize to reference
                
                # Find light-absorbing species (first reactant that's not hv)
                absorber = None
                for reactant in rxn.reactants:
                    if reactant != 'hv' and reactant in species_idx:
                        absorber = reactant
                        break
                
                if absorber:
                    rate = quantum_yield * photon_flux_factor * y[species_idx[absorber]]
                else:
                    rate = 0
                    
            else:  # dark reaction
                k = params[f'k_{i}']
                
                # Calculate rate based on mass action
                rate = k
                for reactant in rxn.reactants:
                    if reactant in species_idx:
                        # Handle second-order reactions (2 A -> ...)
                        if rxn.equation.startswith('2 ' + reactant):
                            rate *= y[species_idx[reactant]] ** 2
                        else:
                            rate *= y[species_idx[reactant]]
            
            # Apply stoichiometry
            for species, coeff in rxn.stoichiometry.items():
                if species in species_idx:
                    dydt[species_idx[species]] += coeff * rate
        
        return dydt
    
    return ode_func, species_list, species_idx


def fit_reaction_network(time_exp: np.ndarray, 
                        oxygen_exp: np.ndarray,
                        reaction_network: dict,
                        experimental_conditions: dict,
                        maxiter: int = 100) -> dict:
    """
    Fit reaction network to experimental oxygen evolution data.
    
    Args:
        time_exp: Experimental time points (s)
        oxygen_exp: Experimental oxygen concentration (µM)
        reaction_network: Reaction network dictionary
        experimental_conditions: Dict with c_Ru, c_S2O8, irradiance, pH
        maxiter: Maximum iterations for optimization
        
    Returns:
        Dict with fitted parameters, predictions, and metrics
    """
    ode_func, species_list, species_idx = create_ode_system(
        reaction_network, experimental_conditions
    )
    
    # Set up initial conditions
    y0 = np.zeros(len(species_list))
    if 'RuII' in species_idx:
        y0[species_idx['RuII']] = experimental_conditions.get('c_Ru', 10)
    if 'S2O8' in species_idx:
        y0[species_idx['S2O8']] = experimental_conditions.get('c_S2O8', 6000)
    
    # Set up parameter bounds and names
    bounds = []
    param_names = []
    reactions = [Reaction.from_dict(r) for r in reaction_network['reactions']]
    
    for i, rxn in enumerate(reactions):
        if rxn.type == 'light':
            bounds.append(rxn.quantum_yield or (0.0, 1.0))
            param_names.append(f'qy_{i}')
        else:
            k_range = rxn.k_range or (1e-3, 1e10)
            # Use log space for better optimization
            bounds.append((np.log10(k_range[0]), np.log10(k_range[1])))
            param_names.append(f'k_{i}')
    
    # Objective function
    def objective(params_log):
        # Convert log-space parameters back
        params_dict = {}
        for name, val in zip(param_names, params_log):
            if name.startswith('qy_'):
                params_dict[name] = val  # Quantum yields are not in log space
            else:
                params_dict[name] = 10 ** val  # Rate constants in log space
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred = odeint(ode_func, y0, time_exp, 
                              args=(params_dict, experimental_conditions),
                              rtol=1e-6, atol=1e-8)
            
            # Extract O2 concentration
            if 'O2' in species_idx:
                o2_pred = y_pred[:, species_idx['O2']]
            else:
                o2_pred = np.zeros_like(time_exp)
            
            # Calculate residual sum of squares
            rss = np.sum((oxygen_exp - o2_pred)**2)
            
            # Penalize negative concentrations
            if np.any(y_pred < -1e-6):
                rss += 1e10
            
            return rss
            
        except Exception as e:
            return 1e10  # Large penalty for failed integration
    
    # Optimize using differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=15,
            seed=42,
            workers=1,
            updating='deferred',
            atol=1e-4,
            tol=0.01
        )
        
        # Get final parameters
        params_dict = {}
        for name, val in zip(param_names, result.x):
            if name.startswith('qy_'):
                params_dict[name] = val
            else:
                params_dict[name] = 10 ** val
        
        # Get final prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_final = odeint(ode_func, y0, time_exp, 
                           args=(params_dict, experimental_conditions),
                           rtol=1e-6, atol=1e-8)
        
        o2_pred = y_final[:, species_idx['O2']] if 'O2' in species_idx else np.zeros_like(time_exp)
        
        return {
            'params': params_dict,
            'rss': result.fun,
            'y_pred': o2_pred,
            'success': result.success,
            'species_list': species_list,
            'full_solution': y_final,
            'species_idx': species_idx
        }
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return {
            'params': {},
            'rss': 1e10,
            'y_pred': np.zeros_like(time_exp),
            'success': False,
            'species_list': species_list,
            'error': str(e)
        }


class KineticTools:
    """Tools for the agent to analyze and modify kinetic models."""
    
    @staticmethod
    def describe_experimental_data(env) -> str:
        """Describe available experimental datasets."""
        description = "Available experimental datasets:\n\n"
        
        for exp_name, data in list(env.experimental_data.items())[:10]:  # Show first 10
            meta = data['metadata']
            description += f"{exp_name}:\n"
            description += f"  - [Ru(bpy)3Cl2]: {meta.get('c_Ru', 'N/A')} µM\n"
            description += f"  - [Na2S2O8]: {meta.get('c_S2O8', 'N/A')} µM\n"
            description += f"  - Irradiance/Power: {meta.get('irradiance', 'N/A')} W/m²\n"
            description += f"  - pH: {meta.get('pH', 'N/A')}\n"
            description += f"  - Data points: {len(data['time'])}\n"
            description += f"  - Time range: {data['time'].min():.1f} - {data['time'].max():.1f} s\n"
            description += f"  - O2 range: {data['oxygen'].min():.3f} - {data['oxygen'].max():.3f} µM\n\n"
        
        if len(env.experimental_data) > 10:
            description += f"... and {len(env.experimental_data) - 10} more experiments\n"
        
        return description
    
    @staticmethod
    def fit_single_experiment(env, exp_name: str, reaction_network: dict) -> dict:
        """Fit reaction network to a single experiment."""
        if exp_name not in env.experimental_data:
            return {'error': f"Experiment {exp_name} not found", 'success': False}
        
        data = env.experimental_data[exp_name]
        result = fit_reaction_network(
            data['time'],
            data['oxygen'],
            reaction_network,
            data['metadata']
        )
        result['exp_name'] = exp_name
        result['experimental_curve'] = data['oxygen']
        result['time'] = data['time']
        return result
    
    @staticmethod
    def fit_all_experiments(env, reaction_network: dict) -> dict:
        """Fit reaction network to all experiments simultaneously."""
        results = []
        for exp_name in env.experimental_data.keys():
            result = KineticTools.fit_single_experiment(env, exp_name, reaction_network)
            results.append(result)
        
        successful_results = [r for r in results if r.get('success', False)]
        total_rss = sum(r['rss'] for r in successful_results)
        
        return {
            'individual_fits': results,
            'num_successful': len(successful_results),
            'num_failed': len(results) - len(successful_results),
            'total_rss': total_rss,
            'avg_rss': total_rss / len(successful_results) if successful_results else 1e10
        }
    
    @staticmethod
    def evaluate_phenomenological_trends(env, reaction_network: dict) -> dict:
        """
        Evaluate how well model reproduces experimental trends.
        """
        # Fit all experiments
        fit_results = KineticTools.fit_all_experiments(env, reaction_network)
        
        if fit_results['num_successful'] == 0:
            return {
                'error': 'No successful fits',
                'overall_score': 0.0
            }
        
        # Extract trends
        trends = {
            'c_Ru': {},
            'c_S2O8': {},
            'irradiance': {}
        }
        
        for result in fit_results['individual_fits']:
            if not result.get('success', False):
                continue
            
            exp_name = result['exp_name']
            data = env.experimental_data[exp_name]
            meta = data['metadata']
            
            # Calculate max rate from oxygen curve
            o2 = data['oxygen']
            time = data['time']
            rates = np.gradient(o2, time)
            max_rate = np.max(rates)
            
            # Group by parameters
            for param in ['c_Ru', 'c_S2O8', 'irradiance']:
                param_val = meta.get(param)
                if param_val is not None:
                    if param_val not in trends[param]:
                        trends[param][param_val] = []
                    trends[param][param_val].append(max_rate)
        
        # Score trends
        scores = {}
        scores['ru_trend_score'] = _score_ru_trend(trends['c_Ru'])
        scores['s2o8_trend_score'] = _score_s2o8_trend(trends['c_S2O8'])
        scores['irradiance_trend_score'] = _score_irradiance_trend(trends['irradiance'])
        scores['fit_quality_score'] = 1.0 / (1.0 + fit_results['avg_rss'] / 100)
        
        # Overall score (weighted)
        scores['overall_score'] = (
            0.3 * scores['ru_trend_score'] +
            0.25 * scores['s2o8_trend_score'] +
            0.25 * scores['irradiance_trend_score'] +
            0.2 * scores['fit_quality_score']
        )
        
        return scores
    
    @staticmethod
    def modify_reaction_network(env, modifications: dict) -> dict:
        """
        Apply modifications to reaction network.
        
        modifications can contain:
        - add_reactions: list of reaction dicts to add
        - remove_reactions: list of indices to remove
        - modify_k_ranges: dict of {index: (new_min, new_max)}
        - modify_quantum_yields: dict of {index: (new_min, new_max)}
        """
        new_network = {'reactions': list(env.current_reaction_network['reactions'])}
        
        # Remove reactions (do this first, in reverse order)
        if 'remove_reactions' in modifications:
            indices = sorted(modifications['remove_reactions'], reverse=True)
            for idx in indices:
                if 0 <= idx < len(new_network['reactions']):
                    new_network['reactions'].pop(idx)
        
        # Modify ranges
        if 'modify_k_ranges' in modifications:
            for idx, new_range in modifications['modify_k_ranges'].items():
                if 0 <= idx < len(new_network['reactions']):
                    new_network['reactions'][idx]['k_range'] = tuple(new_range)
        
        if 'modify_quantum_yields' in modifications:
            for idx, new_range in modifications['modify_quantum_yields'].items():
                if 0 <= idx < len(new_network['reactions']):
                    # Ensure quantum yields stay in [0, 1]
                    qy_range = (max(0, new_range[0]), min(1, new_range[1]))
                    new_network['reactions'][idx]['quantum_yield'] = qy_range
        
        # Add reactions (do this last)
        if 'add_reactions' in modifications:
            new_network['reactions'].extend(modifications['add_reactions'])
        
        env.current_reaction_network = new_network
        return new_network
    
    @staticmethod
    def visualize_fit(env, exp_name: str, reaction_network: dict) -> str:
        """Generate text description of fit quality."""
        result = KineticTools.fit_single_experiment(env, exp_name, reaction_network)
        
        if not result.get('success', False):
            return f"Fit failed for {exp_name}: {result.get('error', 'Unknown error')}"
        
        y_exp = result['experimental_curve']
        y_pred = result['y_pred']
        
        # Calculate metrics
        residuals = y_exp - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        description = f"Fit quality for {exp_name}:\n"
        description += f"  - R² = {r2:.4f}\n"
        description += f"  - RMSE = {rmse:.4f} µM\n"
        description += f"  - MAE = {mae:.4f} µM\n"
        description += f"  - RSS = {result['rss']:.2f}\n"
        
        # Qualitative assessment
        if r2 > 0.95:
            description += "  → Excellent fit\n"
        elif r2 > 0.90:
            description += "  → Good fit\n"
        elif r2 > 0.80:
            description += "  → Acceptable fit\n"
        else:
            description += "  → Poor fit - needs improvement\n"
        
        return description


# Helper functions for trend scoring

def _score_ru_trend(rates_vs_ru: dict) -> float:
    """Score Ru concentration trend (should increase then decrease)."""
    if len(rates_vs_ru) < 3:
        return 0.5  # Not enough data
    
    concentrations = np.array(sorted(rates_vs_ru.keys()))
    rates = np.array([np.mean(rates_vs_ru[c]) for c in concentrations])
    
    # Find maximum
    max_idx = np.argmax(rates)
    
    # Check if maximum is in middle range (not at edges)
    if max_idx == 0 or max_idx == len(rates) - 1:
        position_score = 0.3
    else:
        position_score = 1.0
    
    # Check monotonic increase before max
    if max_idx > 0:
        increase_score = float(np.all(np.diff(rates[:max_idx+1]) >= 0))
    else:
        increase_score = 0.5
    
    # Check decrease after max
    if max_idx < len(rates) - 1:
        decrease_score = float(np.all(np.diff(rates[max_idx:]) <= 0))
    else:
        decrease_score = 0.5
    
    return (position_score + increase_score + decrease_score) / 3


def _score_s2o8_trend(rates_vs_s2o8: dict) -> float:
    """Score S2O8 trend (should be monotonic increase with saturation)."""
    if len(rates_vs_s2o8) < 3:
        return 0.5
    
    concentrations = np.array(sorted(rates_vs_s2o8.keys()))
    rates = np.array([np.mean(rates_vs_s2o8[c]) for c in concentrations])
    
    # Check monotonic increase
    monotonic_score = float(np.all(np.diff(rates) >= 0))
    
    # Check for saturation (decreasing slope)
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
    
    # Fit linear model
    slope, intercept, r_value, p_value, std_err = linregress(concentrations, rates)
    
    # Score based on R²
    return max(0, r_value**2)