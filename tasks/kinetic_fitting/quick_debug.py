#!/usr/bin/env python3
"""
Quick debug script to test a single experiment with reference parameters.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def quick_debug_test(exp_name=None, custom_params=None):
    """Quick test of ODE system with reference parameters."""
    
    from kinetic_fitting.tools import (
        load_experimental_data,
        load_reaction_network,
        save_reaction_network,
        initialize_default_network,
        create_ode_system,
        _get_persistent_output_dir,
        _create_fit_plot
    )
    import numpy as np
    import json
    import warnings
    from scipy.integrate import odeint
    
    # File paths
    data_path = "data/experimental_data.h5"
    network_path = "current_network.json"
    
    # Get experiment name
    if exp_name is None:
        try:
            data = load_experimental_data(data_path)
            exp_name = list(data.keys())[0]
            print(f"Using first experiment: {exp_name}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    # Default reference parameters for Akhtar network (moderate starting values to avoid overflow)
    if custom_params is None:
        ref_params = {
            "qy_0": 0.5, "k_1": 1e6, "k_2": 1e7, "k_3": 1e2, "k_4": 1e7, 
            "k_5": 1e2, "qy_6": 0.5, "k_7": 1e6, "k_8": 1e4, "k_9": 1e5, "k_10": 1e7
        }
    else:
        try:
            ref_params = json.loads(custom_params) if isinstance(custom_params, str) else custom_params
        except json.JSONDecodeError as e:
            print(f"Error parsing custom parameters: {e}")
            return
    
    print(f"Testing experiment: {exp_name}")
    print(f"Reference parameters: {ref_params}")
    print("-" * 50)
    
    try:
        # Load data and network
        data = load_experimental_data(data_path)
        
        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            network = initialize_default_network()
            save_reaction_network(network_path, network)
        
        # Get experiment data
        exp_data = data[exp_name]
        time_exp = np.array(exp_data["time"])
        oxygen_exp = np.array(exp_data["oxygen"])
        
        # Create ODE system
        ode_func, species_list, species_idx = create_ode_system(network, exp_data["metadata"])
        
        # Set initial conditions
        y0 = np.zeros(len(species_list))
        if "RuII" in species_idx:
            y0[species_idx["RuII"]] = exp_data["metadata"].get("c_Ru", 10)
        if "S2O8" in species_idx:
            y0[species_idx["S2O8"]] = exp_data["metadata"].get("c_S2O8", 6000)
        
        # Test ODE integration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_test = odeint(
                ode_func, y0, time_exp,
                args=(ref_params, exp_data["metadata"]),
                rtol=1e-6, atol=1e-8,
            )
        
        o2_pred = y_test[:, species_idx["O2"]] if "O2" in species_idx else np.zeros_like(time_exp)
        
        # Calculate metrics
        o2_range = np.max(o2_pred) - np.min(o2_pred)
        o2_max = np.max(o2_pred)
        exp_range = np.max(oxygen_exp) - np.min(oxygen_exp)
        exp_max = np.max(oxygen_exp)
        rss = np.sum((oxygen_exp - o2_pred) ** 2)
        
        # Report results
        print("RESULTS:")
        print(f"  Species found: {species_list}")
        print(f"  Initial conditions: {dict(zip(species_list, y0))}")
        print(f"  O₂ predicted range: {o2_range:.6f} µM")  
        print(f"  O₂ predicted max: {o2_max:.6f} µM")
        print(f"  Experimental range: {exp_range:.3f} µM")
        print(f"  Experimental max: {exp_max:.3f} µM") 
        print(f"  RSS: {rss:.2f}")
        print(f"  Ratio pred/exp range: {o2_range/exp_range:.6f}" if exp_range > 0 else "  Ratio: undefined")
        
        # Diagnosis
        print("\nDIAGNOSIS:")
        if o2_max < 0.001:
            print("  ❌ CRITICAL: Essentially zero O₂ production - ODE system not working")
        elif o2_range < 0.01 * exp_range:
            print("  ❌ CRITICAL: Flat line output - parameters/network issue")
        elif o2_max < 0.1 * exp_max:
            print("  ⚠️  Issue: O₂ production too low - parameter scaling problem")
        else:
            print("  ✅ Good: O₂ production in reasonable range")
        
        # Save diagnostic plot
        try:
            persistent_dir = _get_persistent_output_dir()
            test_plot_path = persistent_dir / f"quick_debug_{exp_name}.png"
            _create_fit_plot(
                time_exp, oxygen_exp, o2_pred, f"Quick Debug - {exp_name}",
                exp_data["metadata"], save_path=str(test_plot_path)
            )
            print(f"\n📊 Plot saved to: {test_plot_path}")
        except Exception as e:
            print(f"\n⚠️  Plot save failed: {e}")
        
        print(f"\n🔬 Final concentrations: {dict(zip(species_list, y_test[-1]))}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # You can customize these parameters
    experiment_name = None  # Will use first experiment if None
    custom_parameters = None  # Will use default literature values if None
    
    # Example custom parameters:
    # custom_parameters = '{"qy_0": 0.8, "k_1": 50, "k_2": 0.05, "k_3": 60, "k_4": 0.004, "k_5": 0.006}'
    
    quick_debug_test(experiment_name, custom_parameters)