#!/usr/bin/env python3
"""
Debug script for kinetic fitting flat line issues.

This script runs diagnostic tools to identify why fits are producing flat lines.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from kinetic_fitting.tools import (
    load_experimental_data,
    load_reaction_network,
    save_reaction_network,
    initialize_default_network,
    create_ode_system,
    fit_reaction_network,
    _get_persistent_output_dir,
    _create_fit_plot
)
import numpy as np
import json
import warnings
from scipy.integrate import odeint

def test_ode_direct(data, network, exp_name, reference_params):
    """Test ODE system directly without tool wrapper."""
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
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_test = odeint(
                ode_func, y0, time_exp,
                args=(reference_params, exp_data["metadata"]),
                rtol=1e-6, atol=1e-8,
            )
        
        o2_pred = y_test[:, species_idx["O2"]] if "O2" in species_idx else np.zeros_like(time_exp)
        
        # Calculate metrics
        o2_range = np.max(o2_pred) - np.min(o2_pred)
        o2_max = np.max(o2_pred)
        exp_range = np.max(oxygen_exp) - np.min(oxygen_exp)
        exp_max = np.max(oxygen_exp)
        rss = np.sum((oxygen_exp - o2_pred) ** 2)
        
        # Save diagnostic plot
        try:
            persistent_dir = _get_persistent_output_dir()
            test_plot_path = persistent_dir / f"ode_test_{exp_name}.png"
            _create_fit_plot(
                time_exp, oxygen_exp, o2_pred, f"ODE Test - {exp_name}",
                exp_data["metadata"], save_path=str(test_plot_path)
            )
            plot_saved = str(test_plot_path)
        except Exception as e:
            plot_saved = f"Plot save failed: {e}"
        
        return {
            'species_list': species_list,
            'initial_conditions': dict(zip(species_list, y0)),
            'o2_range': o2_range,
            'o2_max': o2_max,
            'exp_range': exp_range,
            'exp_max': exp_max,
            'rss': rss,
            'plot_path': plot_saved,
            'final_concentrations': dict(zip(species_list, y_test[-1])),
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

def run_debugging_sequence():
    """Run a complete debugging sequence to identify flat line issues."""
    
    # File paths
    data_path = "data/experimental_data.h5"
    network_path = "current_network.json"
    results_path = "fit_results.json"
    
    print("=" * 60)
    print("KINETIC FITTING DEBUG SESSION")
    print("=" * 60)
    
    # Step 1: Check data availability
    print("\n1. EXPERIMENTAL DATA OVERVIEW")
    print("-" * 40)
    try:
        # Load data directly
        data = load_experimental_data(data_path)
        
        # Show overview
        print("Available experimental datasets:\n")
        for i, (exp_name, exp_data) in enumerate(list(data.items())[:10]):
            meta = exp_data["metadata"]
            time = np.array(exp_data["time"])
            oxygen = np.array(exp_data["oxygen"])
            
            print(f"{exp_name}:")
            print(f"  [Ru(bpy)3Cl2]: {meta.get('c_Ru', 'N/A')} µM")
            print(f"  [Na2S2O8]: {meta.get('c_S2O8', 'N/A')} µM")
            print(f"  Power/Irradiance: {meta.get('irradiance', 'N/A')} W/m²")
            print(f"  pH: {meta.get('pH', 'N/A')}")
            print(f"  Data points: {len(time)}")
            print(f"  Time range: {time.min():.1f} - {time.max():.1f} s")
            print(f"  O2 range: {oxygen.min():.3f} - {oxygen.max():.3f} µM\n")
        
        if len(data) > 10:
            print(f"... and {len(data) - 10} more experiments")
        
        sample_exp = list(data.keys())[0] if data else None
        print(f"\nUsing sample experiment: {sample_exp}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure experimental_data.h5 exists in the data/ directory")
        return
    
    if not sample_exp:
        print("No experimental data found!")
        return
    
    # Step 2: Test ODE system with reference parameters
    print("\n\n2. ODE SYSTEM TEST WITH REFERENCE PARAMETERS")
    print("-" * 50)
    print("Testing if ODE system works with known good parameters...")
    
    # Optimized parameters for this experimental system (from successful fit: R²=0.993)
    reference_params = {"qy_0": 0.994, "k_1": 56.3, "k_2": 0.773, "k_3": 0.077, "k_4": 0.348, "k_5": 0.010}
    
    try:
        # Load or create network
        try:
            network = load_reaction_network(network_path)
        except FileNotFoundError:
            network = initialize_default_network()
            save_reaction_network(network_path, network)
        
        # Test ODE system
        ode_result = test_ode_direct(data, network, sample_exp, reference_params)
        
        if ode_result['success']:
            print(f"ODE System Test Results for {sample_exp}:")
            print(f"  Species found: {ode_result['species_list']}")
            print(f"  Initial conditions: {ode_result['initial_conditions']}")
            print(f"  Reference parameters used: {reference_params}")
            print(f"  ")
            print(f"  Results:")
            print(f"    O₂ predicted range: {ode_result['o2_range']:.6f} µM")  
            print(f"    O₂ predicted max: {ode_result['o2_max']:.6f} µM")
            print(f"    Experimental range: {ode_result['exp_range']:.3f} µM")
            print(f"    Experimental max: {ode_result['exp_max']:.3f} µM") 
            print(f"    RSS: {ode_result['rss']:.2f}")
            print(f"    Ratio pred/exp range: {ode_result['o2_range']/ode_result['exp_range']:.6f}" if ode_result['exp_range'] > 0 else "    Ratio: undefined")
            print(f"  ")
            print(f"  Diagnosis:")
            
            if ode_result['o2_max'] < 0.001:
                print("    CRITICAL: Essentially zero O₂ production - ODE system not working")
            elif ode_result['o2_range'] < 0.01 * ode_result['exp_range']:
                print("    CRITICAL: Flat line output - parameters/network issue")
            elif ode_result['o2_max'] < 0.1 * ode_result['exp_max']:
                print("    Issue: O₂ production too low - parameter scaling problem")
            else:
                print("    Good: O₂ production in reasonable range")
            
            print(f"  ")
            print(f"  Test plot saved to: {ode_result['plot_path']}")
            print(f"  Final concentrations: {ode_result['final_concentrations']}")
        else:
            print(f"ODE test failed: {ode_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"ODE test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Test optimization with reference starting point
    print("\n\n3. OPTIMIZATION TEST WITH REFERENCE SEED")
    print("-" * 45)
    print("Testing if optimization can improve from good starting point...")
    
    try:
        # Test optimization with reference parameters
        opt_result = fit_reaction_network(
            np.array(data[sample_exp]["time"]),
            np.array(data[sample_exp]["oxygen"]),
            network,
            data[sample_exp]["metadata"],
            reference_params=reference_params,
        )
        
        print(f"Reference-seeded optimization for {sample_exp}:")
        print(f"  Success: {opt_result['success']}")
        print(f"  R² = {opt_result['r2']:.4f}")
        print(f"  RSS = {opt_result['rss']:.2f}")
        print(f"  O₂ max: {opt_result.get('o2_max', 'N/A'):.4f} µM")
        print(f"  O₂ range: {opt_result.get('o2_range', 'N/A'):.4f} µM")
        if not opt_result['success']:
            print(f"  Failure reason: {opt_result.get('failure_reason', 'Unknown')}")
        
    except Exception as e:
        print(f"Reference-seeded optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Test multiple experiments quickly
    print("\n\n4. MULTI-EXPERIMENT TEST")
    print("-" * 30)
    print("Testing ODE system with multiple experiments...")
    
    exp_names = list(data.keys())[:3]  # Test first 3 experiments
    for exp_name in exp_names:
        print(f"\n--- Testing {exp_name} ---")
        try:
            result = test_ode_direct(data, network, exp_name, reference_params)
            if result['success']:
                print(f"  O₂ max: {result['o2_max']:.6f} µM")
                print(f"  RSS: {result['rss']:.2f}")
                if result['o2_max'] < 0.001:
                    print("  ❌ CRITICAL: Zero O₂ production")
                elif result['o2_range'] < 0.01 * result['exp_range']:
                    print("  ❌ CRITICAL: Flat line output") 
                elif result['o2_max'] < 0.1 * result['exp_max']:
                    print("  ⚠️  Issue: O₂ production too low")
                else:
                    print("  ✅ Good: O₂ production reasonable")
            else:
                print(f"  ❌ Test failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    print("\n\n" + "=" * 60)
    print("DEBUG SESSION COMPLETE")
    print("=" * 60)
    print("\nCheck the persistent output directory for plots and detailed logs.")
    print("Files saved:")
    print("  - ode_test_{experiment}.png - ODE system test plots")
    print("  - reference_fit_{experiment}.json - Detailed fit results")
    print("  - reference_fit_plot_{experiment}.png - Reference fit plots")
    print("  - fit_params_{timestamp}.json - Parameter logging")
    

if __name__ == "__main__":
    run_debugging_sequence()