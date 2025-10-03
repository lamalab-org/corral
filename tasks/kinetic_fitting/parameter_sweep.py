#!/usr/bin/env python3
"""
Parameter sweep script to test different parameter combinations and identify
which parameters lead to flat line vs realistic O2 evolution.
"""

import os
import sys
import json
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_parameter_set(exp_name, params_dict, data_path, network_path):
    """Test a single parameter set and return key metrics."""
    
    from kinetic_fitting.tools import test_ode_system_with_reference_params
    
    params_json = json.dumps(params_dict)
    
    try:
        result = test_ode_system_with_reference_params(
            data_path, network_path, exp_name, params_json
        )
        
        # Extract key metrics from result
        o2_max = 0
        o2_range = 0
        rss = float('inf')
        diagnosis = "Unknown"
        
        lines = result.split('\n')
        for line in lines:
            if 'O₂ predicted max:' in line:
                try:
                    o2_max = float(line.split(':')[1].split('µM')[0].strip())
                except:
                    pass
            elif 'O₂ predicted range:' in line:
                try:
                    o2_range = float(line.split(':')[1].split('µM')[0].strip())
                except:
                    pass
            elif 'RSS:' in line:
                try:
                    rss = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'CRITICAL:' in line or 'Issue:' in line or 'Good:' in line:
                diagnosis = line.strip()
        
        return {
            'o2_max': o2_max,
            'o2_range': o2_range, 
            'rss': rss,
            'diagnosis': diagnosis,
            'success': 'Good:' in diagnosis
        }
        
    except Exception as e:
        return {
            'o2_max': 0,
            'o2_range': 0,
            'rss': float('inf'),
            'diagnosis': f"Error: {e}",
            'success': False
        }

def parameter_sweep():
    """Run parameter sweep to identify good vs bad parameter ranges."""
    
    from kinetic_fitting.tools import load_experimental_data
    
    # File paths
    data_path = "data/experimental_data.h5"
    network_path = "current_network.json"
    
    # Get experiment name
    try:
        data = load_experimental_data(data_path)
        exp_name = list(data.keys())[0]
        print(f"Testing with experiment: {exp_name}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Define parameter sets to test
    parameter_sets = [
        # Literature reference values
        {
            "name": "Literature Reference",
            "params": {"qy_0": 1.0, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 0.005, "k_5": 0.003}
        },
        # Higher k values
        {
            "name": "Higher k values",
            "params": {"qy_0": 1.0, "k_1": 100, "k_2": 0.1, "k_3": 100, "k_4": 0.01, "k_5": 0.01}
        },
        # Lower k values  
        {
            "name": "Lower k values",
            "params": {"qy_0": 1.0, "k_1": 10, "k_2": 0.01, "k_3": 10, "k_4": 0.001, "k_5": 0.001}
        },
        # Lower quantum yields
        {
            "name": "Lower quantum yields", 
            "params": {"qy_0": 0.1, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 0.005, "k_5": 0.003}
        },
        # Very small k values (potential issue)
        {
            "name": "Very small k values",
            "params": {"qy_0": 1.0, "k_1": 1, "k_2": 0.001, "k_3": 1, "k_4": 0.0001, "k_5": 0.0001}
        },
        # Only photoexcitation (minimal network)
        {
            "name": "Minimal network test",
            "params": {"qy_0": 1.0, "k_1": 0, "k_2": 0, "k_3": 0, "k_4": 0, "k_5": 0}
        },
        # High H2O2 conversion
        {
            "name": "Fast H2O2 conversion",
            "params": {"qy_0": 1.0, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 1.0, "k_5": 0.003}
        }
    ]
    
    print("PARAMETER SWEEP RESULTS")
    print("=" * 80)
    print(f"{'Set Name':<25} {'O₂ Max':<10} {'O₂ Range':<12} {'RSS':<12} {'Status'}")
    print("-" * 80)
    
    results = []
    for param_set in parameter_sets:
        name = param_set["name"]
        params = param_set["params"]
        
        result = test_parameter_set(exp_name, params, data_path, network_path)
        results.append({**param_set, **result})
        
        # Format output
        o2_max_str = f"{result['o2_max']:.4f}" if result['o2_max'] < 1000 else f"{result['o2_max']:.2e}"
        o2_range_str = f"{result['o2_range']:.4f}" if result['o2_range'] < 1000 else f"{result['o2_range']:.2e}"
        rss_str = f"{result['rss']:.2f}" if result['rss'] < 1000 else f"{result['rss']:.2e}"
        status = "✓ GOOD" if result['success'] else "✗ BAD"
        
        print(f"{name:<25} {o2_max_str:<10} {o2_range_str:<12} {rss_str:<12} {status}")
    
    print("\n" + "=" * 80)
    print("DETAILED DIAGNOSES")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Parameters: {result['params']}")
        print(f"  {result['diagnosis']}")
    
    # Find best performing set
    successful_sets = [r for r in results if r['success']]
    if successful_sets:
        best = min(successful_sets, key=lambda x: x['rss'])
        print(f"\n🏆 BEST PERFORMING PARAMETERS:")
        print(f"   Set: {best['name']}")
        print(f"   Parameters: {best['params']}")
        print(f"   RSS: {best['rss']:.2f}")
    else:
        print(f"\n❌ NO SUCCESSFUL PARAMETER SETS FOUND")
        print("   This indicates a fundamental issue with the ODE system or network structure.")

if __name__ == "__main__":
    parameter_sweep()