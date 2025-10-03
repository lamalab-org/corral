#!/usr/bin/env python3
"""Simple test to check parameter mismatch."""

import sys, json
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import without matplotlib
def test_parameters():
    # Load the network
    network_path = "current_network.json"
    with open(network_path, 'r') as f:
        network = json.load(f)
    
    print("Current network reactions:")
    param_names = []
    for i, rxn in enumerate(network["reactions"]):
        rxn_type = rxn["type"]
        param_name = f"qy_{i}" if rxn_type == "light" else f"k_{i}"
        param_names.append(param_name)
        print(f"  {i}: {rxn['equation']} ({rxn_type}) -> {param_name}")
    
    print(f"\nExpected parameters: {param_names}")
    
    # Reference parameters from debug script
    ref_params = {"qy_0": 1.0, "k_1": 59, "k_2": 0.03, "k_3": 59, "k_4": 0.005, "k_5": 0.003}
    print(f"Reference parameters: {list(ref_params.keys())}")
    
    # Check for mismatches
    missing = [p for p in param_names if p not in ref_params]
    extra = [p for p in ref_params if p not in param_names]
    
    if missing:
        print(f"\nERROR: Missing parameters in reference: {missing}")
    if extra:
        print(f"ERROR: Extra parameters in reference: {extra}")
    
    if not missing and not extra:
        print("\n✅ Parameters match!")
    else:
        print("\n❌ Parameter mismatch found!")
        
    return not missing and not extra

if __name__ == "__main__":
    test_parameters()