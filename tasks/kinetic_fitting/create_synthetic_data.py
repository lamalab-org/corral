#!/usr/bin/env python3
"""Create synthetic experimental data for testing."""

import h5py
import numpy as np
from pathlib import Path

def create_synthetic_oxygen_evolution_curve(time, c_ru, c_s2o8, irradiance, pH=7.0):
    """Create synthetic oxygen evolution curve with realistic kinetics."""
    
    # Parameters that depend on conditions
    # Ru concentration effect: optimum around 5-10 µM, then decreases (catalyst deactivation)
    if c_ru <= 10:
        ru_factor = c_ru / 10.0  # Linear increase up to 10 µM
    else:
        ru_factor = 10.0 / c_ru  # Decrease above 10 µM due to deactivation
    
    # S2O8 concentration effect: monotonic increase with saturation
    s2o8_factor = c_s2o8 / (c_s2o8 + 1000)  # Michaelis-Menten like saturation
    
    # Irradiance effect: linear relationship
    irr_factor = irradiance / 1000.0  # Normalized to 1000 W/m²
    
    # Overall rate constant
    k_eff = 0.05 * ru_factor * s2o8_factor * irr_factor
    
    # Induction period (catalyst activation)
    induction_time = 50 + 20 * np.random.normal()  # ~50s with some variation
    induction_factor = 1 / (1 + np.exp(-(time - induction_time) / 10))
    
    # Exponential rise to plateau with some curvature
    max_oxygen = 15 + 5 * np.random.normal()  # ~15 µM max with variation
    oxygen = max_oxygen * (1 - np.exp(-k_eff * time)) * induction_factor
    
    # Add some realistic noise
    noise = 0.1 * np.random.normal(size=len(time))
    oxygen = np.maximum(0, oxygen + noise)  # Ensure non-negative
    
    return oxygen

def create_synthetic_dataset():
    """Create a synthetic HDF5 dataset with realistic experimental conditions."""
    
    # Create time points (0 to 300 seconds)
    time = np.linspace(0, 300, 150)
    
    # Define experimental conditions that reproduce the desired trends
    conditions = [
        # Ru concentration series (varying [Ru] at fixed [S2O8] and irradiance)
        {'c_Ru': 1.0, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 2.5, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 5.0, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 10.0, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 15.0, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 25.0, 'c_S2O8': 3000, 'irradiance': 1000, 'pH': 7.0},
        
        # S2O8 concentration series (varying [S2O8] at fixed [Ru] and irradiance)
        {'c_Ru': 7.5, 'c_S2O8': 500, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 1000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 2000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 4000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 6000, 'irradiance': 1000, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 8000, 'irradiance': 1000, 'pH': 7.0},
        
        # Irradiance series (varying irradiance at fixed [Ru] and [S2O8])
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 200, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 400, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 600, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 800, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 1200, 'pH': 7.0},
        {'c_Ru': 7.5, 'c_S2O8': 3000, 'irradiance': 1600, 'pH': 7.0},
    ]
    
    # Create HDF5 file
    output_path = Path("data/experimental_data.h5")
    
    with h5py.File(output_path, 'w') as f:
        for i, cond in enumerate(conditions):
            # Create experiment name
            exp_name = f"SYNTH-EXP-{i+1:02d}"
            
            # Generate synthetic oxygen evolution data
            oxygen = create_synthetic_oxygen_evolution_curve(
                time, cond['c_Ru'], cond['c_S2O8'], cond['irradiance'], cond['pH']
            )
            
            # Create experiment group
            exp_group = f.create_group(exp_name)
            
            # Create datasets group and add time-series data
            datasets_group = exp_group.create_group('datasets')
            data_corrected = np.column_stack([time, oxygen])
            datasets_group.create_dataset('data_corrected', data=data_corrected)
            
            # Create experiment metadata
            meta_group = exp_group.create_group('experiment_metadata')
            meta_group.create_dataset('ru_concentration', data=cond['c_Ru'])
            meta_group.create_dataset('oxidant_concentration', data=cond['c_S2O8'])
            meta_group.create_dataset('power_output', data=cond['irradiance'])
            meta_group.create_dataset('pH', data=cond['pH'])
            
            print(f"Created {exp_name}: [Ru]={cond['c_Ru']} µM, [S2O8]={cond['c_S2O8']} µM, "
                  f"Irr={cond['irradiance']} W/m², Max O2={oxygen.max():.2f} µM")
    
    print(f"\nSynthetic dataset created with {len(conditions)} experiments")
    print(f"Saved to: {output_path.absolute()}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    create_synthetic_dataset()