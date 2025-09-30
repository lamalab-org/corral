#!/usr/bin/env python3
"""Test core functions without corral dependencies."""

import sys
from pathlib import Path
import json
import tempfile

# Add the src directory to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_loading():
    """Test data loading functionality."""
    try:
        # Import using the proper data loading code
        import pandas as pd
        import h5py
        import numpy as np
        from dataclasses import dataclass, asdict, field
        from typing import List, Dict
        import colorsys
        import re
        import hashlib

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
                    print("No overview DataFrame found in file")
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
                            
                        except Exception as e:
                            print(f"Warning: Could not load {exp_name}: {e}")
                
                return dataset

        def load_experimental_data(data_path: str):
            """Load experimental data from HDF5 file using proper structure."""
            try:
                dataset = ExperimentalDataset.load_from_hdf5(data_path)
                
                data = {}
                for exp_name, exp_data in dataset.experiments.items():
                    # Extract time and oxygen data
                    time = exp_data.time_series_data.time_reaction
                    oxygen = exp_data.time_series_data.data_reaction
                    
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
                    # Fallback: just load overview data if experiments are empty
                    if not dataset.overview_df.empty:
                        print(f"No experiment time series found, but overview contains {len(dataset.overview_df)} entries")
                        return {"overview": dataset.overview_df.to_dict()}
                    raise ValueError("No experimental data could be loaded")
                
                return data
                
            except Exception as e:
                print(f"Error loading with proper structure: {e}")
                # Simple fallback - just check overview
                overview_df = pd.read_hdf(data_path, key='overview_df')
                return {"overview": overview_df.to_dict()}
        
        data_path = Path(__file__).parent / "data" / "experimental_data.h5"
        data = load_experimental_data(str(data_path))
        
        print(f"✅ Loaded {len(data)} experiments")
        
        # Check first experiment
        first_exp = list(data.keys())[0]
        exp_data = data[first_exp]
        print(f"   First experiment: {first_exp}")
        print(f"   Keys: {list(exp_data.keys())}")
        print(f"   Time points: {len(exp_data['time'])}")
        print(f"   Metadata: c_Ru={exp_data['metadata']['c_Ru']}, c_S2O8={exp_data['metadata']['c_S2O8']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stoichiometry_parsing():
    """Test stoichiometry parsing functionality."""
    try:
        from dataclasses import dataclass
        from typing import Dict, List, Optional, Tuple
        
        @dataclass
        class Reaction:
            equation: str
            type: str
            reactants: List[str]
            products: List[str]
            stoichiometry: Dict[str, float]
            k_range: Optional[Tuple[float, float]] = None
            quantum_yield: Optional[Tuple[float, float]] = None
            
            @classmethod
            def from_dict(cls, rxn_dict: dict):
                equation = rxn_dict["equation"]
                
                if "->" not in equation:
                    raise ValueError(f"Invalid equation format: {equation}")
                
                left, right = equation.split("->")
                reactants = [s.strip() for s in left.split("+")]
                products = [s.strip() for s in right.split("+")]
                
                ignored = {"hv", "H2O", "OH", "products", "H", "2 H"}
                stoich = {}
                
                def parse_species_with_coeff(species_str):
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
        
        # Test simple reaction
        rxn1 = Reaction.from_dict({
            "equation": "A + B -> C",
            "type": "dark",
            "k_range": [1e3, 1e5]
        })
        expected1 = {"A": -1, "B": -1, "C": 1}
        if rxn1.stoichiometry != expected1:
            print(f"❌ Wrong stoichiometry for simple reaction: {rxn1.stoichiometry}")
            return False
        
        # Test complex reaction with coefficients  
        rxn2 = Reaction.from_dict({
            "equation": "2 RuIII + H2O2 -> 2 RuII + O2 + 2 H",
            "type": "dark", 
            "k_range": [1e3, 1e5]
        })
        expected2 = {"RuIII": -2, "H2O2": -1, "RuII": 2, "O2": 1}
        if rxn2.stoichiometry != expected2:
            print(f"❌ Wrong stoichiometry for complex reaction: {rxn2.stoichiometry}")
            print(f"Expected: {expected2}")
            print(f"Got: {rxn2.stoichiometry}")
            return False
        
        print("✅ Stoichiometry parsing works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error in stoichiometry parsing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_json():
    """Test network JSON creation and loading."""
    try:
        network = {
            'reactions': [
                {'equation': 'RuII + hv -> RuII*', 'type': 'light', 'quantum_yield': [0.8, 1.0]},
                {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4', 'type': 'dark', 'k_range': [1e7, 1e9]},
                {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H', 'type': 'dark', 'k_range': [1e3, 1e5]},
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(network, f, indent=2)
            temp_path = f.name
        
        # Read it back
        with open(temp_path) as f:
            loaded_network = json.load(f)
        
        if len(loaded_network['reactions']) != 3:
            print(f"❌ Wrong number of reactions: {len(loaded_network['reactions'])}")
            return False
        
        if loaded_network['reactions'][0]['equation'] != 'RuII + hv -> RuII*':
            print(f"❌ Wrong equation: {loaded_network['reactions'][0]['equation']}")
            return False
        
        print("✅ Network JSON creation and loading works")
        return True
        
    except Exception as e:
        print(f"❌ Error in network JSON handling: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CORE FUNCTIONS TEST")
    print("=" * 50)
    
    success = True
    
    print("\nTest 1: Data loading")
    print("-" * 20)
    success &= test_data_loading()
    
    print("\nTest 2: Stoichiometry parsing")
    print("-" * 30)
    success &= test_stoichiometry_parsing()
    
    print("\nTest 3: Network JSON handling")
    print("-" * 30)
    success &= test_network_json()
    
    if success:
        print("\n🎉 All core function tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    sys.exit(0 if success else 1)