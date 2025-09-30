from corral.core import TaskEnvironment
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

class KineticFittingEnv(TaskEnvironment):
    def __init__(self, data_path: Optional[str | Path] = None):
        super().__init__()
        
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "experimental_data.h5"
        
        self.data_path = Path(data_path)
        self.experimental_data = self._load_experimental_data()
        self.current_reaction_network = self._initialize_default_network()
        self.fit_history = []
        self.best_score = 0.0
        
    def _load_experimental_data(self):
        """Load all experimental datasets and metadata from HDF5 file."""
        data = {}
        
        with h5py.File(self.data_path, 'r') as f:
            for exp_name in f.keys():
                try:
                    # Load data_corrected (2D array: column 0 = time, column 1 = oxygen)
                    data_corrected = np.array(f[exp_name]['datasets']['data_corrected'])
                    
                    if data_corrected.ndim != 2 or data_corrected.shape[1] != 2:
                        print(f"Warning: Unexpected shape for {exp_name}: {data_corrected.shape}")
                        continue
                    
                    time = data_corrected[:, 0]
                    oxygen = data_corrected[:, 1]
                    
                    # Load metadata - stored as individual datasets in experiment_metadata group
                    metadata = {}
                    if 'experiment_metadata' in f[exp_name]:
                        meta_group = f[exp_name]['experiment_metadata']
                        
                        for key in meta_group.keys():
                            dataset = meta_group[key]
                            if isinstance(dataset, h5py.Dataset):
                                value = dataset[()]
                                # Convert bytes to string if necessary
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                                # Convert numpy types to Python native types
                                elif isinstance(value, (np.integer, np.floating)):
                                    value = value.item()
                                metadata[key] = value
                    
    
                    standardized_metadata = {
                        'c_Ru': metadata.get('ru_concentration'),
                        'c_S2O8': metadata.get('oxidant_concentration'),
                        'power_output': metadata.get('power_output'),
                        'pH': metadata.get('pH'),
                        'buffer_concentration': metadata.get('buffer_concentration'),
                        'buffer_used': metadata.get('buffer_used'),
                        'experiment_name': metadata.get('experiment_name'),
                        'color': metadata.get('color'),
                    }
                    
                    # ToDo: power_output might need to be converted to irradiance
                    # For now, we'll use power_output directly
                    standardized_metadata['irradiance'] = standardized_metadata['power_output']
                    
                    data[exp_name] = {
                        'time': time,
                        'oxygen': oxygen,
                        'metadata': standardized_metadata,
                        'raw_metadata': metadata  # Keep original for reference
                    }
                    
                except KeyError as e:
                    print(f"Warning: Missing key in {exp_name}: {e}")
                except Exception as e:
                    print(f"Warning: Could not load {exp_name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if not data:
            raise ValueError("No experimental data could be loaded from HDF5 file")
        
        print(f"Successfully loaded {len(data)} experiments")
        return data
    
    def _initialize_default_network(self):
        """Initialize with reactions from Akhtar 2016 (Scheme 2b)."""
        # ToDo: To be checked and validated
        return {
            'reactions': [
                # Step 1: Photoexcitation (MLCT)
                {'equation': 'RuII + hv -> RuII*', 
                 'type': 'light', 'quantum_yield': (0.8, 1.0)},
                
                # Steps 2-3: Oxidative quenching
                {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4',
                 'type': 'dark', 'k_range': (1e7, 1e9)},
                {'equation': 'RuII + SO4_rad -> RuIII + SO4',
                 'type': 'dark', 'k_range': (1e8, 1e10)},
                
                # Step 6-7: Dark pathway (OH- oxidation)
                {'equation': 'RuIII + OH -> RuII + OH_rad',
                 'type': 'dark', 'k_range': (1e3, 1e5)},
                {'equation': '2 OH_rad -> H2O2',
                 'type': 'dark', 'k_range': (1e9, 1e10)},
                
                # Step 9: H2O2 oxidation to O2
                {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H',
                 'type': 'dark', 'k_range': (1e3, 1e5)},
                
                # Steps 10-12: Light-induced decomposition
                {'equation': 'RuIII + hv -> RuIII*',
                 'type': 'light', 'quantum_yield': (0.8, 1.0)},
                {'equation': 'RuIII* + S2O8 -> RuIV_intermediate',
                 'type': 'dark', 'k_range': (1e7, 1e9)},
                
                # Step 13: Dimerization
                {'equation': '2 RuIV_intermediate -> Ru_Dimer_active',
                 'type': 'dark', 'k_range': (1e5, 1e7)},
                
                # Step 16: Oligomerization
                {'equation': 'RuIV_intermediate + RuIV_intermediate -> Ru_oligomer_inactive',
                 'type': 'dark', 'k_range': (1e6, 1e8)},
                
                # Step 8: OH radical decomposition
                {'equation': 'OH_rad + RuII -> decomposed_Ru',
                 'type': 'dark', 'k_range': (1e8, 1e10)},
            ]
        }
    
    def get_state(self):
        return {
            'current_network': self.current_reaction_network,
            'num_experiments': len(self.experimental_data),
            'best_score': self.best_score,
            'num_iterations': len(self.fit_history)
        }
    
    def reset(self):
        self.current_reaction_network = self._initialize_default_network()
        self.fit_history = []
        self.best_score = 0.0
        return self.get_state()