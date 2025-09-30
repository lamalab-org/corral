"""Kinetic fitting environment."""

from pathlib import Path
from typing import Optional
import h5py
import numpy as np

from corral.backend.env import Environment
from corral.backend.task import TaskDefinition, TaskGroup
from tools import (
    describe_experimental_data,
    fit_single_experiment,
    fit_all_experiments,
    evaluate_phenomenological_trends,
    modify_reaction_network,
    analyze_fit_with_vision,
)


class KineticFittingEnv(Environment):
    """Environment for fitting kinetic models to photocatalytic water oxidation data."""
    
    def __init__(self, task_id: str,  task_group: TaskGroup,
        work_dir: str, data_path: Optional[str | Path] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "experimental_data.h5"
        
        self.data_path = Path(data_path)
        self.experimental_data = self._load_experimental_data()
        self.current_reaction_network = self._initialize_default_network()
        self.fit_history = []
        self.best_score = 0.0
        
        super().__init__(f"{task_group.group_id}_{task_id}", base_work_dir=work_dir)
        
        # Register tools
        self.add_tool(describe_experimental_data)
        self.add_tool(fit_single_experiment)
        self.add_tool(fit_all_experiments)
        self.add_tool(evaluate_phenomenological_trends)
        self.add_tool(modify_reaction_network)
        self.add_tool(analyze_fit_with_vision)
    
    def _load_experimental_data(self):
        """Load all experimental datasets and metadata from HDF5 file."""
        data = {}
        
        with h5py.File(self.data_path, 'r') as f:
            for exp_name in f.keys():
                try:
                    data_corrected = np.array(f[exp_name]['datasets']['data_corrected'])
                    
                    if data_corrected.ndim != 2 or data_corrected.shape[1] != 2:
                        continue
                    
                    time = data_corrected[:, 0]
                    oxygen = data_corrected[:, 1]
                    
                    metadata = {}
                    if 'experiment_metadata' in f[exp_name]:
                        meta_group = f[exp_name]['experiment_metadata']
                        for key in meta_group.keys():
                            dataset = meta_group[key]
                            if isinstance(dataset, h5py.Dataset):
                                value = dataset[()]
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                                elif isinstance(value, (np.integer, np.floating)):
                                    value = value.item()
                                metadata[key] = value
                    
                    standardized_metadata = {
                        'c_Ru': metadata.get('ru_concentration'),
                        'c_S2O8': metadata.get('oxidant_concentration'),
                        'power_output': metadata.get('power_output'),
                        'pH': metadata.get('pH'),
                        'irradiance': metadata.get('power_output'),  # Use power as irradiance proxy
                    }
                    
                    data[exp_name] = {
                        'time': time,
                        'oxygen': oxygen,
                        'metadata': standardized_metadata,
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not load {exp_name}: {e}")
        
        if not data:
            raise ValueError("No experimental data could be loaded")
        
        return data
    
    def _initialize_default_network(self):
        """Initialize with reactions from Akhtar 2016."""
        return {
            'reactions': [
                {'equation': 'RuII + hv -> RuII*', 'type': 'light', 'quantum_yield': (0.8, 1.0)},
                {'equation': 'RuII* + S2O8 -> RuIII + SO4_rad + SO4', 'type': 'dark', 'k_range': (1e7, 1e9)},
                {'equation': 'RuII + SO4_rad -> RuIII + SO4', 'type': 'dark', 'k_range': (1e8, 1e10)},
                {'equation': 'RuIII + OH -> RuII + OH_rad', 'type': 'dark', 'k_range': (1e3, 1e5)},
                {'equation': '2 OH_rad -> H2O2', 'type': 'dark', 'k_range': (1e9, 1e10)},
                {'equation': '2 RuIII + H2O2 -> 2 RuII + O2 + 2 H', 'type': 'dark', 'k_range': (1e3, 1e5)},
                {'equation': 'RuIII + hv -> RuIII*', 'type': 'light', 'quantum_yield': (0.8, 1.0)},
                {'equation': 'RuIII* + S2O8 -> RuIV_intermediate', 'type': 'dark', 'k_range': (1e7, 1e9)},
                {'equation': '2 RuIV_intermediate -> Ru_Dimer_active', 'type': 'dark', 'k_range': (1e5, 1e7)},
                {'equation': 'RuIV_intermediate + RuIV_intermediate -> Ru_oligomer_inactive', 'type': 'dark', 'k_range': (1e6, 1e8)},
                {'equation': 'OH_rad + RuII -> decomposed_Ru', 'type': 'dark', 'k_range': (1e8, 1e10)},
            ]
        }
    
    def get_task_prompt(self) -> str:
        return """Develop and refine a kinetic model for photocatalytic water oxidation using Ru(bpy)3²⁺.

Your goal is to find a reaction network that:
1. Fits individual experimental O2 evolution curves well (low RSS, high R²)
2. Reproduces phenomenological trends:
   - [Ru] dependence: rate increases to max ~5-10 µM, then decreases
   - [S2O8] dependence: monotonic increase with saturation
   - Irradiance dependence: linear relationship
3. Is physically meaningful (quantum yields 0-1, reasonable rate constants)

Use the available tools to analyze fits, modify the reaction network, and evaluate performance.
Focus on maximizing the overall phenomenological trend score."""
    
    def score(self) -> float:
        """Score based on best phenomenological trend score achieved."""
        return self.best_score