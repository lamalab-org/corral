# Kinetic Fitting Environment

LLM agent environment for automated kinetic model optimization in photocatalysis research.

## Overview

This environment enables LLM agents to:
- Fit reaction networks to experimental oxygen evolution data from real photocatalysis experiments
- Optimize both individual fit quality (RSS, R²) and phenomenological trends
- Iteratively modify reaction mechanisms to improve model accuracy
- Provide mechanistic insights into photocatalytic water oxidation

## Architecture

### Core Components

```
kinetic_fitting/
├── src/kinetic_fitting/
│   ├── env.py              # Corral environment setup and integration
│   ├── tools.py            # Agent tools for data loading and fitting
├── tests/                  # Pytest test suite
├── tasks/                  # Task definitions for Corral
└── data/                   # Real experimental HDF5 data
```

### Data Architecture

**Real Experimental Data**: 61 experiments from photocatalytic water oxidation
- **Format**: HDF5 with time-series data and metadata
- **Structure**: 
  - `overview_df`: Experimental conditions and summary results
  - `{experiment}/time_series_data`: Real oxygen evolution curves (0.293-81.897 µM O₂)
  - Time ranges: 0-315 seconds with 78-94 data points per experiment

**Data Loading Pipeline**:
1. Load HDF5 file with real experimental time-series data  
2. Extract metadata from overview dataframe (reliable units)
3. Combine real time-series with properly converted metadata (M → µM)
4. Standardize format for agent consumption

### Tool Architecture

**File-Based Stateless Tools**: All tools operate on file paths, not environment state
- `describe_experimental_data(data_path)` → Load and summarize real experimental data
- `get_current_network(network_path)` → Load reaction network from JSON
- `fit_single_experiment(data_path, network_path, results_path, exp_name)` → Fit kinetic model
- `fit_all_experiments(data_path, network_path, results_path)` → Batch fitting
- `evaluate_phenomenological_trends(data_path, network_path, results_path)` → Score trends
- `modify_reaction_network(network_path, modifications)` → Update reaction network
- `analyze_fit_with_vision(results_path)` → Vision-based fit analysis


### Kinetic Modeling Architecture

**Reaction Network Representation**:
```json
{
  "reactions": [
    {"equation": "RuII + hv -> RuII*", "type": "light", "quantum_yield": [0.8, 1.0]},
    {"equation": "RuII* + S2O8 -> RuIII + SO4_rad + SO4", "type": "dark", "k_range": [1e7, 1e9]}
  ]
}
```

**ODE System Generation**:
- Automatic stoichiometry parsing from reaction equations
- Species balance equation generation
- Light/dark reaction handling with quantum yields
- Parameter optimization via differential evolution

**Phenomenological Scoring**:
- **[Ru] trends**: Rate peaks at 5-10 µM, then decreases
- **[S₂O₈²⁻] trends**: Monotonic increase with saturation  
- **Irradiance trends**: Linear relationship
- Combined score calculation with experimental validation

## Chemical System

- **Catalyst**: Ru(bpy)₃²⁺ photocatalyst (0-25 µM range)
- **Oxidant**: Persulfate (S₂O₈²⁻) (0-8000 µM range)
- **Product**: O₂ evolution (real experimental measurements)
- **Conditions**: pH 7-9.6, 200-1600 W/m² irradiance
- **Challenge**: Complex reaction network with competing pathways and catalyst deactivation

## Goals

Find minimal reaction network that reproduces experimental trends:
- **[Ru] dependence**: Rate peaks at 5-10 µM, then decreases (catalyst deactivation)
- **[S₂O₈²⁻] dependence**: Monotonic increase with saturation
- **Irradiance dependence**: Linear relationship

## Tools Available

### Data Tools
- `describe_experimental_data`: List 61 real datasets with metadata and oxygen evolution curves
- `get_current_network`: Show current reaction network JSON

### Fitting Tools  
- `fit_single_experiment`: Fit network to one experiment with ODE integration
- `fit_all_experiments`: Fit network to all 61 experiments with batch optimization
- `evaluate_phenomenological_trends`: Score trend reproduction against experimental patterns

### Modification Tools
- `modify_reaction_network`: Add/remove/modify reactions with automatic validation
- `analyze_fit_with_vision`: Use vision model to analyze fit quality plots


## Development

### Setup
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run tests
pytest
```


### Usage
```python
from kinetic_fitting.env import create_environments

# Create environment  
envs = create_environments('tasks/kinetic_fitting_tasks.json')
env = envs['kinetic_fitting']

# Load and examine data
result = env.call_tool('describe_experimental_data', {'data_path': 'data/experimental_data.h5'})
print(result.result)  # Shows 61 real experiments

# Fit kinetic model
result = env.call_tool('fit_single_experiment', {
    'data_path': 'data/experimental_data.h5',
    'network_path': 'network.json', 
    'results_path': 'results/',
    'exp_name': 'MRG-059-ZM-1-1'
})
```