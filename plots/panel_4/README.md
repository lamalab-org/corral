# Panel 4: IRT Results Plotting

Unified plotting script for IRT (Item Response Theory) model results.

## Overview

`plot_irt_results.py` generates all visualizations for the latent factor modeling analysis, combining capability estimates (θ_K, θ_R) and agent model effects.

**CLI Interface:** Uses Python Fire for a clean, intuitive command-line interface with automatic help generation.

**Four model versions available:**
- **`baseline`**: Knowledge, reasoning, scaffold, level, verbosity (no task or category effects)
- **`category`**: Adds category effect (κ_c) for task vs subtask distinction
- **`task`**: Adds task-specific random effects (η_t) for idiosyncratic task difficulty
- **`category_task`**: Includes both category and task effects

## Prerequisites

Run the IRT models first:
```bash
cd ../../analysis

# All 4 models (takes ~200 min total, recommend parallel execution)
snakemake -j2

# Or run individually
python latent_factor_modeling.py --output_dir results/irt_baseline
python latent_factor_modeling.py --with_category --output_dir results/irt_category
python latent_factor_modeling.py --with_task_effects --output_dir results/irt_task
python latent_factor_modeling.py --with_task_effects --with_category --output_dir results/irt_category_task
```

Results will be in:
- `../../analysis/results/irt_baseline/` - Baseline model
- `../../analysis/results/irt_category/` - Category effects only
- `../../analysis/results/irt_task/` - Task effects only
- `../../analysis/results/irt_category_task/` - Both category and task effects

## Usage

### List available plots
```bash
python plot_irt_results.py list_plots
```

### Generate all plots (baseline model)
```bash
python plot_irt_results.py generate --model_type=baseline
```

### Generate plots for other models
```bash
python plot_irt_results.py generate --model_type=category
python plot_irt_results.py generate --model_type=task
python plot_irt_results.py generate --model_type=category_task
```

### Generate specific plots (comma-separated)
```bash
python plot_irt_results.py generate --model_type=baseline --plots=capability_heatmaps,lambda_forest
```

### Custom output directory
```bash
python plot_irt_results.py generate --model_type=baseline --output_dir=./my_plots
```

### Advanced: Specify custom results directory
```bash
python plot_irt_results.py generate --results_dir=../../analysis/results/irt_baseline --output_dir=./output
```

### Get help
```bash
python plot_irt_results.py generate -- --help
```

## Available Plots

### Capability Plots (from IRT theta estimates)
- **capability_heatmaps** - Heatmaps showing θ_K and θ_R across models and environments
- **capability_profiles** - Line plots showing capability trends across domains
- **capability_comparison** - Grouped bar charts comparing knowledge vs reasoning
- **knowledge_vs_reasoning** - Scatter plot of θ_K vs θ_R for each (model, environment)

### Agent Model Plots (from MCMC trace)
- **lambda_forest** - Knowledge loading (λ) forest plot showing domain knowledge importance
- **psi_forest** - Reasoning loading (ψ) forest plot showing reasoning importance
- **variance_decomposition** - Bar chart of variance explained by each component
- **scaffold_effects** - Agent scaffold (ReAct vs ToolCalling) effects
- **level_effects** - Difficulty level effects across tasks

## Output

All plots are saved as:
- PNG (300 DPI) for presentations/papers
- PDF (vector) for high-quality publications

## Model Results

Results from 4 model specifications to understand variance decomposition:

**Baseline (irt_baseline):**
- Knowledge, reasoning, scaffold, level, verbosity effects only
- Provides the simplest explanation of agent performance

**Category (irt_category):**
- Adds category effect (κ_c) to test if tasks and subtasks differ systematically
- Helps understand if task complexity affects performance beyond domain-specific factors

**Task (irt_task):**
- Adds task-specific random effects (η_t) to capture idiosyncratic task difficulty
- Useful when each task has unique characteristics not explained by domain or level

**Category + Task (irt_category_task):**
- Full model with both category and task effects
- Most flexible specification, allows comparison of variance explained by systematic (category) vs idiosyncratic (task) effects

**Key Finding:** Compare variance decomposition across all 4 models to determine which effects meaningfully improve explanatory power.

## Integration with Analysis Pipeline

This script uses shared utilities from the `analysis/` directory:
- `plot_config.py` - Color palettes and naming conventions
- `plot_utils.py` - Data loading and filtering functions

Model colors are consistent with the rest of the Corral benchmark plots.
