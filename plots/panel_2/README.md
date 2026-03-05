# Panel 2 Plots

This directory contains scripts for generating comparison plots across environments, ordered by QA scores.

## Available Plots

### 1. Dumbbell Plot (`1_slope.py`)
**Purpose**: Compare React vs Tool-Calling agent performance

Generates dumbbell plots showing performance differences between React and Tool-Calling agents across environments. Each environment shows dumbbells (one per model) with filled circles for React and hollow circles for Tool-Calling.

**Quick Start**:
```bash
python 1_slope.py
```

**Key Features**:
- Always shows both agent types for comparison
- Supports 1-3 models
- Configurable environment ordering, verbosity, task type, and level strategies

### 2. Slope Plot (`2_slope.py`)
**Purpose**: Show model performance trends across environments

Generates slope plots where each model is represented by a colored line connecting its scores across environments. Useful for comparing overall model performance patterns.

**Quick Start**:
```bash
python 2_slope.py
```

**Key Features**:
- One line per model
- Can filter by agent type (react/tool_calling/average)
- Configurable environment ordering, verbosity, task type, and level strategies

**Documentation**: See [README_slope.md](README_slope.md) for detailed usage

### 3. Performance Gap Plot (`3_gap_plot.py`)
**Purpose**: Show performance gaps across environments

Generates a plot with two lines showing:
1. **Model Gap**: (best model - worst model) averaged across agents
2. **Agent Gap**: (best agent - worst agent) averaged across models

**Quick Start**:
```bash
python 3_gap_plot.py
```

**Key Features**:
- Two lines showing different types of performance variance
- Model gap (rose color) - shows how much models differ
- Agent gap (teal color) - shows how much agents/scaffolds differ
- Configurable environment ordering, verbosity, task type, and level strategies

## Common Configuration Options

Both scripts support these strategies:

| Strategy | Options | Default | Description |
|----------|---------|---------|-------------|
| **Environment Ordering** | `average_qa`, `model_specific_qa` | `average_qa` | How to order environments |
| **QA Type** | `qa`, `reasoning_qa` | `qa` | Which QA scores to use for ordering |
| **Order Direction** | `ascending`, `descending` | `descending` | Sort order |
| **Verbosity** | `average`, `brief`, `workflow`, `comprehensive` | `average` | Tool verbosity filter |
| **Task Type** | `tasks`, `subtasks`, `both` | `both` | Task category filter |
| **Level** | `all`, `default_map`, `1-4` | `default_map` | Difficulty level filter |
| **Metric** | `average_score`, `pass_at_k`, `pass_hat_k` | `average_score` | Performance metric |

### Dumbbell-Specific Options
- **Agent Type**: Always shows both (React and Tool-Calling)

### Slope-Specific Options
- **Agent Type**: `average`, `react`, `tool_calling` - can filter or average

## Data Requirements

Both scripts require data files in `analysis/results/data/`:
1. `reports.jsonl` - Main benchmark results
2. `qa_topic_reports.jsonl` - QA evaluation scores

Run the Snakefile in `analysis/` to download these files:
```bash
cd ../../analysis
snakemake -c1
```

## Configuration

Colors, font sizes, and display names are centralized in:
- `analysis/plot_config.py` - Shared configuration for all plots

## Output

Both scripts generate:
- `.pdf` - Vector graphics (recommended for publications)
- `.png` - Raster graphics (300 DPI)

## Examples

### Compare agent types across environments ordered by reasoning QA
```bash
python 1_slope.py --qa_type_for_ordering=reasoning_qa
```

### Show React agent performance trends
```bash
python 2_slope.py --agent_type_strategy=react
```

### Use Pass@5 metric with all three models
```bash
python 2_slope.py --metric=pass_at_k --k_value=5 --models="claude-4.5,gpt-4o,gpt-oss-120b"
```

### Show performance gaps ordered by reasoning QA
```bash
python 3_gap_plot.py --qa_type_for_ordering=reasoning_qa
```
