# Panel 2 Plots - Slope Plot

## Overview

The `2_slope.py` script generates slope plots showing model performance across different environments, ordered by QA scores. Each model is represented by a colored line connecting its scores across environments.

## Features

### 1. Environment Ordering by QA Score
- **Strategy**: `average_qa` (default) or `model_specific_qa`
  - `average_qa`: Orders environments by average QA score across all models
  - `model_specific_qa`: Orders by QA score of a specific model (requires `--model_for_ordering`)
- **QA Type**: `qa` (default) or `reasoning_qa`
  - `qa`: Use standard QA scores for ordering
  - `reasoning_qa`: Use reasoning QA scores for ordering
- **Direction**: `ascending` or `descending` (default)

### 2. Verbosity Filtering
- `average` (default): Average across all verbosities (brief, workflow, comprehensive)
- `brief`: Only brief verbosity
- `workflow`: Only workflow verbosity
- `comprehensive`: Only comprehensive verbosity

### 3. Task Type Filtering
- `both` (default): Include both task and subtask categories
- `tasks`: Only category="task"
- `subtasks`: Only category="subtask"
- **Note**: Task type (task/subtask) is independent from level (1, 2, 3, 4)

### 4. Level Filtering
- `default_map` (default): Use per-environment level mapping
  - AFM: level 1, Catalyst: level 1, MD: level 2, ML: level 1
  - Resistor: level 1, Retro: level 2, Spectra: level 1, Wetlab: level 2
- `all`: Average across all levels for each environment
- `1`, `2`, `3`, `4`: Use specific level for all environments

### 5. Agent Type Selection
- `average` (default): Average across both React and Tool-Calling agents
- `react`: Only React agent
- `tool_calling`: Only Tool-Calling agent

### 6. Metric Selection
- **Metric Type**: `average_score` (default), `pass_at_k`, or `pass_hat_k`
  - `average_score`: Average score across trials
  - `pass_at_k`: Pass@k metric (at least one success in k trials)
  - `pass_hat_k`: Pass^k metric (all k trials successful)
- **K Value**: Integer from 1-5 (only for `pass_at_k` and `pass_hat_k`)

### 7. Model Selection
- Default: `claude-4.5,gpt-4o`
- Can specify 1-3 models: `claude-4.5`, `gpt-4o`, `gpt-oss-120b`

## Usage

### Basic Usage (Default Settings)

```bash
python 2_slope.py
```

This generates `slope_plot.pdf` with:
- Environments ordered by average QA score (descending)
- Average verbosity across all levels
- Both tasks and subtasks
- Average across both agent types (React and Tool-Calling)
- Claude-4.5 and GPT-4o models

### Custom Configuration Examples

#### 1. Compare only React agents across all environments

```bash
python 2_slope.py \
    --agent_type_strategy=react \
    --output_filename=react_agents_slope.pdf
```

#### 2. Compare only Tool-Calling agents with reasoning QA ordering

```bash
python 2_slope.py \
    --agent_type_strategy=tool_calling \
    --qa_type_for_ordering=reasoning_qa \
    --output_filename=toolcalling_reasoningqa_slope.pdf
```

#### 3. Order by Claude's QA score, ascending order, workflow verbosity

```bash
python 2_slope.py \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=workflow \
    --output_filename=claude_ordered_workflow.pdf
```

#### 4. Only task category, level 1 for all environments

```bash
python 2_slope.py \
    --task_type_strategy=tasks \
    --level_strategy=1 \
    --output_filename=tasks_level1.pdf
```

#### 5. Use Pass@3 metric with all three models

```bash
python 2_slope.py \
    --metric=pass_at_k \
    --k_value=3 \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --output_filename=passat3_all_models.pdf
```

#### 6. Full custom configuration

```bash
python 2_slope.py \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=gpt \
    --qa_type_for_ordering=reasoning_qa \
    --order_direction=descending \
    --verbosity_strategy=comprehensive \
    --task_type_strategy=both \
    --level_strategy=default_map \
    --agent_type_strategy=average \
    --metric=pass_at_k \
    --k_value=5 \
    --models="claude-4.5,gpt-4o" \
    --output_filename=custom_slope.pdf
```

## Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--ordering_strategy` | str | `average_qa` | `average_qa`, `model_specific_qa` | How to order environments |
| `--order_direction` | str | `descending` | `ascending`, `descending` | Sort order |
| `--model_for_ordering` | str | None | `claude`, `gpt`, `gpt_oss` | Model for `model_specific_qa` |
| `--qa_type_for_ordering` | str | `qa` | `qa`, `reasoning_qa` | Type of QA to use for ordering |
| `--verbosity_strategy` | str | `average` | `average`, `brief`, `workflow`, `comprehensive` | Tool verbosity filter |
| `--task_type_strategy` | str | `both` | `tasks`, `subtasks`, `both` | Task type filter (task vs subtask category) |
| `--level_strategy` | str | `default_map` | `all`, `default_map`, `1`, `2`, `3`, `4` | Level filter for environments |
| `--agent_type_strategy` | str | `average` | `average`, `react`, `tool_calling` | Agent type filter/aggregation |
| `--metric` | str | `average_score` | `average_score`, `pass_at_k`, `pass_hat_k` | Metric to plot |
| `--k_value` | int | `5` | `1-5` | K value for Pass@k or Pass^k |
| `--models` | str | `claude-4.5,gpt-4o` | Comma-separated list | Models to plot |
| `--output_filename` | str | `slope_plot.pdf` | Any filename | Output file path |

## Output

The script generates two files:
- `{output_filename}.pdf` - Vector graphics (recommended for publications)
- `{output_filename}.png` - Raster graphics (300 DPI)

## Data Sources

The script uses two datasets downloaded from HuggingFace:
1. **Main Benchmark Reports** (`analysis/results/data/reports.jsonl`)
   - Contains agent performance metrics across environments
2. **QA Topic Reports** (`analysis/results/data/qa_topic_reports.jsonl`)
   - Contains QA evaluation scores used for ordering

## Plot Interpretation

- **X-axis**: Environments (ordered by QA score)
- **Y-axis**: Score (0-1)
- **Lines**: Each line represents a model, connecting its scores across environments
- **Vertical dashed lines**: Separate each environment
- **Colors**:
  - Purple (#711c91): Claude-4.5
  - Magenta (#ea00d9): GPT-4o
  - Violet (#7150e0): GPT-OSS-120b

## Comparison with Dumbbell Plot

| Feature | Slope Plot (2_slope.py) | Dumbbell Plot (1_slope.py) |
|---------|-------------------------|----------------------------|
| Purpose | Show model performance trends | Compare React vs Tool-Calling |
| Lines | One per model | One per model per environment |
| Agent Strategy | Can select one or average both | Always shows both |
| Best for | Comparing models across environments | Comparing agent types |

## Requirements

- Python 3.10+
- Dependencies: pandas, matplotlib, fire, loguru, scipy, numpy, lama_aesthetics
- Data files must be downloaded first (run Snakefile in `analysis/`)

## Notes

- The script automatically filters out environments with no data
- Missing data points are handled gracefully
- Plot styling uses `plot_config.py` for consistent colors and fonts across panels
- All code passes `ruff` linting checks
- Font sizes are centralized in `plot_config.py` for easy customization
