# Panel 2 Plots - Dumbbell Plot

## Overview

The `1_slope.py` script generates dumbbell plots comparing React vs Tool-Calling agents across different environments, ordered by QA scores.

## Features

### 1. Environment Ordering by QA Score
- **Strategy**: `average_qa` (default) or `model_specific_qa`
  - `average_qa`: Orders environments by average QA score across all models
  - `model_specific_qa`: Orders by QA score of a specific model (requires `--model_for_ordering`)
- **Direction**: `ascending` or `descending` (default)

### 2. Verbosity Filtering
- `average` (default): Average across all verbosities (brief, workflow, comprehensive)
- `brief`: Only brief verbosity
- `workflow`: Only workflow verbosity
- `comprehensive`: Only comprehensive verbosity

### 3. Task Type Filtering
- `both` (default): Include both tasks and subtasks
- `tasks`: Only level 1 (main tasks)
- `subtasks`: Only levels > 1 (subtasks)

### 4. Agent Type Selection
- `average` (default): Shows both React and Tool-Calling agents
- Note: For dumbbell plots, both agent types are always shown for comparison

### 5. Model Selection
- Default: `claude-4.5,gpt-4o`
- Can specify 1-3 models: `claude-4.5`, `gpt-4o`, `gpt-oss-120b`

## Usage

### Basic Usage (Default Settings)

```bash
python 1_slope.py
```

This generates `dumbbell_plot.pdf` with:
- Environments ordered by average QA score (descending)
- Average verbosity across all levels
- Both tasks and subtasks
- Claude-4.5 and GPT-4o models

### Custom Configuration Examples


```bash
python 1_slope.py \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending \
    --verbosity_strategy=average \
    --task_type_strategy=both

```

#### 1. Order by Claude's QA score, ascending order

```bash
python 1_slope.py \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=claude \
    --order_direction=ascending
```

#### 2. Only workflow verbosity, only main tasks

```bash
python 1_slope.py \
    --verbosity_strategy=workflow \
    --task_type_strategy=tasks
```

#### 3. All three models, brief verbosity, subtasks only

```bash
python 1_slope.py \
    --models="claude-4.5,gpt-4o,gpt-oss-120b" \
    --verbosity_strategy=brief \
    --task_type_strategy=subtasks \
    --output_filename=three_models_brief_subtasks.pdf
```

#### 4. Full custom configuration

```bash
python 1_slope.py \
    --ordering_strategy=model_specific_qa \
    --model_for_ordering=gpt \
    --order_direction=descending \
    --verbosity_strategy=comprehensive \
    --task_type_strategy=both \
    --models="claude-4.5,gpt-4o" \
    --output_filename=custom_plot.pdf
```

## Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `--ordering_strategy` | str | `average_qa` | `average_qa`, `model_specific_qa` | How to order environments |
| `--order_direction` | str | `descending` | `ascending`, `descending` | Sort order |
| `--model_for_ordering` | str | None | `claude`, `gpt`, `gpt_oss` | Model for `model_specific_qa` |
| `--verbosity_strategy` | str | `average` | `average`, `brief`, `workflow`, `comprehensive` | Tool verbosity filter |
| `--task_type_strategy` | str | `both` | `tasks`, `subtasks`, `both` | Task level filter |
| `--agent_type_strategy` | str | `average` | `average` | Agent type (always shows both for dumbbells) |
| `--models` | str | `claude-4.5,gpt-4o` | Comma-separated list | Models to plot |
| `--output_filename` | str | `dumbbell_plot.pdf` | Any filename | Output file path |

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
- **Y-axis**: Average Score (0-1)
- **Lines**: Each line connects React (filled circle) to Tool-Calling (hollow circle) for a model
- **Colors**:
  - Purple (#8D5F8C): Claude-4.5
  - Blue (#696FC7): GPT-4o
  - Teal (#4A90A4): GPT-OSS-120b

## Requirements

- Python 3.10+
- Dependencies: pandas, matplotlib, fire, loguru, scipy, numpy
- Data files must be downloaded first (run Snakefile in `analysis/`)

## Notes

- The script automatically filters out environments with no data
- Missing data points are handled gracefully
- Plot styling uses `plot_config.py` for consistent colors across panels
- All code passes `ruff` linting checks
