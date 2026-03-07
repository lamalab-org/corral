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
- Model gap (violet color) - shows how much models differ
- Agent gap (rose color) - shows how much agents/scaffolds differ
- Configurable environment ordering, verbosity, task type, and level strategies

### 4. Subtask Heaviness Plot (`4_task_heaviness.py`)
**Purpose**: Visualize subtask reasoning complexity and success rates

Generates flowing ribbon plots for each environment showing:
- **Color**: Reasoning heaviness (darker = more reasoning intensive)
  - Reasoning (darkest)
  - Validation
  - Code Execution
  - Experiment Execution
  - Retrieval (lightest)
- **Thickness**: Success rate (thicker = higher success)

**Quick Start**:
```bash
python 4_task_heaviness.py
```

**Key Features**:
- Smooth ribbons showing workflow progression
- Each environment shows subtask sequence from first to last
- Can filter to specific model, agent, or environment
- Configurable verbosity and level strategies
- Uses `reasoning.json` for heaviness mapping

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

### Show subtask heaviness for Claude-4.5 only
```bash
python 4_task_heaviness.py --model_strategy=claude-4.5
```

### Show subtask heaviness for catalyst environment only
```bash
python 4_task_heaviness.py --env_filter=catalyst
```

### 5. Gap Scatter Plot (`5_gap_scatter.py`)
**Purpose**: Scatter plot showing model gap vs agent gap

Generates a scatter plot where each point represents an environment, with agent gap on the x-axis and model gap on the y-axis. Points above the diagonal line (y=x) indicate that model choice contributes more performance variance than agent scaffold.

**Quick Start**:
```bash
python 5_gap_scatter.py
```

**Key Features**:
- Each point = one environment
- Diagonal line shows equal gap (y=x)
- Points above diagonal = model dominance
- Automatic label positioning to avoid overlap
- Shows clear quantitative model/agent ratio

### 6. Gap Bar Chart (`6_gap_bars.py`)
**Purpose**: Side-by-side bars comparing model and agent gaps

Generates grouped bar charts showing model gap and agent gap for each environment. Bars are sorted by model gap (descending) to emphasize consistent model dominance.

**Quick Start**:
```bash
python 6_gap_bars.py
```

**Key Features**:
- Two bars per environment (model gap | agent gap)
- Sorted by model gap to show consistent pattern
- Optional sorting by agent gap or no sorting
- Easy visual comparison of gap magnitudes

### 7. Clustered Scatter Plot (`7_clustered_scatter.py`)
**Purpose**: Show all model×agent combinations without derived gap metric

Generates scatter plot showing every (model, agent, environment) combination. Points are colored by model and shaped by agent type. Visual clustering shows that model clusters are more separated than agent variations within each model.

**Quick Start**:
```bash
python 7_clustered_scatter.py
```

**Key Features**:
- No derived "gap" metric - uses raw performance scores
- Color = model (3 colors)
- Shape = agent type (circle for ReAct, square for Tool-Calling)
- Points clustered by model show model dominance
- X-axis jitter for visibility

### 8. Performance Heatmap (`8_performance_heatmap.py`)
**Purpose**: Dual heatmap view showing model×environment performance by agent type

Generates two side-by-side heatmaps (ReAct and Tool-Calling), each showing a 3×7 matrix where rows are models and columns are environments. The similarity between the two heatmaps shows minimal agent effect, while strong row-wise color variation within each heatmap shows large model effect.

**Quick Start**:
```bash
python 8_performance_heatmap.py
```

**Key Features**:
- No derived "gap" metric - uses raw performance scores
- Two 3×7 matrices side-by-side (one per agent type)
- Rows = models (shared y-axis), Columns = environments
- Consistent color scale across both heatmaps
- Annotated cells show exact scores
- Visual pattern: Two heatmaps look very similar (small agent effect), but rows within each show strong color variation (large model effect)
- Prints comparison statistics including mean absolute difference between agent types

### 9. Full Coverage Heatmap (`9_full_coverage_heatmap.py`)
**Purpose**: Show complete experimental coverage with all environment-level combinations

Generates a single wide heatmap where rows are model×agent configurations (6 total) and columns are all environment-level combinations tested (14 total: AFM-1 through AFM-4, Catalyst-1, MD-1, MD-2, ML-1, Resistor-1, Retro-1 through Retro-3, Spectra-1, Spectra-2). This shows the full experimental scope.

**Quick Start**:
```bash
python 9_full_coverage_heatmap.py
```

**Key Features**:
- No derived "gap" metric - uses raw performance scores
- 6 rows (model×agent configs) × 14 columns (env-level combos)
- Shows complete experimental coverage (84 total conditions, with 8 missing data points)
- Column labels include level information (e.g., "AFM-L2", "MD-L1")
- Title displays experimental scope
- Identifies missing data points

### 10. Bubble Chart (`10_bubble_chart.py`)
**Purpose**: Alternative visualization showing performance clustering by model

Generates a bubble chart where x-axis shows environment-level combinations, y-axis shows performance score, color represents model, and shape represents agent type. Points cluster by color (model), not by shape (agent), demonstrating model dominance.

**Quick Start**:
```bash
python 10_bubble_chart.py
```

**Key Features**:
- No derived "gap" metric - uses raw performance scores
- X-axis: Environment-level combinations (14)
- Y-axis: Performance score
- Color: Model (3 colors)
- Shape: Agent type (○ ReAct, □ Tool-Calling)
- Size: Number of runs (uniform if all same)
- Visual pattern: Horizontal bands of same color (model clustering) rather than shape clustering
