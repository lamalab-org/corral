import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

# Load the processed_results.json file
with Path("processed_results.json").open() as f:
    data = json.load(f)

# Define the environments and columns (removed Model from columns)
envs = ["MD", "ML", "Catalyst", "Spectra"]
columns = [
    "Agent",
    "Verbosity",
    "MD (single)",
    "MD (chained)",
    "ML (single)",
    "ML (chained)",
    "Catalyst (single)",
    "Catalyst (chained)",
    "Spectra (single)",
    "Spectra (chained)",
]

# Group entries by (model, agent_type, verbosity_level)
results = defaultdict(lambda: {col: None for col in columns[2:]})


def normalize_model(model):
    if model is None:
        return model
    m = model.lower()
    if m.startswith(("claude_35", "claude_35_sonnet")) or m == "claude":
        return "claude"
    if m.startswith("gpt_4o") or m == "gpt4o":
        return "gpt-4o"
    return model


for entry in data:
    model = normalize_model(entry.get("model"))
    key = (model, entry.get("agent_type"), entry.get("verbosity_level"))
    env = entry.get("env")
    chained = entry.get("chained")
    pass5 = entry.get("pass@5")
    if env in envs:
        col = f"{env} ({'chained' if chained else 'single'})"
        # If already present, take the mean if both are not None
        if results[key][col] is not None and pass5 is not None:
            try:
                results[key][col] = (results[key][col] + pass5) / 2
            except Exception:
                results[key][col] = pass5
        elif pass5 is not None:
            results[key][col] = pass5

# Prepare rows for the DataFrame, grouped by model
model_groups = defaultdict(list)
for (model, agent, verbosity), vals in results.items():
    row = [agent, verbosity]
    for env in envs:
        row.append(vals[f"{env} (single)"])
        row.append(vals[f"{env} (chained)"])
    model_groups[model].append(row)


# Custom sorting function to put workflow second
def sort_by_agent_with_workflow_second(rows):
    """Sort rows so that workflow appears second in each group of agents."""
    # Group by agent type
    agent_groups = defaultdict(list)
    for row in rows:
        agent_type = row[0]
        agent_groups[agent_type].append(row)

    # Sort each agent group by verbosity
    for agent_type in agent_groups:
        agent_groups[agent_type].sort(key=lambda x: x[1] if x[1] is not None else "")

    # Define the desired order: put workflow second if it exists
    agent_types = list(agent_groups.keys())
    if None in agent_types:
        agent_types.remove(None)
        agent_types.append(None)  # Put None at the end

    # Custom sort: workflow second, others alphabetically
    def agent_sort_key(agent):
        if agent is None:
            return ("z", agent)  # Put None at the end
        elif agent == "workflow":
            return ("b", agent)  # Put workflow second (after 'a' but before 'c')
        else:
            return ("a" if agent < "workflow" else "c", agent)

    sorted_agents = sorted(agent_types, key=agent_sort_key)

    # Reconstruct rows with spacing between different agents
    result_rows = []
    for i, agent_type in enumerate(sorted_agents):
        if i > 0:  # Add empty row between different agents
            result_rows.append([""] * len(columns))
        result_rows.extend(agent_groups[agent_type])

    return result_rows


# Sort models for consistent output
sorted_models = sorted(model_groups.keys(), key=lambda x: (x is None, x))

# Create DataFrame with model headers and reorganized rows
all_rows = []
for i, model in enumerate(sorted_models):
    # Add model header row
    model_name = model if model is not None else "Unknown Model"
    all_rows.append([f"=== {model_name.upper()} ==="] + [""] * (len(columns) - 1))

    # Sort rows with custom function
    sorted_rows = sort_by_agent_with_workflow_second(model_groups[model])

    # Add data rows for this model
    all_rows.extend(sorted_rows)

    # Add empty row for spacing between models (except after last model)
    if i < len(sorted_models) - 1:
        all_rows.append([""] * len(columns))

results_table = pd.DataFrame(all_rows, columns=columns)

# Round all float columns to 2 decimals (skip header rows)
for col in columns[2:]:
    results_table[col] = results_table[col].apply(
        lambda x: round(x, 2) if isinstance(x, float) else x
    )

# Find the best (maximum) score for each numerical column
best_scores = {}
for col in columns[2:]:  # Skip 'Agent' and 'Verbosity' columns
    # Get all numerical values from this column (excluding headers and empty rows)
    numerical_values = []
    for _idx, row in results_table.iterrows():
        val = row[col]
        if isinstance(val, int | float) and not pd.isna(val):
            numerical_values.append(val)

    if numerical_values:
        best_scores[col] = max(numerical_values)

# Save to CSV or print
logger.info(results_table.to_string(index=False))

# Print as LaTeX table with custom formatting for headers and multirow agents
logger.info("\nLaTeX Table:\n")


def create_latex_table_with_multirow(df, columns, best_scores):
    """Create LaTeX table with proper multirow handling for agents and highlighting best scores."""
    latex_lines = []
    latex_lines.append("\\begin{tabular}{lccccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append(
        "\\multirow{2}{*}{Agent} & \\multirow{2}{*}{Verbosity} & \\multicolumn{2}{c}{MD} & \\multicolumn{2}{c}{ML} & \\multicolumn{2}{c}{Catalyst} & \\multicolumn{2}{c}{Spectra} \\\\"
    )
    latex_lines.append(
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}"
    )
    latex_lines.append(
        "& & single & chained & single & chained & single & chained & single & chained \\\\"
    )
    latex_lines.append("\\midrule")

    i = 0
    while i < len(df):
        row = df.iloc[i]

        # Handle model headers
        if str(row[0]).startswith("==="):
            if i > 0:  # Add midrule before new model (except first)
                latex_lines.append("\\midrule")
            model_header = str(row[0]).replace("=== ", "").replace(" ===", "")
            latex_lines.append(
                f"\\multicolumn{{{len(columns)}}}{{c}}{{\\textbf{{{model_header}}}}} \\\\"
            )
            latex_lines.append("\\midrule")
            i += 1
            continue

        # Skip empty rows (but add some spacing in LaTeX)
        if str(row[0]) == "":
            latex_lines.append("\\addlinespace[0.5em]")
            i += 1
            continue

        # Handle data rows with multirow for agents
        current_agent = str(row[0]) if not pd.isna(row[0]) else ""

        # Count consecutive rows with same agent
        j = i
        agent_rows = []
        while j < len(df):
            next_row = df.iloc[j]
            if (
                str(next_row[0]).startswith("===")
                or str(next_row[0]) == ""
                or j > i
                and str(next_row[0]) != current_agent
            ):
                break
            if str(next_row[0]) == current_agent:
                agent_rows.append(next_row)
            j += 1

        # Generate LaTeX for this agent group
        if len(agent_rows) > 1:
            # Multirow case
            for idx, agent_row in enumerate(agent_rows):
                row_data = []
                if idx == 0:
                    # First row gets the multirow command
                    row_data.append(
                        f"\\multirow{{{len(agent_rows)}}}{{*}}{{{current_agent}}}"
                    )
                else:
                    # Subsequent rows get empty agent cell
                    row_data.append("")

                # Add remaining columns
                for col_idx, val in enumerate(agent_row[1:], start=1):
                    if pd.isna(val) or val == "" or val is None:
                        row_data.append("")
                    else:
                        # Check if this is a numerical column and if the value is the best score
                        col_name = columns[col_idx]
                        if (
                            col_name in best_scores
                            and isinstance(val, int | float)
                            and abs(val - best_scores[col_name]) < 1e-10
                        ):  # Use small tolerance for float comparison
                            row_data.append(f"\\textbf{{{val}}}")
                        else:
                            row_data.append(str(val))

                latex_lines.append(" & ".join(row_data) + " \\\\")
        else:
            # Single row case
            row_data = []
            for col_idx, val in enumerate(agent_rows[0]):
                if pd.isna(val) or val == "" or val is None:
                    row_data.append("")
                else:
                    # Check if this is a numerical column and if the value is the best score
                    col_name = columns[col_idx]
                    if (
                        col_idx > 1  # Skip Agent and Verbosity columns
                        and col_name in best_scores
                        and isinstance(val, int | float)
                        and abs(val - best_scores[col_name]) < 1e-10
                    ):  # Use small tolerance for float comparison
                        row_data.append(f"\\textbf{{{val}}}")
                    else:
                        row_data.append(str(val))
            latex_lines.append(" & ".join(row_data) + " \\\\")

        i = j  # Move to next unprocessed row

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines[9], latex_lines[10] = latex_lines[10], latex_lines[9]
    latex_lines[13], latex_lines[14] = latex_lines[14], latex_lines[13]
    latex_lines[20], latex_lines[21] = latex_lines[21], latex_lines[20]
    latex_lines[24], latex_lines[25] = latex_lines[25], latex_lines[24]

    return "\n".join(latex_lines)


table = create_latex_table_with_multirow(results_table, columns, best_scores)
table = table.replace("tool_calling", "Tool Calling")
table = table.replace("react", "ReAct")
table = table.replace("CLAUDE", "Claude 3.5 Sonnet")
table = table.replace("GPT-4O", "GPT-4o")
table = table.replace("\\addlinespace[0.5em]\n\\midrule", "\\midrule")

logger.info(table)

# Save LaTeX table to a .tex file
with Path("summary_table.tex").open("w") as texfile:
    texfile.write(table)
