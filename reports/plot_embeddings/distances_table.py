"""
Creates a tex file with the table showing the mean distances between tasks and tools
The script loads the data from the JSON files parsing the correct ones by their names.
The table is saved in the .tex file "distances_table.tex"
"""

import json
from pathlib import Path

# Collect data from all JSON files
table_data = []

for file in Path("heatmaps").rglob("*/*.json"):
    if (
        "melting" in file.name
        or "quenching" in file.name
        or "surface_energy" in file.name
    ):
        continue
    if "distances" in file.name:
        continue
    with file.open() as f:
        data = json.load(f)

    # Extract and format the data
    task_with_tools = data["task"]
    # Apply replacements to task name
    task_with_tools = task_with_tools.replace("catalyst", "\\opencatalyst")
    task_with_tools = task_with_tools.replace("corral_md", "\\md")
    task_with_tools = task_with_tools.replace("ml", "\\ml")
    task_with_tools = task_with_tools.replace("spectra_elucidation", "\\spectra")

    number_tools = data["tool_count"]
    verbosity = data["verbosity"]
    # Apply replacements to verbosity
    verbosity = verbosity.replace("full", "comprehensive")

    mean_distance = round(data["mean_distance_tools_vs_tasks"], 2)
    mean_of_min = round(data["mean_of_min_distances_per_task"], 2)
    mean_consecutive = round(data["mean_consecutive_task_distance"], 2)

    table_data.append(
        [
            task_with_tools,
            number_tools,
            mean_consecutive,
            verbosity,
            mean_distance,
            mean_of_min,
        ]
    )

# Sort data by task first, then by verbosity to group them properly
task_order = {"\\spectra": 1, "\\md": 2, "\\opencatalyst": 3, "\\ml": 4}
verbosity_order = {"brief": 1, "workflow": 2, "full": 3}
table_data.sort(key=lambda x: (task_order.get(x[0], 5), verbosity_order.get(x[3], 4)))

table_lines = []
# Print LaTeX table
table_lines.append(
    "\\begin{tabularx}{\\textwidth}{l>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X>{\\centering\\arraybackslash}X}"
)
table_lines.append("\\toprule")
table_lines.append(
    "\\multirow{2}{*}{Task} & \\multirow{2}{*}{Tools} & \\multirow{2}{*}{\\parbox{2cm}{\\centering Mean of consecutive tasks}} & \\multirow{2}{*}{Verbosity} & \\multicolumn{2}{c}{Distance Tools-Tasks} \\\\"
)
table_lines.append("\\cmidrule(lr){5-6}")
table_lines.append("& & & & Mean & Mean of Minimum \\\\")
table_lines.append("\\midrule")
table_lines.append("\\midrule")

# Group data by task and create multirow entries
current_task = None
task_count = 0

for _i, row in enumerate(table_data):
    task_name = row[0]

    # Count how many rows this task has
    if task_name != current_task:
        # Add a white line before the new task (except for the first task)
        if current_task is not None:
            table_lines.append("\\addlinespace[0.5em]")

        current_task = task_name
        # Count occurrences of this task
        task_rows = sum(1 for r in table_data if r[0] == task_name)

        # First row of a task group - use multirow
        table_lines.append(
            f"\\multirow{{{task_rows}}}{{*}}{{{task_name}}} & \\multirow{{{task_rows}}}{{*}}{{{row[1]}}} & \\multirow{{{task_rows}}}{{*}}{{{row[2]}}} & {row[3]} & {row[4]} & {row[5]} \\\\"
        )
    else:
        # Subsequent rows - empty first three columns
        table_lines.append(f" &  &  & {row[3]} & {row[4]} & {row[5]} \\\\")

table_lines.append("\\bottomrule")
table_lines.append("\\end{tabularx}")

# Write the table to a .tex file
with Path("distances_table.tex").open("w") as tex_file:
    tex_file.write("\n".join(table_lines) + "\n")
