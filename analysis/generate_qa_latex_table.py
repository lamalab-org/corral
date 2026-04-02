"""
Generate a LaTeX table from qa_topic_reports.jsonl summarising overall_score
by question type (Knowledge / Reasoning), environment, and model.

Output: analysis/results/tables/qa_scores_table.tex
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constants import MODEL_CANONICAL, MODEL_DISPLAY

DATA_PATH = Path(__file__).parent / "results" / "data" / "qa_topic_reports.jsonl"
OUT_PATH = Path(__file__).parent / "results" / "tables" / "qa_scores_table.tex"

# corral_md is the reasoning-focused variant of the md environment
ENV_ALIASES = {"corral_md": "md"}

ENV_LABELS = {
    "afm": "AFM",
    "catalyst": "Catalyst",
    "md": "MD",
    "ml": "ML",
    "resistor": "Resistor Network",
    "retro": "Retrosynthesis",
    "spectra": "Spectra Elucidation",
    "wetlab": "Wet Lab",
}

QA_TYPE_LABELS = {
    "qa": "Knowledge Questions",
    "reasoning_qa": "Reasoning Questions",
}

# Canonical display order; wetlab is included for future data even if not yet present.
ENV_ORDER = ["afm", "catalyst", "md", "ml", "resistor", "retro", "spectra", "wetlab"]

records = []
with DATA_PATH.open() as fh:
    for raw_line in fh:
        stripped = raw_line.strip()
        if stripped:
            records.append(json.loads(stripped))

# data[qa_type][env][model] -> (overall_score, n_questions)
data: dict[str, dict[str, dict[str, tuple[float, int]]]] = defaultdict(
    lambda: defaultdict(dict)
)

for r in records:
    qa_type = r.get("qa_type", "")
    env = ENV_ALIASES.get(r.get("env", ""), r.get("env", ""))
    raw_model = r.get("model", "")
    canonical = MODEL_CANONICAL.get(raw_model, raw_model)
    model = MODEL_DISPLAY.get(canonical, raw_model)
    score = r.get("overall_score")
    n_q = r.get("total_questions_completed")
    if qa_type and env and model and score is not None:
        data[qa_type][env][model] = (float(score), int(n_q))

# Sorted by display name so column order is stable across runs.
all_models: list[str] = sorted(
    {model for qt in data.values() for env_d in qt.values() for model in env_d}
)

n_model_cols = len(all_models)
# columns: Environment | N questions | model1 | model2 | ...
col_spec = "l r " + " ".join(["S[table-format=1.2]"] * n_model_cols)

# Model names contain '-' and '.', so each header cell needs {} wrapping for siunitx S columns.
model_header = " & ".join(f"{{\\textbf{{{m}}}}}" for m in all_models)

lines: list[str] = []
lines.append(r"  \begin{tabular}{" + col_spec + r"}")
lines.append(r"    \toprule")
lines.append(
    r"    \textbf{Environment} & \textbf{\# Questions} & " + model_header + r" \\"
)
lines.append(r"    \midrule")

for qa_type in ["qa", "reasoning_qa"]:
    qt_label = QA_TYPE_LABELS.get(qa_type, qa_type)
    lines.append(
        rf"    \multicolumn{{{2 + n_model_cols}}}{{l}}{{\textit{{{qt_label}}}}} \\"
    )
    lines.append(r"    \midrule")

    envs_present = [e for e in ENV_ORDER if e in data.get(qa_type, {})]
    for env in envs_present:
        env_label = ENV_LABELS.get(env, env.capitalize())
        model_data = data[qa_type][env]

        # Use the n_questions from any model (they should match per env/qa_type)
        n_q_values = [v[1] for v in model_data.values()]
        n_q = n_q_values[0] if n_q_values else "{--}"

        # Find best score for bolding
        available_scores = {m: model_data[m][0] for m in all_models if m in model_data}
        best_score = max(available_scores.values()) if available_scores else None

        score_cells = []
        for model in all_models:
            if model in model_data:
                score = model_data[model][0]
                cell = f"{score:.2f}"
                if best_score is not None and score == best_score:
                    # Outer {} required by siunitx to treat the cell as text, not a number.
                    cell = r"{\textbf{" + cell + r"}}"
                score_cells.append(cell)
            else:
                score_cells.append("{--}")

        row = f"    {env_label} & {n_q} & " + " & ".join(score_cells) + r" \\"
        lines.append(row)

    if qa_type != list(QA_TYPE_LABELS)[-1]:
        lines.append(r"    \midrule")

lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")

tex_content = "\n".join(lines) + "\n"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(tex_content)
sys.stdout.write(f"Table written to: {OUT_PATH}\n\n{tex_content}")
