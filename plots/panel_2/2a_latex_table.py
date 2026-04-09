"""Generate LaTeX tables for full coverage data — one table per verbosity level.

Each table has:
  - Rows: Environment-level combinations, grouped by environment group
  - Columns: Model x Agent configurations (6 columns: 3 models x 2 agents)

Usage:
    python 2a_latex_table.py --task_type_strategy=both
"""

import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "plots"))

from plot_config import (  # noqa: E402
    AGENT_NAMES,
    ENVIRONMENT_GROUPS,
    ENVIRONMENT_MAX_LEVELS,
    ENVIRONMENT_NAMES,
    MODEL_NAMES,
)
from plot_utils import (  # noqa: E402
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

VERBOSITY_DISPLAY = {
    "brief": "Brief",
    "workflow": "Workflow",
    "comprehensive": "Comprehensive",
}


def _ordered_env_levels() -> list[tuple[str, str, int]]:
    """Return (group_name, env, level) triples in display order."""
    return [
        (group_name, env, level)
        for group_name, group_info in ENVIRONMENT_GROUPS.items()
        for env in group_info["environments"]
        for level in range(1, ENVIRONMENT_MAX_LEVELS.get(env, 1) + 1)
    ]


def _env_display_name(env: str) -> str:
    """Single-line short environment display name."""
    short_names = {
        "afm": "AFM Exp.",
        "catalyst": "Adsorption Surf.",
        "md": "Mol. Simulation",
        "ml": "ML Property",
        "resistor": "Circuit Inf.",
        "retro": "Retrosynthesis",
        "spectra": "Spectra Eluc.",
        "wetlab": "Inorg. Qual. Analysis",
    }
    return short_names.get(env, ENVIRONMENT_NAMES.get(env, env).replace("\n", " "))


def build_table_for_verbosity(
    df: pd.DataFrame,
    verbosity: str,
    metric_column: str,
) -> str:
    """Build a LaTeX table string for one verbosity level.

    Rows = environment-level combinations (grouped by environment group).
    Columns = model x agent configurations.
    """
    # Filter to this verbosity
    vdf = filter_by_verbosity(df, verbosity)
    if vdf.empty:
        return f"% No data for verbosity={verbosity}\n"

    # Pivot: rows = env-level, columns = model + agent_type
    vdf = vdf.copy()
    vdf["config"] = vdf["model"] + "|" + vdf["agent_type"]
    vdf["env_level"] = vdf["environment"] + "-" + vdf["level"].astype(str)

    pivot = vdf.pivot_table(
        values=metric_column,
        index="env_level",
        columns="config",
        aggfunc="mean",
    )

    # Ordered env-level triples (rows)
    env_levels = _ordered_env_levels()
    env_levels = [(g, e, lv) for g, e, lv in env_levels if f"{e}-{lv}" in pivot.index]

    # Ordered column configs: model order x agent order
    model_order = list(MODEL_NAMES.keys())
    agent_order = list(AGENT_NAMES.keys())
    col_configs = []
    for m in model_order:
        for a in agent_order:
            cfg = f"{m}|{a}"
            if cfg in pivot.columns:
                col_configs.append((m, a, cfg))

    n_model_cols = len(col_configs)

    # ---- Build column spec ----
    # 2 row-header columns (Environment, Level) + fixed-width data columns
    # p{} columns with centering give truly equal widths unaffected by headers
    data_col = r"S[table-format=1.2]"
    col_spec = r"l l " + " ".join([data_col] * n_model_cols)

    # ---- Build header rows ----
    # Row 1: Model names spanning their agent columns
    model_spans = []
    cur_model = None
    cur_start = 0
    for i, (m, _a, _cfg) in enumerate(col_configs):
        if m != cur_model:
            if cur_model is not None:
                model_spans.append((cur_model, cur_start, i - 1))
            cur_model = m
            cur_start = i
    if cur_model is not None:
        model_spans.append((cur_model, cur_start, len(col_configs) - 1))

    col_offset = 3  # first data column is column 3 in LaTeX (1-indexed)

    model_header_cells = [r"\multicolumn{2}{l}{}"]
    model_cmidrules = []
    for m, ms, me in model_spans:
        span = me - ms + 1
        model_display = MODEL_NAMES.get(m, m)
        model_header_cells.append(
            rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{model_display}}}}}"
        )
        model_cmidrules.append(rf"\cmidrule(lr){{{col_offset + ms}-{col_offset + me}}}")
    model_header_row = " & ".join(model_header_cells) + r" \\"
    model_cmidrule_str = " ".join(model_cmidrules)

    # Row 2: Abbreviated agent names (R = ReAct, TC = Tool calling)
    # Wrap in {} to protect from siunitx S-column parsing
    agent_abbrev = {"react": "{R}", "tool_calling": "{TC}"}
    agent_header_cells = [r"\textbf{Environment}", r"\textbf{Lvl}"]
    for _m, a, _cfg in col_configs:
        agent_header_cells.append(agent_abbrev.get(a, a))
    agent_header_row = " & ".join(agent_header_cells) + r" \\"

    # ---- Build data rows (environments as rows, grouped) ----
    data_rows = []
    prev_group = None
    prev_env = None
    for group_name, env, level in env_levels:
        env_key = f"{env}-{level}"

        # Group separator
        if prev_group is not None and group_name != prev_group:
            data_rows.append(r"\midrule")
            data_rows.append(
                rf"\multicolumn{{{2 + n_model_cols}}}{{l}}"
                rf"{{\textit{{{group_name}}}}}"
                r" \\[2pt]"
            )
        elif prev_group is None:
            data_rows.append(
                rf"\multicolumn{{{2 + n_model_cols}}}{{l}}"
                rf"{{\textit{{{group_name}}}}}"
                r" \\[2pt]"
            )

        # Environment name on first level row only, blank for subsequent levels
        if env != prev_env:
            env_display = _env_display_name(env)
            env_cell = env_display
        else:
            env_cell = ""

        cells = [env_cell, f"S{level}"]
        for _m, _a, cfg in col_configs:
            val = pivot.loc[env_key, cfg] if cfg in pivot.columns else float("nan")
            if pd.isna(val):
                cells.append("{--}")
            else:
                cells.append(f"{val:.2f}")

        data_rows.append(" & ".join(cells) + r" \\")
        prev_group = group_name
        prev_env = env

    verbosity_label = VERBOSITY_DISPLAY.get(verbosity, verbosity)

    # ---- Assemble table ----
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Average Score by Environment — {verbosity_label} verbosity. "
        r"R\,=\,ReAct, TC\,=\,Tool Calling.}",
        rf"\label{{tab:coverage_{verbosity}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        model_header_row,
        model_cmidrule_str,
        agent_header_row,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(
    task_type_strategy: str = "both",
    metric: str = "average_score",
    k_value: int = 5,
    output_filename: str | None = None,
) -> None:
    """Generate LaTeX tables for all three verbosity levels.

    Args:
        task_type_strategy: "tasks", "subtasks", or "both"
        metric: "average_score", "pass_at_k", or "pass_hat_k"
        k_value: K value for Pass@k / Pass^k
        output_filename: Output .tex file path
    """
    logger.info("Generating LaTeX coverage tables")

    metric_column = get_metric_column_name(metric, k_value)
    reports_df = load_reports_data()
    filtered_df = filter_by_task_type(reports_df, task_type_strategy)

    logger.info(f"Filtered to {len(filtered_df)} rows (task_type={task_type_strategy})")

    tables = []
    for verbosity in ["brief", "workflow", "comprehensive"]:
        logger.info(f"Building table for verbosity={verbosity}")
        table_str = build_table_for_verbosity(filtered_df, verbosity, metric_column)
        tables.append(table_str)

    output = (
        "% Auto-generated LaTeX tables — do not edit by hand\n"
        "% One table per verbosity level (Brief, Workflow, Comprehensive)\n"
        "% Requires: booktabs, siunitx packages\n\n" + "\n\n".join(tables) + "\n"
    )

    if output_filename is None:
        output_path = Path(__file__).parent / "2a_coverage_tables.tex"
    else:
        output_path = Path(output_filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)
    logger.success(f"Saved: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
