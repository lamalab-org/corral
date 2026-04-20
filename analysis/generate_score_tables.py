"""Generate LaTeX score tables from reports data — one tabular per verbosity level.

Each generated block contains only the ``tabular`` environment (no surrounding
``table`` float) so the caller can add ``\\caption``, ``\\label``, and any
float options they prefer.

Rows:   Environment x Scope (S1, S2, ...), grouped by environment category.
Columns: Model x Agent type (3 models x 2 agents = 6 data columns).

Usage:

    # Generate all three verbosity tables → analysis/results/figures/score_tables/
    python analysis/generate_score_tables.py

    # Only the "brief" verbosity
    python analysis/generate_score_tables.py --verbosity=brief

    # Different metric (pass_at_k with k=5)
    python analysis/generate_score_tables.py --metric=pass_at_k --k_value=5

    # Custom output directory
    python analysis/generate_score_tables.py --output_dir=my_tables
"""

from pathlib import Path

import fire
import pandas as pd
from loguru import logger
from plot_config import (
    AGENT_NAMES,
    ENVIRONMENT_GROUPS,
    ENVIRONMENT_MAX_LEVELS,
    ENVIRONMENT_NAMES,
    MODEL_NAMES,
)
from plot_utils import (
    filter_by_task_type,
    filter_by_verbosity,
    get_metric_column_name,
    load_reports_data,
)

OUT_DIR = Path(__file__).parent / "results" / "tables" / "score_tables"

# Short environment names used in the table rows
_SHORT_ENV_NAMES: dict[str, str] = {
    "afm": "AFM Exp.",
    "catalyst": "Adsorption Surf.",
    "md": "Mol. Simulation",
    "ml": "ML Property",
    "resistor": "Circuit Inf.",
    "retro": "Retrosynthesis",
    "spectra": "Spectra Eluc.",
    "wetlab": "Inorg. Qual. Analysis",
}


def _env_display_name(env: str) -> str:
    return _SHORT_ENV_NAMES.get(env, ENVIRONMENT_NAMES.get(env, env).replace("\n", " "))


def _ordered_env_levels() -> list[tuple[str, str, int]]:
    """Return (group_name, env, level) triples in the canonical display order."""
    return [
        (group_name, env, level)
        for group_name, group_info in ENVIRONMENT_GROUPS.items()
        for env in group_info["environments"]
        for level in range(1, ENVIRONMENT_MAX_LEVELS.get(env, 1) + 1)
    ]


def build_tabular(
    df: pd.DataFrame,
    verbosity: str,
    metric_column: str,
) -> str:
    """Return a LaTeX ``tabular`` block (no surrounding table float).

    Args:
        df:            Full reports dataframe (all verbosities).
        verbosity:     One of ``"brief"``, ``"workflow"``, ``"comprehensive"``.
        metric_column: DataFrame column holding the score values.

    Returns:
        Multi-line LaTeX string starting with ``\\footnotesize`` and ending
        after ``\\end{tabular}``.
    """
    vdf = filter_by_verbosity(df, verbosity)
    if vdf.empty:
        return f"% No data for verbosity={verbosity}\n"

    vdf = vdf.copy()
    vdf["config"] = vdf["model"] + "|" + vdf["agent_type"]
    vdf["env_level"] = vdf["environment"] + "-" + vdf["level"].astype(str)

    pivot = vdf.pivot_table(
        values=metric_column,
        index="env_level",
        columns="config",
        aggfunc="mean",
    )

    # Keep all canonical env/level combos; missing data will render as {--}
    env_levels = _ordered_env_levels()

    # Determine ordered column configs: model-major, agent-minor
    model_order = list(MODEL_NAMES.keys())
    agent_order = list(AGENT_NAMES.keys())
    col_configs = [
        (m, a, f"{m}|{a}")
        for m in model_order
        for a in agent_order
        if f"{m}|{a}" in pivot.columns
    ]
    n_data_cols = len(col_configs)

    # --- Column specification ---
    #   col 1: l  (environment name)
    #   col 2: l  (scope label)
    #   remaining: fixed-width centred data columns
    data_col = r"w{c}{2.1em}"
    col_spec = r"@{}l l " + " ".join([data_col] * n_data_cols) + r"@{}"

    # --- Model header row (spanning cmidrules) ---
    model_spans: list[tuple[str, int, int]] = []
    cur_model: str | None = None
    cur_start = 0
    for i, (m, _a, _cfg) in enumerate(col_configs):
        if m != cur_model:
            if cur_model is not None:
                model_spans.append((cur_model, cur_start, i - 1))
            cur_model = m
            cur_start = i
    if cur_model is not None:
        model_spans.append((cur_model, cur_start, n_data_cols - 1))

    col_offset = 3  # two fixed left cols → data cols start at LaTeX col 3
    model_header_cells = [r"\textbf{Environment}", r"\textbf{Scope}"]
    model_cmidrules: list[str] = []
    for m, ms, me in model_spans:
        span = me - ms + 1
        display = MODEL_NAMES.get(m, m)
        model_header_cells.append(
            rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{display}}}}}"
        )
        model_cmidrules.append(rf"\cmidrule(lr){{{col_offset + ms}-{col_offset + me}}}")
    model_header_row = " & ".join(model_header_cells) + r" \\"
    model_cmidrule_str = " ".join(model_cmidrules)

    # --- Agent sub-header row ---
    agent_abbrev = {"react": "R", "tool_calling": "TC"}
    agent_header_cells = [r"\multicolumn{2}{l}{}"]
    for _m, a, _cfg in col_configs:
        agent_header_cells.append(agent_abbrev.get(a, a))
    agent_header_row = " & ".join(agent_header_cells) + r" \\"

    # --- Data rows ---
    data_rows: list[str] = []
    prev_group: str | None = None
    prev_env: str | None = None

    for group_name, env, level in env_levels:
        env_key = f"{env}-{level}"

        # Collect data values for this row
        raw_vals: list[float] = []
        for _m, _a, cfg in col_configs:
            val = (
                pivot.loc[env_key, cfg]
                if env_key in pivot.index and cfg in pivot.columns
                else float("nan")
            )
            raw_vals.append(val)

        numeric_vals = [v for v in raw_vals if not pd.isna(v)]
        row_max = max(numeric_vals) if numeric_vals else float("nan")

        data_vals: list[str] = []
        for val in raw_vals:
            if pd.isna(val):
                data_vals.append("{--}")
            else:
                formatted = f"{val:.2f}"
                if val == row_max:
                    formatted = rf"\textbf{{{formatted}}}"
                data_vals.append(formatted)

        # Skip rows where every data cell is empty
        if all(v == "{--}" for v in data_vals):
            continue

        # Group separator / header (emitted only when we have a real row)
        if prev_group is None:
            data_rows.append(
                rf"\multicolumn{{{2 + n_data_cols}}}{{l}}"
                rf"{{\textit{{{group_name}}}}}"
                r" \\"
            )
            data_rows.append(r"\midrule")
        elif group_name != prev_group:
            data_rows.append(r"\midrule")
            data_rows.append(
                rf"\multicolumn{{{2 + n_data_cols}}}{{l}}"
                rf"{{\textit{{{group_name}}}}}"
                r" \\"
            )
            data_rows.append(r"\midrule")

        # Environment name (only on first row for that env)
        env_cell = _env_display_name(env) if env != prev_env else ""

        cells = [env_cell, f"S{level}", *data_vals]

        data_rows.append(" & ".join(cells) + r" \\")
        prev_group = group_name
        prev_env = env

    # --- Assemble the tabular block ---
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        model_header_row,
        model_cmidrule_str,
        agent_header_row,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main(
    verbosity: str | None = None,
    metric: str = "average_score",
    k_value: int = 5,
    output_dir: str | None = None,
) -> None:
    """Generate LaTeX score tabular blocks and write them to .tex files.

    Produces one file per (verbosity, task_type) combination, e.g.
    ``score_table_brief_tasks.tex`` and ``score_table_brief_subtasks.tex``.

    Args:
        verbosity:
            Which verbosity level(s) to generate. One of ``"brief"``,
            ``"workflow"``, ``"comprehensive"``, or ``None`` (all three).
        metric:
            Score metric: ``"average_score"``, ``"pass_at_k"``, or
            ``"pass_hat_k"``.
        k_value:
            K value used when ``metric`` is ``"pass_at_k"`` or
            ``"pass_hat_k"``.
        output_dir:
            Directory for output ``.tex`` files.  Defaults to
            ``analysis/results/tables/score_tables/``.
    """
    out_dir = Path(output_dir) if output_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_column = get_metric_column_name(metric, k_value)
    reports_df = load_reports_data()

    verbosities = [verbosity] if verbosity else ["brief", "workflow", "comprehensive"]

    for task_type in ["tasks", "subtasks"]:
        filtered_df = filter_by_task_type(reports_df, task_type)
        logger.info(f"Loaded {len(filtered_df)} rows (task_type={task_type})")

        for verb in verbosities:
            logger.info(
                f"Building score table for verbosity={verb}, task_type={task_type}"
            )
            tabular = build_tabular(filtered_df, verb, metric_column)
            out_path = out_dir / f"score_table_{verb}_{task_type}.tex"
            header = (
                f"% Auto-generated score table — verbosity: {verb}, task_type: {task_type}\n"
                "% Add \\begin{table}[H], \\caption{}, \\label{}, \\centering\n"
                "% around this block as needed.  Requires: booktabs package.\n\n"
            )
            out_path.write_text(header + tabular + "\n")
            logger.success(f"Saved: {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
