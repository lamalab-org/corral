"""Generate a LaTeX summary table giving, for each domain, the number of scopes,
tasks per scope, tool count, and typical trace length.

Trace length is computed as the average number of messages minus 2 (to exclude
the system and user prompts) across all task-level trials in ``traces.jsonl``
(downloaded via :mod:`download_traces_from_hf`).

Output: analysis/results/tables/domain_summary_table.tex
"""

import json
import sys
from pathlib import Path

import fire
from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from constants import ENV_MAP  # noqa: E402

DATA_PATH = _SCRIPT_DIR / "results" / "data" / "traces.jsonl"
OUT_PATH = _SCRIPT_DIR / "results" / "tables" / "domain_summary_table.tex"

# Canonical environment order and display labels.
ENV_ORDER = ["afm", "catalyst", "md", "ml", "resistor", "retro", "spectra", "wetlab"]

ENV_LABELS = {
    "afm": "AFM experiment execution",
    "catalyst": "Adsorption surface construction",
    "md": "Molecular simulation",
    "ml": "ML-based property prediction",
    "resistor": "Circuit inference",
    "retro": "Retrosynthetic planning",
    "spectra": "Spectroscopic structure elucidation",
    "wetlab": "Inorganic qualitative analysis",
}

# Number of scopes (difficulty levels) per environment.
ENV_SCOPES = {
    "afm": 4,
    "catalyst": 1,
    "md": 2,
    "ml": 1,
    "resistor": 1,
    "retro": 3,
    "spectra": 2,
    "wetlab": 3,
}

# Tasks per scope.  When the count is uniform across scopes a single int is
# used; otherwise a string like "2--3" encodes the range.
ENV_TASKS_PER_SCOPE = {
    "afm": "1",
    "catalyst": "3",
    "md": "2--3",
    "ml": "3",
    "resistor": "6",
    "retro": "8",
    "spectra": "20",
    "wetlab": "10",
}

# Number of tools available in each environment.
ENV_TOOL_COUNT = {
    "afm": 6,
    "catalyst": 15,
    "md": 8,
    "ml": 14,
    "resistor": 9,
    "retro": 15,
    "spectra": 16,
    "wetlab": 14,
}


def _compute_avg_trace_lengths(data_path: Path) -> dict[str, float]:
    """Return the mean trace length (messages - 2) per environment.

    Only rows with ``category`` in ``{"tasks", "task"}`` are considered
    (subtasks are excluded).  Environment names are mapped to their
    canonical short forms via :data:`constants.ENV_MAP`.
    """
    from collections import defaultdict

    env_msg_counts: dict[str, list[int]] = defaultdict(list)

    with data_path.open() as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            category = row.get("category", "")
            if category not in ("tasks", "task"):
                continue
            raw_env = row.get("env", "")
            env = ENV_MAP.get(raw_env, raw_env)
            messages = row.get("messages")
            if isinstance(messages, str):
                messages = json.loads(messages)
            if messages:
                trace_len = max(0, len(messages) - 2)
                env_msg_counts[env].append(trace_len)

    return {
        env: sum(counts) / len(counts)
        for env, counts in env_msg_counts.items()
        if counts
    }


def main(
    data_path: str | None = None,
    out_path: str | None = None,
) -> None:
    """Generate the domain summary LaTeX table.

    Args:
        data_path: Path to ``traces.jsonl``.  Defaults to the standard
            location under ``analysis/results/data/``.
        out_path: Destination for the generated ``.tex`` file.
    """
    data_path = Path(data_path) if data_path else DATA_PATH
    out_path = Path(out_path) if out_path else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    avg_trace: dict[str, float] = {}
    if data_path.exists():
        avg_trace = _compute_avg_trace_lengths(data_path)
        logger.info(f"Loaded trace lengths from {data_path}")
    else:
        logger.warning(f"{data_path} not found — trace-length column will show '--'")

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"  \toprule")
    lines.append(r"  Environment & Scopes & Tasks per scope & Tools & Trace length \\")
    lines.append(r"  \midrule")

    for env in ENV_ORDER:
        label = ENV_LABELS[env]
        scopes = ENV_SCOPES[env]
        tasks = ENV_TASKS_PER_SCOPE[env]
        tools = ENV_TOOL_COUNT[env]
        trace = avg_trace.get(env)
        trace_str = f"{trace:.1f}" if trace is not None else "--"

        lines.append(f"  {label} & {scopes} & {tasks} & {tools} & {trace_str} \\\\")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    tex = "\n".join(lines) + "\n"
    out_path.write_text(tex)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
