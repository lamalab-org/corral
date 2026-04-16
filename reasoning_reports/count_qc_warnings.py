"""
Aggregate QC warning counts across all annotated JSON files.

Bins each warning string into one of the following categories:

  quote_mismatch_node   - PassA quote not found verbatim (node stage)
  quote_mismatch_edge   - PassB quote not found verbatim (edge stage)
  invalid_node_type     - node returned with a type outside the vocabulary
  invalid_edge_relation - edge returned with a relation outside the vocabulary
  edge_unknown_node     - edge references a node_id not in the node set
  disallowed_type_combo - edge (relation, src_type, dst_type) not permitted
  extra_observation_node- extra E/C node removed from an observation message
  node_changed_to_N     - node type changed to N to fix observation mismatch
  malformed_response    - Pass A/B returned a non-list (LLM formatting failure)
  other                 - anything else

Usage:
  python count_qc_warnings.py [root_dir ...]

  If no root_dir is given, defaults to the three model directories relative
  to this script: claude_sonnet_45/, gpt_4o/, gpt_oss_120b/

Output:
  A table of counts printed to stdout, and a JSON summary written to
  qc_warning_summary.json next to this script.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_ROOTS = [
    SCRIPT_DIR / "claude_sonnet_45",
    SCRIPT_DIR / "gpt_4o",
    SCRIPT_DIR / "gpt_oss_120b",
]

# Rules are evaluated in declaration order; the first match wins, so more
# specific patterns must precede broader ones (e.g. PassA/PassB quote checks
# before the catch-all "other").
_RULES: list[tuple[str, re.Pattern]] = [
    ("quote_mismatch_node", re.compile(r"^PassA\[.*\] Quote not found verbatim", re.I)),
    ("quote_mismatch_edge", re.compile(r"^PassB\[.*\] Quote not found verbatim", re.I)),
    ("invalid_node_type", re.compile(r"invalid node type", re.I)),
    ("invalid_edge_relation", re.compile(r"invalid edge relation", re.I)),
    ("edge_unknown_node", re.compile(r"edge references unknown node_id", re.I)),
    ("disallowed_type_combo", re.compile(r"disallowed type combination", re.I)),
    ("extra_observation_node", re.compile(r"removed extra node.*observation", re.I)),
    ("node_changed_to_N", re.compile(r"changed node.*from.*to N", re.I)),
    ("malformed_response", re.compile(r"returned non-list", re.I)),
]


def categorize(warning: str) -> str:
    for category, pattern in _RULES:
        if pattern.search(warning):
            return category
    return "other"


def collect(roots: list[Path]) -> dict:
    total_files = 0
    files_with_any_warning = 0

    global_counts: Counter[str] = Counter()
    per_model: dict[str, Counter] = defaultdict(Counter)
    other_examples: list[str] = []

    for root in roots:
        if not root.exists():
            logger.warning(f"[skip] {root} does not exist")
            continue
        model_name = root.name
        for path in sorted(root.rglob("*.annotated.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"[warn] could not read {path}: {exc}")
                continue

            total_files += 1
            warnings: list[str] = data.get("qc", {}).get("warnings", [])
            if warnings:
                files_with_any_warning += 1

            for w in warnings:
                cat = categorize(w)
                global_counts[cat] += 1
                per_model[model_name][cat] += 1
                if cat == "other" and len(other_examples) < 20:
                    other_examples.append(w)

    return {
        "total_files": total_files,
        "files_with_warnings": files_with_any_warning,
        "global_counts": dict(global_counts),
        "per_model_counts": {m: dict(c) for m, c in per_model.items()},
        "other_examples": other_examples,
    }


def print_table(summary: dict) -> None:
    gc = summary["global_counts"]
    total_warnings = sum(gc.values())

    # Verbatim-quote failures are shown first because they are the most common
    # QC signal; structural-correction categories follow; "other" is last.
    ORDER = [
        "quote_mismatch_node",
        "quote_mismatch_edge",
        "invalid_node_type",
        "invalid_edge_relation",
        "edge_unknown_node",
        "disallowed_type_combo",
        "extra_observation_node",
        "node_changed_to_N",
        "malformed_response",
        "other",
    ]

    logger.info(f"\n{'='*62}")
    logger.info(f"  QC warning summary across {summary['total_files']} annotated files")
    logger.info(f"  Files with ≥1 warning: {summary['files_with_warnings']}")
    logger.info(f"  Total warning instances: {total_warnings}")
    logger.info(f"{'='*62}")
    logger.info(f"  {'Category':<30}  {'Count':>7}  {'%':>6}")
    logger.info(f"  {'-'*30}  {'-'*7}  {'-'*6}")
    for cat in ORDER:
        n = gc.get(cat, 0)
        pct = 100.0 * n / total_warnings if total_warnings else 0.0
        logger.info(f"  {cat:<30}  {n:>7,}  {pct:>5.1f}%")
    logger.info(f"{'='*62}")

    if summary["per_model_counts"]:
        logger.info("\nPer-model breakdown:")
        models = sorted(summary["per_model_counts"])
        col_w = max(len(m) for m in models) + 2
        header = f"  {'Category':<30}" + "".join(f"  {m:>{col_w}}" for m in models)
        logger.info(header)
        logger.info(f"  {'-'*30}" + "".join(f"  {'-'*col_w}" for _ in models))
        for cat in ORDER:
            row = f"  {cat:<30}"
            for m in models:
                n = summary["per_model_counts"][m].get(cat, 0)
                row += f"  {n:>{col_w},}"
            logger.info(row)

    if summary["other_examples"]:
        logger.info("\nUnclassified 'other' warning examples (up to 20):")
        for ex in summary["other_examples"]:
            logger.info(f"  - {ex[:120]}")


_CATEGORY_LABELS: dict[str, str] = {
    "quote_mismatch_node": "Non-verbatim quote (node stage)",
    "quote_mismatch_edge": "Non-verbatim quote (edge stage)",
    "invalid_node_type": "Invalid node type",
    "invalid_edge_relation": "Invalid edge relation",
    "edge_unknown_node": "Edge references unknown node",
    "disallowed_type_combo": "Disallowed (relation, src, dst) combination",
    "extra_observation_node": "Extra node at observation message (removed)",
    "node_changed_to_N": "Node type corrected to enforce E-only rule",
    "malformed_response": "Malformed LLM response (non-list)",
    "other": "Other (node additions / type corrections)",
}

_MODEL_LABELS: dict[str, str] = {
    "claude_sonnet_45": r"\claudesonnet",
    "gpt_4o": r"\gptfour",
    "gpt_oss_120b": r"\gptoss",
}

ORDER = [
    "quote_mismatch_node",
    "quote_mismatch_edge",
    "disallowed_type_combo",
    "extra_observation_node",
    "node_changed_to_N",
    "other",
    "invalid_node_type",
    "invalid_edge_relation",
    "edge_unknown_node",
    "malformed_response",
]

# Categories with zero counts across all models are collapsed into a single
# "None observed" row at the bottom to keep the table compact.
_ZERO_LABEL = "Schema violations (none observed)"


def write_latex(summary: dict, out_path: Path) -> None:
    gc = summary["global_counts"]
    total_warnings = sum(gc.values())
    models = sorted(summary["per_model_counts"])
    model_labels = [_MODEL_LABELS.get(m, m.replace("_", r"\_")) for m in models]

    n_model_cols = len(models)
    # column spec: description | per-model counts | total | %
    col_spec = "l" + "c" * n_model_cols + "cc"

    lines: list[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"  \toprule")

    # Header
    header_cells = ["Failure mode", *model_labels, "Total", r"\%"]
    lines.append("  " + " & ".join(header_cells) + r" \\")
    lines.append(r"  \midrule")

    zero_cats: list[str] = []

    for cat in ORDER:
        n_total = gc.get(cat, 0)
        per_model_vals = [
            summary["per_model_counts"].get(m, {}).get(cat, 0) for m in models
        ]

        # Collect zero-across-all rows separately
        if n_total == 0:
            zero_cats.append(cat)
            continue

        pct = 100.0 * n_total / total_warnings if total_warnings else 0.0
        label = _CATEGORY_LABELS.get(cat, cat.replace("_", r"\_"))
        cells = (
            [label]
            + [f"{v:,}" for v in per_model_vals]
            + [f"{n_total:,}", f"{pct:.1f}"]
        )
        lines.append("  " + " & ".join(cells) + r" \\")

    # Collapsed zero row
    if zero_cats:
        zero_cells = [_ZERO_LABEL] + ["0"] * n_model_cols + ["0", "0.0"]
        lines.append(r"  \midrule")
        lines.append("  " + " & ".join(zero_cells) + r" \\")

    lines.append(r"  \midrule")
    # Total row
    model_totals = [
        sum(summary["per_model_counts"].get(m, {}).values()) for m in models
    ]
    total_cells = (
        [r"\textbf{Total}"]
        + [f"\\textbf{{{v:,}}}" for v in model_totals]
        + [f"\\textbf{{{total_warnings:,}}}", "100.0"]
    )
    lines.append("  " + " & ".join(total_cells) + r" \\")
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    tex = "\n".join(lines) + "\n"
    out_path.write_text(tex, encoding="utf-8")
    logger.info(f"LaTeX tabular written to {out_path}")


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_ROOTS
    summary = collect(roots)
    print_table(summary)

    out_path = SCRIPT_DIR / "qc_warning_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"\nJSON summary written to {out_path}")

    tex_path = SCRIPT_DIR / "qc_warning_table.tex"
    write_latex(summary, tex_path)


if __name__ == "__main__":
    main()
