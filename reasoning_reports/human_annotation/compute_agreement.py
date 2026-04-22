"""Compute inter-annotator agreement for human annotations.

Comparisons for both `new` and `old` annotation batches:
  1. Human-Human: pairwise Cohen's kappa + PABAK between reviewers
  2. Human-LLM:   percent agreement between each reviewer and the LLM
     (the LLM always labels "correct", making Cohen's kappa degenerate)

Aggregation: global (all items pooled across files).

Outputs:
    - Logged summary
  - LaTeX tables  (reasoning_reports/human_annotation/agreement_tables.tex)
  - Bar-chart plot (reasoning_reports/human_annotation/agreement_plot.pdf)
"""

import itertools
import json
from pathlib import Path

import lama_aesthetics
import matplotlib.pyplot as plt
import numpy as np
from lama_aesthetics import ONE_COL_HEIGHT, TWO_COL_WIDTH
from lama_aesthetics.plotutils import range_frame
from loguru import logger
from sklearn.metrics import cohen_kappa_score

lama_aesthetics.get_style("main")

ANNOTATION_DIR = Path(__file__).parent
OUTPUT_DIR = ANNOTATION_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CORAL = "#E07A5F"
VIOLET_MID = "#8B5CF6"

BATCHES = {
    "new": "annotations_reviewer_{r}.json",
    "old": "old_annotations_reviewer_{r}.json",
}
REVIEWER_IDS = [1, 2, 3]


def load_reviewer(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def extract_decisions(file_data: dict, item_kind: str) -> dict[str, str]:
    """Return {item_id: decision} for nodes or edges, skipping None."""
    items = file_data.get(item_kind, {})
    out = {}
    for item_id, val in items.items():
        if isinstance(val, dict):
            dec = val.get("decision")
            if dec is not None:
                out[item_id] = dec
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    return cohen_kappa_score(a, b)


def pabak_score(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    return 2.0 * po - 1.0


def percent_agree(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / n


def compute_human_human(reviewers: dict[int, dict], item_kind: str) -> list[dict]:
    """Pairwise human-human agreement (global pooling)."""
    rows = []
    for ra, rb in itertools.combinations(sorted(reviewers), 2):
        files_a = reviewers[ra]["files"]
        files_b = reviewers[rb]["files"]
        common = sorted(set(files_a) & set(files_b))
        ga, gb = [], []
        for fname in common:
            da = extract_decisions(files_a[fname], item_kind)
            db = extract_decisions(files_b[fname], item_kind)
            shared = sorted(set(da) & set(db))
            ga.extend(da[i] for i in shared)
            gb.extend(db[i] for i in shared)
        rows.append(
            {
                "pair": f"R{ra} vs R{rb}",
                "kappa": cohen_kappa(ga, gb),
                "pabak": pabak_score(ga, gb),
                "pct_agree": percent_agree(ga, gb),
                "n": len(ga),
            }
        )
    return rows


def compute_human_llm(reviewers: dict[int, dict], item_kind: str) -> list[dict]:
    """Each reviewer vs LLM (global pooling). LLM = all 'correct'."""
    rows = []
    for rid in sorted(reviewers):
        human, llm = [], []
        for _fname, fdata in sorted(reviewers[rid]["files"].items()):
            dec = extract_decisions(fdata, item_kind)
            human.extend(dec.values())
            llm.extend(["correct"] * len(dec))
        rows.append(
            {
                "reviewer": f"R{rid}",
                "pct_agree": percent_agree(human, llm),
                "n": len(human),
            }
        )
    return rows


def _f(val, fmt=".3f") -> str:
    if val is None:
        return "--"
    return f"{val:{fmt}}"


def _f_tex(val, fmt=".3f") -> str:
    """Format a value for LaTeX, escaping % signs."""
    if val is None:
        return "--"
    s = f"{val:{fmt}}"
    return s.replace("%", r"\%")


def latex_human_human(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Pair & Cohen's $\kappa$ & PABAK & \% Agree & $n$ \\",
        r"\midrule",
    ]
    lines.extend(
        [
            f"  {r['pair']} & {_f_tex(r['kappa'])} & {_f_tex(r['pabak'])} "
            f"& {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in rows
        ]
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_human_human_combined(nodes_rows: list[dict], edges_rows: list[dict]) -> str:
    """Combined HH table with Nodes / Edges sections and an Overall row."""
    ncols = 5
    lines = [
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Pair & Cohen's $\kappa$ & PABAK & \% Agree & $n$ \\",
    ]
    lines.append(r"\midrule")
    lines.append(rf"  \multicolumn{{{ncols}}}{{c}}{{\textit{{Nodes}}}} \\")
    lines.append(r"\midrule")
    lines.extend(
        [
            f"  {r['pair']} & {_f_tex(r['kappa'])} & {_f_tex(r['pabak'])} "
            f"& {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in nodes_rows
        ]
    )
    lines.append(r"\midrule")
    lines.append(rf"  \multicolumn{{{ncols}}}{{l}}{{\textit{{Edges}}}} \\")
    lines.append(r"\midrule")
    lines.extend(
        [
            f"  {r['pair']} & {_f_tex(r['kappa'])} & {_f_tex(r['pabak'])} "
            f"& {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in edges_rows
        ]
    )
    all_rows = nodes_rows + edges_rows
    kappas = [r["kappa"] for r in all_rows if r["kappa"] is not None]
    pabaks = [r["pabak"] for r in all_rows if r["pabak"] is not None]
    agrees = [r["pct_agree"] for r in all_rows if r["pct_agree"] is not None]
    total_n = sum(r["n"] for r in all_rows)
    lines.append(r"\midrule")
    lines.append(
        f"  Overall & {_f_tex(np.mean(kappas) if kappas else None)} "
        f"& {_f_tex(np.mean(pabaks) if pabaks else None)} "
        f"& {_f_tex(np.mean(agrees) if agrees else None, '.1%')} "
        f"& {total_n} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_human_llm(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Reviewer & \% Agree & $n$ \\",
        r"\midrule",
    ]
    lines.extend(
        [
            f"  {r['reviewer']} & {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in rows
        ]
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def latex_human_llm_combined(nodes_rows: list[dict], edges_rows: list[dict]) -> str:
    """Combined HL table with Nodes / Edges sections and an Overall row."""
    ncols = 3
    lines = [
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Reviewer & \% Agree & $n$ \\",
    ]
    lines.append(r"\midrule")
    lines.append(rf"  \multicolumn{{{ncols}}}{{l}}{{\textit{{Nodes}}}} \\")
    lines.append(r"\midrule")
    lines.extend(
        [
            f"  {r['reviewer']} & {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in nodes_rows
        ]
    )
    lines.append(r"\midrule")
    lines.append(rf"  \multicolumn{{{ncols}}}{{c}}{{\textit{{Edges}}}} \\")
    lines.append(r"\midrule")
    lines.extend(
        [
            f"  {r['reviewer']} & {_f_tex(r['pct_agree'], '.1%')} & {r['n']} \\\\"
            for r in edges_rows
        ]
    )
    all_rows = nodes_rows + edges_rows
    agrees = [r["pct_agree"] for r in all_rows if r["pct_agree"] is not None]
    total_n = sum(r["n"] for r in all_rows)
    lines.append(r"\midrule")
    lines.append(
        f"  Overall & {_f_tex(np.mean(agrees) if agrees else None, '.1%')} "
        f"& {total_n} \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_plot(all_results: dict) -> None:
    """Bar chart: average Human-Human PABAK and Human-LLM %agree."""
    batches = list(all_results.keys())
    kinds = ["nodes", "edges"]

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH, ONE_COL_HEIGHT), sharey=True)

    x = np.arange(len(batches), dtype=float) * 0.6
    width = 0.15

    for ax, kind in zip(axes, kinds, strict=True):
        hh_vals, hl_vals = [], []
        for batch in batches:
            hh_rows = all_results[batch][kind]["hh"]
            pabaks = [r["pabak"] for r in hh_rows if r["pabak"] is not None]
            hh_vals.append(np.mean(pabaks) if pabaks else 0)

            hl_rows = all_results[batch][kind]["hl"]
            agrees = [r["pct_agree"] for r in hl_rows if r["pct_agree"] is not None]
            hl_vals.append(np.mean(agrees) if agrees else 0)

        bars1 = ax.bar(
            x - width / 2,
            hh_vals,
            width,
            label="Human-Human (PABAK)",
            color=CORAL,
        )
        bars2 = ax.bar(
            x + width / 2,
            hl_vals,
            width,
            label=r"Human-LLM (\% agree)",
            color=VIOLET_MID,
        )

        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
            )

        ax.set_title(kind.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in batches])
        ax.set_ylabel("Agreement" if ax == axes[0] else "")

        # `range_frame` gives the desired y-axis styling, but the x-axis is reset
        # below so the visible spine ends exactly at the grouped bars.
        range_frame(
            ax,
            x=np.array([x.min(), x.max()]),
            y=np.array([0, 1.0]),
            pad=0.02,
            nice=True,
        )
        ax.spines["bottom"].set_bounds(x.min(), x.max())
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in batches])
        ax.set_xlim(x.min() - width * 1.2, x.max() + width * 1.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
    )

    fig.tight_layout()
    out = OUTPUT_DIR / "agreement_plot.pdf"
    fig.savefig(out, bbox_inches="tight")
    out_png = OUTPUT_DIR / "agreement_plot.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved plot to {}", out)
    logger.info("Saved plot to {}", out_png)


def log_averages(all_results: dict) -> None:
    """Log corpus-level agreement averages across batches and item kinds."""
    all_hl, all_hh_agree, all_hh_kappa, all_hh_pabak = [], [], [], []
    for batch_data in all_results.values():
        for kind_data in batch_data.values():
            all_hl.extend(
                row["pct_agree"]
                for row in kind_data["hl"]
                if row["pct_agree"] is not None
            )
            all_hh_agree.extend(
                row["pct_agree"]
                for row in kind_data["hh"]
                if row["pct_agree"] is not None
            )
            all_hh_kappa.extend(
                row["kappa"] for row in kind_data["hh"] if row["kappa"] is not None
            )
            all_hh_pabak.extend(
                row["pabak"] for row in kind_data["hh"] if row["pabak"] is not None
            )
    logger.info(
        "\n{}",
        "\n".join(
            [
                f"Average human-LLM agreement: {np.mean(all_hl):.1%}",
                f"Average human-human agreement: {np.mean(all_hh_agree):.1%}",
                f"Average human-human Cohen's kappa: {np.mean(all_hh_kappa):.3f}",
                f"Average human-human PABAK: {np.mean(all_hh_pabak):.3f}",
            ]
        ),
    )


def _log_block(message: str) -> None:
    """Emit multi-line summaries as a single log record."""
    logger.info("\n{}", message)


def log_hh(title: str, rows: list[dict]) -> None:
    lines = [f"{'=' * 68}", f"  {title}", f"{'=' * 68}"]
    hdr = f"  {'Pair':<16} {'kappa':>8} {'PABAK':>8} {'%agree':>8} {'n':>6}"
    lines.extend([hdr, "-" * len(hdr)])
    lines.extend(
        [
            f"  {r['pair']:<16} "
            f"{_f(r['kappa']):>8} "
            f"{_f(r['pabak']):>8} "
            f"{_f(r['pct_agree'], '.1%'):>8} "
            f"{r['n']:>6}"
            for r in rows
        ]
    )
    _log_block("\n".join(lines))


def log_hl(title: str, rows: list[dict]) -> None:
    lines = [f"{'=' * 48}", f"  {title}", f"{'=' * 48}"]
    hdr = f"  {'Reviewer':<16} {'%agree':>8} {'n':>6}"
    lines.extend([hdr, "-" * len(hdr)])
    lines.extend(
        [
            f"  {r['reviewer']:<16} {_f(r['pct_agree'], '.1%'):>8} {r['n']:>6}"
            for r in rows
        ]
    )
    _log_block("\n".join(lines))


def main() -> None:
    all_results: dict[str, dict] = {}

    for batch_name, pattern in BATCHES.items():
        reviewers: dict[int, dict] = {}
        for rid in REVIEWER_IDS:
            p = ANNOTATION_DIR / pattern.format(r=rid)
            if not p.exists():
                logger.warning("Skipping missing reviewer file: {}", p.name)
                continue
            reviewers[rid] = load_reviewer(p)
            logger.info(
                "Loaded {} reviewer {}: {} files",
                batch_name,
                rid,
                len(reviewers[rid]["files"]),
            )

        if len(reviewers) < 2:
            logger.warning(
                "Skipping batch '{}' because fewer than two reviewers were loaded",
                batch_name,
            )
            continue

        all_results[batch_name] = {}

        for kind in ("nodes", "edges"):
            hh = compute_human_human(reviewers, kind)
            hl = compute_human_llm(reviewers, kind)
            all_results[batch_name][kind] = {"hh": hh, "hl": hl}

            log_hh(f"[{batch_name}] Human-Human ({kind})", hh)
            log_hl(f"[{batch_name}] Human-LLM ({kind})", hl)

            hh_path = OUTPUT_DIR / f"agreement_hh_{kind}_{batch_name}.tex"
            hh_path.write_text(latex_human_human(hh) + "\n")
            logger.info("Wrote {}", hh_path.name)

            hl_path = OUTPUT_DIR / f"agreement_hl_{kind}_{batch_name}.tex"
            hl_path.write_text(latex_human_llm(hl) + "\n")
            logger.info("Wrote {}", hl_path.name)

        nodes_hh = all_results[batch_name]["nodes"]["hh"]
        edges_hh = all_results[batch_name]["edges"]["hh"]
        nodes_hl = all_results[batch_name]["nodes"]["hl"]
        edges_hl = all_results[batch_name]["edges"]["hl"]

        hh_comb_path = OUTPUT_DIR / f"agreement_hh_{batch_name}.tex"
        hh_comb_path.write_text(latex_human_human_combined(nodes_hh, edges_hh) + "\n")
        logger.info("Wrote {}", hh_comb_path.name)

        hl_comb_path = OUTPUT_DIR / f"agreement_hl_{batch_name}.tex"
        hl_comb_path.write_text(latex_human_llm_combined(nodes_hl, edges_hl) + "\n")
        logger.info("Wrote {}", hl_comb_path.name)

    if all_results:
        log_averages(all_results)

    if all_results:
        make_plot(all_results)


if __name__ == "__main__":
    main()
