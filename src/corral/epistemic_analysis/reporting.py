"""Reporting for epistemic trace analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from corral.epistemic_analysis.patterns import (
    ANTIPATTERN_DESCRIPTIONS,
    AP_CONTRADICTION_WITHOUT_REPAIR,
    AP_DISCONNECTED_EVIDENCE,
    AP_EVIDENCE_NON_UPTAKE,
    AP_FIXED_BELIEF_TRACE,
    AP_ONE_SIDED_CONFIRMATION,
    AP_PRECOMMITTED_TEST_PLAN,
    AP_PREMATURE_COMMITMENT,
    AP_STALLED_REVISION,
    AP_UNINFORMATIVE_TEST,
    AP_UNSUPPORTED_JUDGMENT,
    AP_UNTESTED_CLAIM,
    SG_CONVERGENT_MULTI_TEST_EVIDENCE,
    SG_EVIDENCE_GUIDED_TEST_REDESIGN,
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
    SG_EXPLORE_THEN_TEST_TRANSITION,
    SG_FIXED_HYPOTHESIS_TEST_TUNING,
    SG_HYPOTHESIS_RERANKING,
    SG_REFUTATION_DRIVEN_BELIEF_REVISION,
    SUBGRAPH_DESCRIPTIONS,
)

from .patterns import ANTIPATTERN_NAMES, SUBGRAPH_NAMES

if TYPE_CHECKING:
    from .schema import AggregateResult

_TIKZ_STYLE_DEFS = r"""\definecolor{nodefill}{HTML}{DFE3E8}%
\tikzset{%
  rnode/.style={circle, draw, thick, minimum size=5mm, inner sep=1pt,
                font=\scriptsize\bfseries, fill=nodefill},%
  rlbl/.style={font=\tiny, midway, above},%
  rlblb/.style={font=\tiny, midway, below},%
  missing/.style={dashed, gray},%
  missingnode/.style={rnode, dashed, gray, text=gray, fill=none},%
}%
"""


def _make_tikz(body: str) -> str:
    """Wrap TikZ node/edge commands in a tikzpicture environment."""
    indented = body.strip().replace("\n", "\n  ")
    return (
        r"\begin{tikzpicture}"
        r"[>=Stealth, baseline=(current bounding box.center), node distance=7mm]"
        "\n  " + indented + "\n"
        r"\end{tikzpicture}"
    )


_TIKZ_SUBGRAPH_PATTERNS: dict[str, str] = {
    SG_REFUTATION_DRIVEN_BELIEF_REVISION: _make_tikz(
        r"""\node[rnode] (h1) {H};
\node[rnode, right=of h1] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[rnode, right=of j] (h2) {H$_2$};
\draw[->] (h1) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->] (h1) to[bend right=40] node[rlblb] {upd.} (h2);"""
    ),
    SG_FIXED_HYPOTHESIS_TEST_TUNING: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\draw[->] (h) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->, bend left=50] (j) to node[rlblb] {tests} (t);"""
    ),
    SG_EXPLORE_THEN_TEST_TRANSITION: _make_tikz(
        r"""\node[rnode] (t1) {T};
\node[rnode, right=of t1] (e) {E};
\node[right=4mm of e, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t2) {T};
\draw[->] (t1) -- node[rlbl] {obs.} (e);
\draw[->] (h) -- node[rlbl] {tests} (t2);"""
    ),
    SG_HYPOTHESIS_RERANKING: _make_tikz(
        r"""\node[rnode] (h1) {H$_1$};
\node[rnode, right=15mm of h1] (h2) {H$_2$};
\node[rnode, below left=5mm and 0mm of h1] (t1) {T};
\node[rnode, below right=5mm and 0mm of h2] (t2) {T};
\draw[<->, dashed] (h1) -- node[rlbl] {competes} (h2);
\draw[->] (h1) -- node[left, font=\tiny] {tests} (t1);
\draw[->] (h2) -- node[right, font=\tiny] {tests} (t2);"""
    ),
    SG_EVIDENCE_LED_HYPOTHESIS_GENERATION: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[right=4mm of j, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t) {T};
\draw[->] (e) -- node[rlbl] {inf.} (j);
\draw[->] (h) -- node[rlbl] {tests} (t);"""
    ),
    SG_CONVERGENT_MULTI_TEST_EVIDENCE: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=10mm of h, yshift=7mm] (t1) {T$_1$};
\node[rnode, right=10mm of h] (t2) {T$_2$};
\node[rnode, right=10mm of h, yshift=-7mm] (t3) {T$_3$};
\node[rnode, right=10mm of t2] (e) {E};
\draw[->] (h) -- (t1);
\draw[->] (h) -- (t2);
\draw[->] (h) -- (t3);
\draw[->] (t1) -- (e);
\draw[->] (t2) -- (e);
\draw[->] (t3) -- (e);"""
    ),
    SG_EVIDENCE_GUIDED_TEST_REDESIGN: _make_tikz(
        r"""\node[rnode] (j) {J};
\node[rnode, right=of j] (t) {T};
\node[rnode, right=of t] (e) {E};
\draw[->] (j) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);"""
    ),
}


_TIKZ_ANTIPATTERN_PATTERNS: dict[str, str] = {
    AP_UNTESTED_CLAIM: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[missingnode, right=of h] (t) {T};
\draw[->, missing] (h) -- node[rlbl, text=gray] {tests} (t);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);"""
    ),
    AP_EVIDENCE_NON_UPTAKE: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[missingnode, right=of e] (j) {J};
\draw[->, missing] (e) -- node[rlbl, text=gray] {inf.} (j);
\draw[red, thick] (j.north west) -- (j.south east);
\draw[red, thick] (j.north east) -- (j.south west);"""
    ),
    AP_UNSUPPORTED_JUDGMENT: _make_tikz(
        r"""\node[missingnode] (e) {E};
\node[rnode, right=of e] (j) {J};
\draw[->, missing] (e) -- node[rlbl, text=gray] {inf.} (j);
\draw[red, thick] (e.north west) -- (e.south east);
\draw[red, thick] (e.north east) -- (e.south west);"""
    ),
    AP_STALLED_REVISION: _make_tikz(
        r"""\node[rnode] (h1) {H};
\node[rnode, right=of h1] (h2) {H$_2$};
\node[right=7mm of h2, draw=none, font=\scriptsize, text=gray] (none) {$\varnothing$};
\draw[->] (h1) -- node[rlbl] {upd.} (h2);
\draw[->, missing] (h2) -- (none);"""
    ),
    AP_CONTRADICTION_WITHOUT_REPAIR: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[rnode, right=of e] (h) {H};
\node[missingnode, right=of h] (h2) {H$_2$};
\draw[->] (e) -- node[rlbl] {contr.} (h);
\draw[->, missing] (h) -- node[rlbl, text=gray] {upd.} (h2);
\draw[red, thick] (h2.north west) -- (h2.south east);
\draw[red, thick] (h2.north east) -- (h2.south west);"""
    ),
    AP_PREMATURE_COMMITMENT: _make_tikz(
        r"""\node[rnode] (j) {J};
\node[rnode, above right=5mm and 8mm of j] (h) {H};
\node[rnode, below right=5mm and 8mm of j] (c) {C};
\node[missingnode, right=of h] (t) {T};
\draw[->] (j) -- node[rlbl] {inf.} (h);
\draw[->] (j) -- node[rlblb] {inf.} (c);
\draw[->, missing] (h) -- node[rlbl, text=gray] {tests} (t);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);"""
    ),
    AP_UNINFORMATIVE_TEST: _make_tikz(
        r"""\node[rnode] (t) {T};
\node[missingnode, right=of t] (e) {E};
\draw[->, missing] (t) -- node[rlbl, text=gray] {obs.} (e);
\draw[red, thick] (e.north west) -- (e.south east);
\draw[red, thick] (e.north east) -- (e.south west);"""
    ),
    AP_FIXED_BELIEF_TRACE: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\node[rnode, right=of e] (j) {J};
\node[missingnode, below=5mm of h] (h2) {H$_2$};
\draw[->] (h) -- (t);
\draw[->] (t) -- (e);
\draw[->] (e) -- (j);
\draw[->, missing] (h) -- node[left, font=\tiny, text=gray] {upd.} (h2);
\draw[red, thick] (h2.north west) -- (h2.south east);
\draw[red, thick] (h2.north east) -- (h2.south west);"""
    ),
    AP_DISCONNECTED_EVIDENCE: _make_tikz(
        r"""\node[rnode] (e) {E};
\node[missingnode, left=of e] (t) {T};
\node[missingnode, right=of e] (j) {J};
\draw[->, missing] (t) -- (e);
\draw[->, missing] (e) -- (j);
\draw[red, thick] (t.north west) -- (t.south east);
\draw[red, thick] (t.north east) -- (t.south west);
\draw[red, thick] (j.north west) -- (j.south east);
\draw[red, thick] (j.north east) -- (j.south west);"""
    ),
    AP_ONE_SIDED_CONFIRMATION: _make_tikz(
        r"""\node[rnode] (h) {H};
\node[rnode, below left=5mm and 1mm of h] (es) {E};
\node[rnode, right=12mm of h] (j) {J};
\node[rnode, right=of j] (c) {C};
\node[missingnode, below right=5mm and 1mm of h] (ec) {E$_{\!c}$};
\draw[->] (es) -- node[left, font=\tiny] {inf.} (h);
\draw[->] (j) -- node[rlbl] {inf.} (h);
\draw[->] (j) -- node[rlbl] {inf.} (c);
\draw[->, missing] (ec) -- node[right, font=\tiny, text=gray] {contr.} (h);
\draw[red, thick] (ec.north west) -- (ec.south east);
\draw[red, thick] (ec.north east) -- (ec.south west);"""
    ),
    AP_PRECOMMITTED_TEST_PLAN: _make_tikz(
        r"""\node[rnode] (c) {C};
\node[right=4mm of c, draw=none, font=\scriptsize] (dots) {\ldots};
\node[rnode, right=4mm of dots] (h) {H};
\node[rnode, right=of h] (t) {T};
\node[rnode, right=of t] (e) {E};
\draw[->] (h) -- node[rlbl] {tests} (t);
\draw[->] (t) -- node[rlbl] {obs.} (e);"""
    ),
}


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in *text*."""
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("~", "\\textasciitilde{}")
    return text.replace("^", "\\textasciicircum{}")


def _pretty_name(raw: str) -> str:
    """Turn a snake_case identifier into a readable title."""
    text = raw.replace("_", " ")
    if not text:
        return raw
    result = text[0].upper() + text[1:].lower()
    return re.sub(r"/(.)", lambda m: "/" + m.group(1).upper(), result)


def _split_description(desc: str) -> tuple[str, str]:
    """Split a description of the form 'prose [graph notation].' into a (prose, graph) tuple."""
    m = re.search(r"\[([^\]]+)\]\s*\.?\s*$", desc)
    if m:
        graph = m.group(1)
        prose = desc[: m.start()].rstrip().rstrip(".")
        return prose, graph
    return desc.rstrip("."), ""


_SUBGRAPH_TABLE_ORDER: list[tuple[str | None, list[str]]] = [
    (
        "Hypothesis handling",
        [
            SG_EVIDENCE_LED_HYPOTHESIS_GENERATION,
            SG_HYPOTHESIS_RERANKING,
            SG_REFUTATION_DRIVEN_BELIEF_REVISION,
            SG_EXPLORE_THEN_TEST_TRANSITION,
        ],
    ),
    (
        "Evidence handling",
        [
            SG_CONVERGENT_MULTI_TEST_EVIDENCE,
        ],
    ),
    (
        "Inquiry control",
        [
            SG_FIXED_HYPOTHESIS_TEST_TUNING,
            SG_EVIDENCE_GUIDED_TEST_REDESIGN,
        ],
    ),
]


_ANTIPATTERN_TABLE_ORDER: list[tuple[str | None, list[str]]] = [
    (
        "Hypothesis handling",
        [
            AP_UNTESTED_CLAIM,
            AP_ONE_SIDED_CONFIRMATION,
            AP_CONTRADICTION_WITHOUT_REPAIR,
            AP_PREMATURE_COMMITMENT,
        ],
    ),
    (
        "Evidence handling",
        [
            AP_EVIDENCE_NON_UPTAKE,
            AP_DISCONNECTED_EVIDENCE,
            AP_UNSUPPORTED_JUDGMENT,
            AP_UNINFORMATIVE_TEST,
        ],
    ),
    (
        "Inquiry control",
        [
            AP_FIXED_BELIEF_TRACE,
            AP_PRECOMMITTED_TEST_PLAN,
            AP_STALLED_REVISION,
        ],
    ),
]


def _emit_group(
    merge_name: str | None,
    keys: list[str],
    descriptions: dict[str, str],
    tikz_patterns: dict[str, str] | None = None,
) -> list[str]:
    rows: list[str] = []
    n = len(keys)
    for i, key in enumerate(keys):
        prose, graph_text = _split_description(descriptions[key])
        prose = _latex_escape(prose)
        name = _pretty_name(key)
        name_and_desc = rf"\textbf{{{name}}}. {prose}"
        if tikz_patterns and key in tikz_patterns:
            graph = tikz_patterns[key]
        else:
            graph = _latex_escape(graph_text)
        if i == 0:
            rows.append(
                rf"\multirow{{{n}}}{{=}}{{{merge_name}}} & {graph} & {name_and_desc} \\"
            )
        else:
            rows.append(rf" & {graph} & {name_and_desc} \\")
    return rows


def build_productive_motifs_latex() -> str:
    """Return a LaTeX tabularx table with definitions of productive motifs.

    Column layout: Group (p{2.2cm}), Graph (TikZ picture, c), Name + Description (X).
    Patterns are grouped by reasoning capability (Hypothesis handling,
    Evidence handling, Inquiry control) matching the GROUPS structure used
    in analysis plots. Each graph cell contains an inline TikZ diagram.

    Returns:
        A string of LaTeX source for the complete tabularx environment,
        including TikZ style definitions prepended at the top.
    """
    lines: list[str] = []
    lines.append(_TIKZ_STYLE_DEFS)
    lines.append(r"\begin{tabularx}{\textwidth}{p{2.2cm}cX}")
    lines.append(r"\toprule")
    lines.append(r"Group & Graph & Description \\")
    lines.append(r"\midrule")

    for idx, (merge_name, keys) in enumerate(_SUBGRAPH_TABLE_ORDER):
        if idx:
            lines.append(r"\midrule[0.015em]")
        lines.extend(
            _emit_group(
                merge_name, keys, SUBGRAPH_DESCRIPTIONS, _TIKZ_SUBGRAPH_PATTERNS
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def build_reasoning_breakdowns_latex() -> str:
    """Return a LaTeX tabularx table with definitions of reasoning breakdowns.

    Column layout: Group (p{2.2cm}), Graph (TikZ picture, c), Name + Description (X).
    Patterns are grouped by reasoning capability (Hypothesis handling,
    Evidence handling, Inquiry control) matching the GROUPS structure used
    in analysis plots. Each graph cell contains an inline TikZ diagram.

    Returns:
        A string of LaTeX source for the complete tabularx environment,
        including TikZ style definitions prepended at the top.
    """
    lines: list[str] = []
    lines.append(_TIKZ_STYLE_DEFS)
    lines.append(r"\begin{tabularx}{\textwidth}{p{1.6cm}cX}")
    lines.append(r"\toprule")
    lines.append(r"Group & Graph & Description \\")
    lines.append(r"\midrule")

    for idx, (merge_name, keys) in enumerate(_ANTIPATTERN_TABLE_ORDER):
        if idx:
            lines.append(r"\midrule[0.03em]")
        lines.extend(
            _emit_group(
                merge_name, keys, ANTIPATTERN_DESCRIPTIONS, _TIKZ_ANTIPATTERN_PATTERNS
            )
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def write_pattern_definitions_latex(
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the pattern-definition LaTeX tables to *out_dir*.

    Produces two files: `productive_motifs.tex` and
    `reasoning_breakdowns.tex`. The output directory is created if it does
    not already exist.

    Args:
        out_dir: Destination directory for the two `.tex` files. Defaults
            to `./analysis/results/tables/`.

    Returns:
        A tuple of `(motifs_path, breakdowns_path)` pointing to the written
        files.
    """
    if out_dir is None:
        out_dir = Path("analysis") / "results" / "tables"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motifs_path = out_dir / "productive_motifs.tex"
    motifs_path.write_text(build_productive_motifs_latex(), encoding="utf-8")
    logger.info(f"Productive Motifs LaTeX table written to {motifs_path}")

    breakdowns_path = out_dir / "reasoning_breakdowns.tex"
    breakdowns_path.write_text(build_reasoning_breakdowns_latex(), encoding="utf-8")
    logger.info(f"Reasoning Breakdowns LaTeX table written to {breakdowns_path}")

    return motifs_path, breakdowns_path


def build_report_markdown(summary: AggregateResult) -> str:
    """Render versioned results with configuration, denominators, and origins."""
    import json

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Epistemic analysis",
        "",
        f"Selected traces: {summary.n_selected}",
        f"Analysis version: {summary.analysis_version}; pattern version: {summary.pattern_version}",
        "",
        "Measurement denominators include complete, nonempty annotations. Empty, incomplete, unsupported, missing, and failed annotations are counted separately.",
        "Annotation completion and context coverage are separate. With reduced context, omitted text is unknown; absence-based measurements do not establish absence in the full trace.",
        "",
    ]
    for identity, partition in summary.configurations.items():
        stats = partition["overall"]
        lines += [
            f"## Annotation configuration {identity}",
            "",
            "```json",
            json.dumps(
                partition["annotation_configuration"], indent=2, ensure_ascii=False
            ),
            "```",
            "",
            f"Eligible: {stats['n_traces']}; excluded: {stats['n_excluded']}; selected: {stats['n_selected']}.",
            f"Statuses: {json.dumps(stats['status_counts'], sort_keys=True)}",
            f"Reduced-context traces: {stats['n_reduced_context']}; clipped outputs: {stats['clipped_outputs']}; retained tool characters: {stats['retained_tool_chars']}; omitted: {stats['omitted_tool_chars']}.",
            "",
            "| Measurement | Local count | Global count | Global fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
        for prefix, names in (
            ("subgraph", SUBGRAPH_NAMES),
            ("antipattern", ANTIPATTERN_NAMES),
        ):
            for name in names:
                local = stats[f"{prefix}_presence_local"][name]
                glob = stats[f"{prefix}_presence_global"][name]
                fraction = (
                    f"{glob['fraction']:.3f}" if glob["fraction"] is not None else "—"
                )
                lines.append(
                    f"| {name} | {local['raw_total']} | {glob['raw_total']} | {fraction} |"
                )
        lines += [
            "",
            "### Groups",
            "",
            "| Metadata | Eligible | Excluded |",
            "| --- | ---: | ---: |",
        ]
        for group in partition["groups"]:
            lines.append(
                f"| {cell(json.dumps(group['metadata'], ensure_ascii=False))} | {group['statistics']['n_traces']} | {group['statistics']['n_excluded']} |"
            )
        lines += [
            "",
            "### Sources",
            "",
            "| Original input | Resolved trace | Status | Effective cap | Clipped outputs | Retained chars | Omitted chars | Reduced context |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for trace in partition["traces"]:
            source = trace["source"] or {}
            coverage = trace["statistics"]
            lines.append(
                f"| {cell(source.get('locator', trace['statistics']['input_file']))} | {cell(source.get('identity', 'unspecified'))} | {trace['status']} | {coverage['effective_tool_output_max_chars']} | {coverage['clipped_outputs']} | {coverage['retained_tool_chars']} | {coverage['omitted_tool_chars']} | {coverage['reduced_context']} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_reports(summary: AggregateResult, output: Path) -> None:
    """Write JSON, Markdown, and unchanged LaTeX/TikZ pattern definitions."""
    from .storage import safe_write_json

    output.mkdir(parents=True, exist_ok=True)
    safe_write_json(output / "annotation_summary.json", summary.to_dict())
    (output / "annotation_summary.md").write_text(
        build_report_markdown(summary), encoding="utf-8"
    )
    write_pattern_definitions_latex(output / "tables")
