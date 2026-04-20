"""
Extract short annotated trace excerpts for the four major reasoning-breakdown
categories and emit a TikZ figure (standalone LaTeX) with one panel per
category.

Categories
----------
1. evidence_non_uptake   - agent gathers a result and does not incorporate it
2. untested_claim        - hypothesis stated without a designed test
3. fixed_belief_trace    - agent persists with a belief after contradictory
                           evidence (no updates_to in entire trace)
4. contradiction_without_repair - two incompatible claims coexist without
                                  reconciliation

Output: analysis/results/figures/antipattern_excerpts.tex
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

ANNOTATED_GLOBS = [
    "reasoning_reports/claude_sonnet_45/**/annotated/*.annotated.json",
    "reasoning_reports/gpt_4o/**/annotated/*.annotated.json",
    "reasoning_reports/gpt-4o/**/annotated/*.annotated.json",
]
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "analysis" / "results" / "figures" / "antipattern_excerpts.tex"

# How many characters of each quote to keep (per support entry).
MAX_QUOTE_LEN = 90
# How many characters of the node description in bullet points.
MAX_TEXT_LEN = 160

CATEGORY_LABELS = {
    "evidence_non_uptake": "Evidence non-uptake",
    "untested_claim": "Untested claim",
    "fixed_belief_trace": "Fixed belief trace",
    "contradiction_without_repair": "Contradiction without repair",
}

CATEGORY_DESCRIPTIONS = {
    "evidence_non_uptake": (
        "Agent gathers a result and does not incorporate it into "
        "subsequent reasoning."
    ),
    "untested_claim": ("Hypothesis stated without a designed test to evaluate it."),
    "fixed_belief_trace": (
        "Agent persists with an initial belief; no hypothesis revision "
        "(\\texttt{updates\\_to}) appears in the entire trace."
    ),
    "contradiction_without_repair": (
        "Two incompatible claims coexist without reconciliation: evidence "
        "contradicts a hypothesis, but the hypothesis is never revised."
    ),
}

# Pastel category colours (xcolor, RGB).
CATEGORY_COLOURS = {
    "evidence_non_uptake": "198,219,239",  # light blue
    "untested_claim": "253,205,172",  # light orange
    "fixed_belief_trace": "203,213,232",  # light lavender
    "contradiction_without_repair": "252,187,161",  # light red
}


@dataclass
class MessageExcerpt:
    """One relevant message from a trace (role + abbreviated content)."""

    msg_idx: int
    role: str
    content_snippet: str  # trimmed, LaTeX-safe


@dataclass
class ExcerptPanel:
    """All the information needed to render one panel in the figure."""

    category: str
    model: str
    env: str
    level: str
    task: str
    trial: int
    # The node(s) involved
    node_ids: list[str]
    node_texts: list[str]
    node_types: list[str]
    # The quote(s) that drove the annotation
    quotes: list[str]
    # (Optional) surrounding messages from the trace
    messages: list[MessageExcerpt] = field(default_factory=list)


def _build_index(nodes, edges):
    node_by_id = {n["node_id"]: n for n in nodes}
    out_edges: dict[str, list] = {}
    in_edges: dict[str, list] = {}
    for e in edges:
        out_edges.setdefault(e["src"], []).append(e)
        in_edges.setdefault(e["dst"], []).append(e)
    return node_by_id, out_edges, in_edges


def _parse_metadata(annotated_path: str):
    """Extract model, env, level, task, trial from the annotated file path."""
    # Path pattern: reasoning_reports/<model>/<env>/<level>/annotated/<task>-<trial>.annotated.json
    parts = Path(annotated_path).parts
    # Find 'reasoning_reports' index
    try:
        idx = list(parts).index("reasoning_reports")
    except ValueError:
        idx = 0
    model = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    env = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
    level = parts[idx + 3] if idx + 3 < len(parts) else "unknown"
    fname = parts[-1].replace(".annotated.json", "")
    # Split task-trial on the last hyphen
    m = re.match(r"^(.+)-(\d+)$", fname)
    if m:
        task, trial = m.group(1), int(m.group(2))
    else:
        task, trial = fname, 0
    return model, env, level, task, trial


def _resolve_trace_path(annotated_data: dict, annotated_path: str) -> Path | None:
    """Find the original trace file from annotated data or sibling dir."""
    input_file = annotated_data.get("input_file", "")
    # Try relative to ROOT
    for prefix in ["", "reasoning_reports/"]:
        # input_file may start with 'corral/reasoning_reports/...'
        cleaned = re.sub(r"^corral/", "", input_file)
        candidate = ROOT / prefix / cleaned
        if candidate.exists():
            return candidate
    # Fallback: go from annotated/ up one level
    ann_p = Path(annotated_path)
    candidate = ann_p.parent.parent / ann_p.name.replace(".annotated.json", ".json")
    if candidate.exists():
        return candidate
    return None


def _load_messages(trace_path: Path) -> list[dict]:
    with trace_path.open() as f:
        data = json.load(f)
    return data.get("messages", [])


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("$", "\\$"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("<", "\\textless{}"),
        (">", "\\textgreater{}"),
        ("\u03b3", "$\\gamma$"),
        ("\u00b2", "$^2$"),
        ("\u00c5", "\\AA{}"),
        ("\u2013", "--"),
        ("\u2014", "---"),
        ("\u2018", "`"),
        ("\u2019", "'"),
        ("\u201c", "``"),
        ("\u201d", "''"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _truncate(text: str, maxlen: int) -> str:
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 3] + "..."


def _msg_role_label(msg: dict) -> str:
    role = msg.get("role", "unknown")
    name = msg.get("name", "")
    if role == "user" and name:
        return f"Observation ({name})"
    if role == "assistant":
        return "Agent"
    if role == "system":
        return "System"
    return role.capitalize()


def _msg_snippet(msg: dict, max_len: int = 200) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    # Strip XML-like thought tags for brevity
    content = re.sub(r"</?thought>", "", content).strip()
    return _truncate(content, max_len)


def _is_prose_quote(q: str) -> bool:
    """Return True if the quote is readable prose, not code or raw data."""
    if not q or len(q) < 20:
        return False
    # Reject if it contains code-like characters
    code_chars = sum(1 for c in q if c in "{}[]()=;#<>|/\\_")
    if code_chars / max(len(q), 1) >= 0.04:
        return False
    # Reject if it has function calls, assignments, or comment lines
    if re.search(r"\w+\.\w+\(", q):  # e.g. scan.Start()
        return False
    if re.search(r"^\s*#", q, re.MULTILINE):  # comment lines
        return False
    if re.search(r"\w+\s*=\s*\w+\.\w+", q):  # assignment from method
        return False
    # Reject structured data: dict keys, list literals, error messages
    if re.search(r"['\"][^'\"]+['\"]:\s", q):  # dict keys like 'key':
        return False
    if re.search(r"\['", q) or re.search(r"'\]", q):  # list of strings
        return False
    if re.search(r"error\b", q, re.IGNORECASE):  # error messages
        return False
    # Reject if it's mostly numbers, dots, or formula data
    if sum(c.isdigit() or c == "." for c in q) / max(len(q), 1) > 0.15:
        return False
    # Reject CIF / structured data
    if any(
        kw in q.lower() for kw in ("_cell_", "_chemical_", "data_", "m/z ", "deltas ")
    ):
        return False
    # Must have spaces (prose has words)
    return not q.count(" ") < 4


def _abbreviate_assistant_content(
    content: str,
    max_len: int = 180,
) -> tuple[str, str | None]:
    """Extract thought text and optional tool-call string from an assistant msg."""
    content_clean = content.strip()
    thought_match = re.search(r"<thought>(.*?)</thought>", content_clean, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    action_match = re.search(r"<action>(.*?)</action>", content_clean, re.DOTALL)
    action_name = action_match.group(1).strip() if action_match else None
    if not thought and not action_name:
        thought = re.sub(r"\s+", " ", content_clean)
    else:
        thought = re.sub(r"\s+", " ", thought)
    thought = _truncate(thought, max_len)
    tool_call = f"{action_name}(...)" if action_name else None
    return thought, tool_call


def _abbreviate_observation_content(content: str, max_len: int = 120) -> str:
    """Parse a tool-observation message and return an abbreviated string."""
    content_clean = content.strip()
    if content_clean.lower().startswith("observation:"):
        content_clean = content_clean.split(":", 1)[1].strip()
    tool_match = re.search(r"'tool_name':\s*'([^']+)'", content_clean)
    result_match = re.search(r"'result':\s*['\"](.+?)['\"]", content_clean)
    if not result_match:
        result_match = re.search(
            r"'result':\s*(.+?)(?:,\s*'|\})", content_clean, re.DOTALL
        )
    tool_name = tool_match.group(1) if tool_match else ""
    result = result_match.group(1).strip().strip("'\"") if result_match else ""
    result = re.sub(r"\s+", " ", result)
    if result:
        avail = max(max_len - len(tool_name) - 5, 30)
        result = _truncate(result, avail)
        return f"{tool_name}: {result}"
    return f"{tool_name}: ..."


def _extract_panel_messages(
    trace_path: Path,
    msg_idxs: list[int],
    window: int = 0,
    max_msg_len: int = 180,
) -> list[MessageExcerpt]:
    """Load trace and return abbreviated MessageExcerpts around *msg_idxs*."""
    messages = _load_messages(trace_path)
    if not messages:
        return []
    all_idxs: set[int] = set()
    for idx in msg_idxs:
        for offset in range(-window, window + 1):
            i = idx + offset
            if 2 <= i < len(messages):  # skip system (0) and task guide (1)
                all_idxs.add(i)
    excerpts: list[MessageExcerpt] = []
    for i in sorted(all_idxs):
        msg = messages[i]
        role = msg.get("role", "unknown")
        name = msg.get("name")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if role == "assistant":
            thought, tool_call = _abbreviate_assistant_content(content, max_msg_len)
            snippet = thought
            if tool_call:
                snippet += "\n" + tool_call
            role_label = "Agent"
        elif name:
            snippet = _abbreviate_observation_content(content, max_msg_len)
            role_label = "Obs."
        else:
            snippet = _truncate(re.sub(r"\s+", " ", content).strip(), max_msg_len)
            role_label = role.capitalize()
        excerpts.append(
            MessageExcerpt(msg_idx=i, role=role_label, content_snippet=snippet)
        )
    return excerpts


def _msg_verbosity_score(
    annotated_data: dict,
    annotated_path: str,
    msg_idxs: list[int],
) -> float:
    """Total character length of messages at *msg_idxs*.  Lower = better."""
    trace_path = _resolve_trace_path(annotated_data, annotated_path)
    if trace_path is None:
        return float("inf")
    messages = _load_messages(trace_path)
    total = 0
    for idx in msg_idxs:
        if 0 <= idx < len(messages):
            c = messages[idx].get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            total += len(c)
    return float(total)


def _pick_best_and_attach(
    candidates: list[tuple[ExcerptPanel, str, dict, list[int]]],
    window: int = 0,
    max_msg_len: int = 180,
) -> ExcerptPanel | None:
    """Pick the least-verbose qualifying candidate and attach message excerpts.

    Scores each candidate by the total character length of messages at the
    relevant indices; candidates whose trace file cannot be resolved are
    skipped. Message excerpts are attached in-place on the returned panel.

    Args:
        candidates: List of `(panel, annotated_path, annotated_data,
            msg_idxs)` tuples produced by the per-category scan functions.
        window: Number of context messages to include on each side of each
            msg_idx.
        max_msg_len: Maximum character length per message snippet.

    Returns:
        The panel from the least-verbose qualifying candidate with its
        `messages` field populated, or `None` if no candidate succeeds.
    """
    if not candidates:
        return None
    scored: list[tuple[float, int, ExcerptPanel, str, dict, list[int]]] = []
    for i, (panel, ann_path, ann_data, msg_idxs) in enumerate(candidates):
        score = _msg_verbosity_score(ann_data, ann_path, msg_idxs)
        scored.append((score, i, panel, ann_path, ann_data, msg_idxs))
    scored.sort(key=lambda x: x[0])
    for score, _, panel, ann_path, ann_data, msg_idxs in scored:
        if score == float("inf"):
            continue
        trace_path = _resolve_trace_path(ann_data, ann_path)
        if trace_path is None:
            continue
        panel.messages = _extract_panel_messages(
            trace_path,
            msg_idxs,
            window=window,
            max_msg_len=max_msg_len,
        )
        if panel.messages:
            return panel
    return None


def _find_evidence_non_uptake(
    files: list[str],
    exclude_envs: set[str] | None = None,
) -> ExcerptPanel | None:
    """Find an E node with no outgoing informs to J or H.  Prefer short msgs."""
    exclude_envs = exclude_envs or set()
    candidates: list[tuple[ExcerptPanel, str, dict, list[int]]] = []
    for fp in files:
        model, env, level, task, trial = _parse_metadata(fp)
        if env in exclude_envs:
            continue
        with Path(fp).open() as f:
            data = json.load(f)
        nodes, edges = data.get("nodes", []), data.get("edges", [])
        if not nodes:
            continue
        node_by_id, out_edges, in_edges = _build_index(nodes, edges)
        node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

        for ev_id, ntype in node_type_map.items():
            if ntype != "E":
                continue
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) in {"J", "H"}
                for e in out_edges.get(ev_id, [])
            )
            if used:
                continue
            node = node_by_id[ev_id]
            support = node.get("support", [])
            if not support:
                continue
            quote = support[0].get("quote", "")
            if len(quote) < 20:
                continue
            msg_idx = support[0].get("msg_idx", 2)
            # Evidence msg + next assistant msg that ignores it
            context_idxs = [msg_idx, msg_idx + 1]
            panel = ExcerptPanel(
                category="evidence_non_uptake",
                model=model,
                env=env,
                level=level,
                task=task,
                trial=trial,
                node_ids=[ev_id],
                node_texts=[node.get("text", "")],
                node_types=["E"],
                quotes=[quote],
            )
            candidates.append((panel, fp, data, context_idxs))
    return _pick_best_and_attach(candidates)


def _find_untested_claim(
    files: list[str],
    exclude_envs: set[str] | None = None,
) -> ExcerptPanel | None:
    """Find an H node with no tests edges.  Prefer short msgs."""
    exclude_envs = exclude_envs or set()
    candidates: list[tuple[ExcerptPanel, str, dict, list[int]]] = []
    for fp in files:
        model, env, level, task, trial = _parse_metadata(fp)
        if env in exclude_envs:
            continue
        with Path(fp).open() as f:
            data = json.load(f)
        nodes, edges = data.get("nodes", []), data.get("edges", [])
        if not nodes:
            continue
        node_by_id, out_edges, in_edges = _build_index(nodes, edges)
        node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

        for h_id, ntype in node_type_map.items():
            if ntype != "H":
                continue
            tested = any(e.get("relation") == "tests" for e in out_edges.get(h_id, []))
            if not tested:
                tested = any(
                    e.get("relation") == "tests" for e in in_edges.get(h_id, [])
                )
            if tested:
                continue
            node = node_by_id[h_id]
            support = node.get("support", [])
            if not support:
                continue
            quote = support[0].get("quote", "")
            if len(quote) < 20:
                continue
            msg_idx = support[0].get("msg_idx", 2)
            context_idxs = [msg_idx, msg_idx + 1]
            panel = ExcerptPanel(
                category="untested_claim",
                model=model,
                env=env,
                level=level,
                task=task,
                trial=trial,
                node_ids=[h_id],
                node_texts=[node.get("text", "")],
                node_types=["H"],
                quotes=[quote],
            )
            candidates.append((panel, fp, data, context_idxs))
    return _pick_best_and_attach(candidates)


def _find_fixed_belief_trace(
    files: list[str],
    exclude_envs: set[str] | None = None,
) -> ExcerptPanel | None:
    """Find a trace with H nodes but zero updates_to edges.  Prefer short msgs."""
    exclude_envs = exclude_envs or set()
    candidates: list[tuple[ExcerptPanel, str, dict, list[int]]] = []
    for fp in files:
        model, env, level, task, trial = _parse_metadata(fp)
        if env in exclude_envs:
            continue
        with Path(fp).open() as f:
            data = json.load(f)
        nodes, edges = data.get("nodes", []), data.get("edges", [])
        if not nodes:
            continue
        node_by_id, out_edges, in_edges = _build_index(nodes, edges)

        n_updates = sum(1 for e in edges if e.get("relation") == "updates_to")
        h_nodes = [n for n in nodes if n.get("type") == "H"]
        if n_updates > 0 or len(h_nodes) < 2:
            continue

        first_h = h_nodes[0]
        last_h = h_nodes[-1]
        quotes = []
        for h in (first_h, last_h):
            s = h.get("support", [])
            if s:
                quotes.append(s[0].get("quote", ""))

        if not any(len(q) > 20 for q in quotes):
            continue

        # Collect msg_idxs for first and last hypothesis
        context_idxs = []
        for h in (first_h, last_h):
            s = h.get("support", [])
            if s:
                context_idxs.append(s[0].get("msg_idx", 2))

        panel = ExcerptPanel(
            category="fixed_belief_trace",
            model=model,
            env=env,
            level=level,
            task=task,
            trial=trial,
            node_ids=[first_h["node_id"], last_h["node_id"]],
            node_texts=[first_h.get("text", ""), last_h.get("text", "")],
            node_types=["H", "H"],
            quotes=quotes,
        )
        candidates.append((panel, fp, data, context_idxs))
    return _pick_best_and_attach(candidates)


def _find_contradiction_without_repair(
    files: list[str],
    exclude_envs: set[str] | None = None,
) -> ExcerptPanel | None:
    """Find E/J -> H contradicts edge where H is never revised.  Prefer short msgs."""
    exclude_envs = exclude_envs or set()
    candidates: list[tuple[ExcerptPanel, str, dict, list[int]]] = []
    for fp in files:
        model, env, level, task, trial = _parse_metadata(fp)
        if env in exclude_envs:
            continue
        with Path(fp).open() as f:
            data = json.load(f)
        nodes, edges = data.get("nodes", []), data.get("edges", [])
        if not nodes:
            continue
        node_by_id, out_edges, in_edges = _build_index(nodes, edges)
        node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}

        for e in edges:
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if node_type_map.get(dst) != "H":
                continue
            resolved = any(
                ie.get("relation") in ("updates_to", "competes_with")
                for ie in in_edges.get(dst, [])
            )
            if not resolved:
                resolved = any(
                    oe.get("relation") == "competes_with"
                    for oe in out_edges.get(dst, [])
                )
            if resolved:
                continue

            src_node = node_by_id[e["src"]]
            dst_node = node_by_id[dst]
            edge_support = e.get("support", [])
            src_support = src_node.get("support", [])

            quotes = []
            if edge_support:
                quotes.append(edge_support[0].get("quote", ""))
            elif src_support:
                quotes.append(src_support[0].get("quote", ""))
            if not quotes or len(quotes[0]) < 20:
                continue

            # Gather msg_idxs from both nodes
            context_idxs = [
                sup_list[0].get("msg_idx", 2)
                for sup_list in (
                    src_node.get("support", []),
                    dst_node.get("support", []),
                )
                if sup_list
            ]
            if not context_idxs and edge_support:
                context_idxs.append(edge_support[0].get("msg_idx", 2))

            panel = ExcerptPanel(
                category="contradiction_without_repair",
                model=model,
                env=env,
                level=level,
                task=task,
                trial=trial,
                node_ids=[e["src"], dst],
                node_texts=[src_node.get("text", ""), dst_node.get("text", "")],
                node_types=[node_type_map[e["src"]], "H"],
                quotes=quotes,
            )
            candidates.append((panel, fp, data, context_idxs))
    return _pick_best_and_attach(candidates)


# Each category gets a small TikZ schematic with circular node-type badges,
# short labels beneath, and clearly spaced arrows. Ghost nodes and missing
# edges are shown dashed with a red cross to highlight the absent step.

_TIKZ_DIAGRAM = {
    "evidence_non_uptake": r"""
\node[ncirc, fill={col}!40] (E) {{\textsf{{E}}}};
\node[ncirc, draw=gray!40, text=gray!40, right=40pt of E] (JH) {{\textsf{{J}}}};
\draw[->, gray!40, dashed, thick] (E) -- node[above, font=\tiny, text=gray] {{informs}} (JH);
\node[red!70, font=\bfseries\scriptsize] at ($(E)!0.5!(JH)+(0,-0.22)$) {{\texttimes}};
""",
    "untested_claim": r"""
\node[ncirc, fill={col}!40] (H) {{\textsf{{H}}}};
\node[ncirc, draw=gray!40, text=gray!40, right=40pt of H] (T) {{\textsf{{T}}}};
\draw[->, gray!40, dashed, thick] (H) -- node[above, font=\tiny, text=gray] {{tests}} (T);
\node[red!70, font=\bfseries\scriptsize] at ($(H)!0.5!(T)+(0,-0.22)$) {{\texttimes}};
""",
    "fixed_belief_trace": r"""
\node[ncirc, fill={col}!40] (H1) {{\textsf{{H}}$_1$}};
\node[ncirc, fill={col}!40, right=40pt of H1] (H2) {{\textsf{{H}}$_n$}};
\draw[->, gray!40, dashed, thick] (H1) -- node[above, font=\tiny, text=gray] {{updates\_to}} (H2);
\node[red!70, font=\bfseries\scriptsize] at ($(H1)!0.5!(H2)+(0,-0.22)$) {{\texttimes}};
""",
    "contradiction_without_repair": r"""
\node[ncirc, fill={col}!40] (Ev) {{\textsf{{{src_type}}}}};
\node[ncirc, fill={col}!40, right=40pt of Ev] (H) {{\textsf{{H}}}};
\draw[->, thick, red!60!black] (Ev) -- node[above, font=\tiny] {{contradicts}} (H);
\node[font=\tiny, text=gray!50] at ($(Ev)!0.5!(H)+(0,-0.22)$) {{no \textsf{{updates\_to}}}};
""",
}


def _render_panel_tikz(panel: ExcerptPanel, panel_idx: int, row: int = 0) -> str:
    """Render one compact panel: title, TikZ schematic, message conversation, meta."""
    cat = panel.category
    label = CATEGORY_LABELS[cat]
    colour_rgb = CATEGORY_COLOURS[cat]
    col = f"cat{panel_idx}"

    src_type = panel.node_types[0] if panel.node_types else "E"
    diagram = _TIKZ_DIAGRAM[cat].format(col=col, src_type=src_type)

    msg_items = []
    prev_idx = None
    for msg in panel.messages:
        # Insert vertical dots if messages are far apart
        if prev_idx is not None and msg.msg_idx - prev_idx > 2:
            msg_items.append("    \\item[] {\\scriptsize\\color{gray} $\\vdots$}")
        prev_idx = msg.msg_idx
        role_tex = _tex_escape(msg.role)
        lines = msg.content_snippet.split("\n", 1)
        main_text = _tex_escape(lines[0])
        item = f"    \\item[{role_tex}] {{\\scriptsize {main_text}}}"
        if len(lines) > 1:
            tool_text = _tex_escape(lines[1])
            item += (
                f"\n    {{\\newline\\tiny\\color{{gray}}" f"\\texttt{{{tool_text}}}}}"
            )
        msg_items.append(item)

    msg_block = (
        "\\begin{description}["
        "font=\\sffamily\\bfseries\\scriptsize, "
        "leftmargin=30pt, style=sameline, "
        "itemsep=3pt, parsep=0pt, topsep=2pt]\n" + "\n".join(msg_items) + "\n"
        "\\end{description}\n"
    )

    meta = (
        f"{_tex_escape(panel.env)} / "
        f"{_tex_escape(panel.level)} / "
        f"trial\\,{panel.trial}"
    )

    return (
        f"% — Panel {panel_idx + 1}: {label} —\n"
        f"\\definecolor{{{col}}}{{RGB}}{{{colour_rgb}}}\n"
        f"\\begin{{tcolorbox}}[\n"
        f"  colback={col}!8, colframe={col}!60!black,\n"
        f"  title={{\\small\\sffamily\\bfseries {label}}},\n"
        f"  fontupper=\\scriptsize,\n"
        f"  left=4pt, right=4pt, top=2pt, bottom=10pt,\n"
        f"  boxrule=0.4pt, arc=1.5pt,\n"
        f"  equal height group=row{row},\n"
        f"  enhanced,\n"
        f"  overlay={{\n"
        f"    \\node[anchor=south east, inner sep=4pt,\n"
        f"          font=\\tiny\\sffamily, text=gray]\n"
        f"      at (frame.south east)\n"
        f"      {{{meta}}};\n"
        f"  }},\n"
        f"]\n"
        f"\\centering\n"
        f"\\begin{{tikzpicture}}[\n"
        f"  ncirc/.style={{circle, draw, inner sep=0pt, minimum size=14pt,\n"
        f"                 font=\\scriptsize\\bfseries}},\n"
        f"  node distance=4pt,\n"
        f"]\n"
        f"{diagram}"
        f"\\end{{tikzpicture}}\\\\[3pt]\n"
        f"{msg_block}"
        f"\\end{{tcolorbox}}\n"
    )


def render_figure(panels: list[ExcerptPanel]) -> str:
    """Produce a standalone LaTeX document with a 2x2 grid."""
    blocks = [_render_panel_tikz(p, i, row=i // 2) for i, p in enumerate(panels)]

    top_row = (
        "\\begin{minipage}[t]{0.48\\textwidth}\n"
        f"{blocks[0]}"
        "\\end{minipage}%\n"
        "\\hfill\n"
        "\\begin{minipage}[t]{0.48\\textwidth}\n"
        f"{blocks[1]}"
        "\\end{minipage}\n"
    )
    bot_row = ""
    if len(blocks) > 2:
        bot_row = (
            "\\vspace{3pt}\n"
            "\\begin{minipage}[t]{0.48\\textwidth}\n"
            f"{blocks[2]}"
            "\\end{minipage}%\n"
            "\\hfill\n"
            "\\begin{minipage}[t]{0.48\\textwidth}\n"
            f"{blocks[3] if len(blocks) > 3 else ''}"
            "\\end{minipage}\n"
        )

    return (
        "% Auto-generated by analysis/generate_antipattern_excerpt_figure.py\n"
        "% Compile with: lualatex antipattern_excerpts.tex\n"
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{tcolorbox}\n"
        "\\usepackage{tikz}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{amssymb}\n"
        "\\usepackage{textcomp}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{microtype}\n"
        "\\usetikzlibrary{positioning,calc}\n"
        "\\tcbuselibrary{skins}\n\n"
        "\\begin{document}\n"
        "\\begin{minipage}{\\textwidth}\n"
        f"{top_row}"
        f"{bot_row}"
        "\\end{minipage}\n"
        "\\end{document}\n"
    )


def render_includable(pdf_relpath: str) -> str:
    """Produce a LaTeX figure* fragment that includes the compiled standalone PDF.

    The fragment uses `\\includegraphics` to embed the PDF compiled from the
    standalone document, allowing it to be incorporated into a larger document
    without recompiling TikZ.

    Args:
        pdf_relpath: Relative path (without `.pdf` extension) to the compiled
            standalone PDF, used as the `\\includegraphics` argument.

    Returns:
        A string of LaTeX source for the complete `figure*` environment.
    """
    return (
        "% Auto-generated by analysis/generate_antipattern_excerpt_figure.py\n"
        "% Include in paper: \\input{antipattern_excerpts_includable.tex}\n"
        "% Requires: \\usepackage{graphicx}\n"
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{pdf_relpath}}}\n"
        "\\caption{\\textbf{Annotated trace excerpts illustrating the four major\n"
        "reasoning-breakdown categories.}\n"
        "Each panel shows one antipattern: \\textit{evidence non-uptake} (agent\n"
        "gathers a result and does not incorporate it), \\textit{untested claim}\n"
        "(hypothesis stated without a designed test), \\textit{fixed belief trace}\n"
        "(agent persists with a belief after contradictory evidence), and\n"
        "\\textit{contradiction without repair} (two incompatible claims coexist\n"
        "without reconciliation).  Relevant trace messages, annotated nodes, and\n"
        "the supporting quote that drove the annotation are shown.  Model,\n"
        "environment, scope, and trial index are noted under each panel.}\n"
        "\\label{fig:antipattern-excerpts}\n"
        "\\end{figure*}\n"
    )


# Each entry specifies the exact annotated trace file and node IDs to use for
# one antipattern panel. Set HAND_CURATED = None to fall back to the automatic
# scanning algorithm that picks the least-verbose qualifying excerpt.
# Keys per entry:
#   annotated_path  - path relative to ROOT
#   node_ids        - list of node IDs to highlight
#   msg_idxs        - message indices to include in the conversation excerpt
#   window          - extra context messages on each side of each msg_idx
HAND_CURATED: dict[str, dict] | None = {
    # NMR spectra, level 1 (Claude Sonnet 4.5):
    # Agent calls obtain_isomers_from_molecular_formula and gets a list of 20
    # candidate structures for C18H14O2 — including the correct answer — but
    # completely ignores this list and instead guesses specific structures one
    # by one (phenanthrene, anthracene, …).  The potentially key evidence
    # simply hangs with no outgoing informs edge.
    "evidence_non_uptake": {
        "annotated_path": "reasoning_reports/claude_sonnet_45/spectra/level_1/annotated/10_15227_orgsyn_096_0036-20.annotated.json",
        "node_ids": ["N14"],
        "msg_idxs": [11, 14],
        "window": 0,
    },
    # 1H NMR analysis, level 1 (Claude Sonnet 4.5):
    # Agent reads the NMR spectrum and immediately hypothesises "two coupled
    # CH2 groups, acetyl group at 2.23 ppm, substituted benzene ring" — a
    # rich structural story — but never designs a test for this claim.
    # Instead it jumps straight to mass spectrometry to get the molecular
    # formula, treating the NMR hypothesis as fact.
    "untested_claim": {
        "annotated_path": "reasoning_reports/claude_sonnet_45/spectra/level_1/annotated/10_15227_orgsyn_084_0317m-22.annotated.json",
        "node_ids": ["N3"],
        "msg_idxs": [4, 5],
        "window": 0,
    },
    # LAMMPS molecular dynamics, silicon melting, Stillinger-Weber potential,
    # level 2 (GPT-4o):
    # The trace has 6 hypothesis nodes and ZERO updates_to edges.  The agent
    # opens believing the SW potential file is at /potentials/SW (a directory,
    # not a file), and this belief is never revised even after three different
    # error messages.  A second wrong hypothesis ("lost atoms = heating rate")
    # appears later and is equally never updated.
    "fixed_belief_trace": {
        "annotated_path": "reasoning_reports/gpt_4o/md/level_2/annotated/silicon_melting_2-33.annotated.json",
        "node_ids": ["N12", "N35"],
        "msg_idxs": [16, 28],
        "window": 0,
    },
    # NMR structure elucidation, level 2 (Claude Sonnet 4.5):
    # Agent proposes the isopropyl ester CC(C)OC(=O)c1ccccc1N(C)C.  When it
    # simulates the spectra, the result shows a 6H doublet at 1.46 ppm — but
    # the experimental spectrum shows only a 3H doublet at 1.43 ppm.  The
    # agent explicitly notes "*this is not isopropyl ester*" yet never revises
    # the hypothesis and submits the isopropyl ester as the final answer (N38).
    "contradiction_without_repair": {
        "annotated_path": "reasoning_reports/claude_sonnet_45/spectra/level_2/annotated/55_55555_orgsyn_555_5555-45.annotated.json",
        "node_ids": ["N26", "N30"],
        "msg_idxs": [18, 20],
        "window": 0,
    },
}


def _build_panel_from_curated(category: str, spec: dict) -> ExcerptPanel | None:
    """Construct an ExcerptPanel directly from a hand-curated specification."""
    ann_path = ROOT / spec["annotated_path"]
    if not ann_path.exists():
        logger.warning(f"Curated file not found: {ann_path}")
        return None
    with ann_path.open() as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    node_by_id = {n["node_id"]: n for n in nodes}

    wanted_ids = spec["node_ids"]
    node_texts, node_types, quotes = [], [], []
    for nid in wanted_ids:
        n = node_by_id.get(nid)
        if n is None:
            logger.warning(f"Node {nid} not found in {ann_path.name}")
            continue
        node_texts.append(n.get("text", ""))
        node_types.append(n.get("type", "?"))
        sup = n.get("support", [])
        if sup:
            quotes.append(sup[0].get("quote", ""))

    model, env, level, task, trial = _parse_metadata(str(ann_path))
    panel = ExcerptPanel(
        category=category,
        model=model,
        env=env,
        level=level,
        task=task,
        trial=trial,
        node_ids=wanted_ids,
        node_texts=node_texts,
        node_types=node_types,
        quotes=quotes,
    )
    trace_path = _resolve_trace_path(data, str(ann_path))
    if trace_path is not None:
        panel.messages = _extract_panel_messages(
            trace_path,
            spec["msg_idxs"],
            window=spec.get("window", 0),
            max_msg_len=spec.get("max_msg_len", 120),
        )
    return panel


def main() -> None:
    panels: list[ExcerptPanel] = []

    if HAND_CURATED is not None:
        logger.info("Using hand-curated examples")
        order = [
            "evidence_non_uptake",
            "untested_claim",
            "fixed_belief_trace",
            "contradiction_without_repair",
        ]
        for name in order:
            spec = HAND_CURATED.get(name)
            if spec is None:
                logger.warning(f"No curated spec for '{name}'")
                continue
            panel = _build_panel_from_curated(name, spec)
            if panel is None:
                logger.warning(f"Could not build panel for '{name}'")
            else:
                logger.info(
                    f"{name}: {panel.model}/{panel.env}/{panel.level} "
                    f"trial {panel.trial}  nodes={panel.node_ids}"
                )
                panels.append(panel)
    else:
        all_files: list[str] = []
        for pattern in ANNOTATED_GLOBS:
            all_files.extend(sorted(str(p) for p in ROOT.glob(pattern)))
        logger.info(f"Scanning {len(all_files)} annotated trace files")

        finders = [
            ("evidence_non_uptake", _find_evidence_non_uptake),
            ("untested_claim", _find_untested_claim),
            ("fixed_belief_trace", _find_fixed_belief_trace),
            ("contradiction_without_repair", _find_contradiction_without_repair),
        ]

        used_envs: set[str] = set()
        for name, finder in finders:
            panel = finder(all_files, exclude_envs=used_envs)
            if panel is None:
                panel = finder(all_files)
            if panel is None:
                logger.warning(f"No suitable excerpt found for '{name}'")
            else:
                used_envs.add(panel.env)
                logger.info(
                    f"{name}: {panel.model}/{panel.env}/{panel.level} "
                    f"trial {panel.trial}  nodes={panel.node_ids}"
                )
                panels.append(panel)

    if not panels:
        logger.warning("No excerpts found - nothing to write.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write the standalone LaTeX source (compiled separately as a standalone document).
    standalone_tex = render_figure(panels)
    OUT_PATH.write_text(standalone_tex)
    logger.info(f"Standalone figure -> {OUT_PATH}")

    # Compile to PDF so it can be embedded via \includegraphics in the paper.
    pdf_path = OUT_PATH.with_suffix(".pdf")
    compiled = False
    for engine in ("lualatex", "pdflatex", "xelatex"):
        if shutil.which(engine) is None:
            continue
        logger.info(f"Compiling with {engine}")
        # Run twice: tcolorbox equal-height groups require a second pass to
        # resolve box heights computed in the first pass.
        for _pass_n in range(2):
            result = subprocess.run(
                [engine, "-interaction=nonstopmode", OUT_PATH.name],
                cwd=OUT_PATH.parent,
                capture_output=True,
                text=True,
                check=False,
            )
        if pdf_path.exists():
            compiled = True
            logger.info(f"PDF -> {pdf_path}")
            # Remove auxiliary files produced by LaTeX to keep the output dir clean.
            for ext in (".aux", ".log", ".out"):
                aux = OUT_PATH.with_suffix(ext)
                if aux.exists():
                    aux.unlink()
            break
        logger.warning(f"{engine} failed (rc={result.returncode}). Trying next")
        if result.stderr:
            err_lines = result.stderr.strip().splitlines()[-5:]
            for line in err_lines:
                logger.debug(line)

    if not compiled:
        logger.warning(
            "No LaTeX engine found or compilation failed. "
            "Compile manually: lualatex antipattern_excerpts.tex"
        )

    # Write a thin wrapper for inclusion in the paper via \includegraphics.
    includable_path = OUT_PATH.with_name("antipattern_excerpts_includable.tex")
    pdf_relpath = pdf_path.stem  # typically just "antipattern_excerpts"
    includable_tex = render_includable(pdf_relpath)
    includable_path.write_text(includable_tex)
    logger.info(f"Includable figure -> {includable_path}")

    # Log panel summary for quick inspection after generation.
    logger.info("Panel summary:")
    for p in panels:
        logger.info(f"  {CATEGORY_LABELS[p.category]}:")
        logger.info(f"    Source: {p.model} / {p.env} / {p.level} / trial {p.trial}")
        for nid, nt, ntype in zip(p.node_ids, p.node_texts, p.node_types, strict=False):
            logger.info(f"    {nid} ({ntype}): {_truncate(nt, 80)}")
        for m in p.messages:
            logger.info(
                f"    [{m.role}] (msg {m.msg_idx}): {_truncate(m.content_snippet, 100)}"
            )


if __name__ == "__main__":
    main()
