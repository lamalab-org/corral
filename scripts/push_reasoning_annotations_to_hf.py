"""
Push annotated reasoning reports to HuggingFace Hub.

Collects *.annotated.json from reasoning_reports/{claude_sonnet_45,gpt_4o,gpt_oss_120b},
enriches each record with the node IDs involved in every detected pattern and
antipattern, and pushes the dataset to HF with one config per model.

Usage:

# Push all models
python scripts/push_reasoning_annotations_to_hf.py

# Dry-run: just print what would be pushed
python scripts/push_reasoning_annotations_to_hf.py --dry_run
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import fire
from dotenv import load_dotenv
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "reasoning_reports"))

from analyze import (  # noqa: E402
    ANTIPATTERN_NAMES,
    SUBGRAPH_NAMES,
    _has_edge,
    _has_informs_link,
    _neighbours,
    _nodes_of_type,
    build_index,
    node_time,
)

load_dotenv(dotenv_path=_REPO_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")

HF_REPO = "jablonkagroup/corral-reasoning-annotations"

MODEL_DIRS = ["claude_sonnet_45", "gpt_4o", "gpt_oss_120b"]

MODEL_DISPLAY: dict[str, str] = {
    "claude_sonnet_45": "claude-sonnet-4.5",
    "gpt_4o": "gpt-4o",
    "gpt_oss_120b": "gpt-oss-120b",
}

# reports_v2 uses slightly different directory names for some models.
V2_MODEL_DIRS: dict[str, str] = {
    "claude_sonnet_45": "claude_sonnet_45",
    "gpt_4o": "gpt-4o",
    "gpt_oss_120b": "gpt_oss_120b",
}


def _msg_hash(messages: list[dict]) -> str:
    """Hash non-system message contents for fingerprinting a trace."""
    h = hashlib.sha256()
    for m in messages:
        if m.get("role") == "system":
            continue
        c = m.get("content") or ""
        if isinstance(c, list):
            c = json.dumps(c, sort_keys=True)
        h.update(str(c).encode("utf-8", errors="replace"))
    return h.hexdigest()


def build_v2_index() -> dict[str, tuple[str | None, str | None]]:
    """Build a hash -> (task_id, timestamp) index from reports_v2/ traces."""
    index: dict[str, tuple[str | None, str | None]] = {}
    v2_root = _REPO_ROOT / "reports_v2"
    if not v2_root.exists():
        logger.warning("reports_v2/ not found - task_id/timestamp will be empty")
        return index
    for v2_model in V2_MODEL_DIRS.values():
        base = v2_root / v2_model
        if not base.exists():
            continue
        for fp in base.rglob("*.json"):
            try:
                with fp.open(encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            msgs = d.get("messages")
            if not msgs:
                continue
            h = _msg_hash(msgs)
            index[h] = (d.get("task_id"), d.get("timestamp"))
    logger.info(f"reports_v2 index: {len(index)} entries")
    return index


def _subgraph_nodes_local(
    nodes: list[dict], edges: list[dict]
) -> dict[str, list[list[str]]]:
    """Return, for each subgraph pattern, a list of node-ID groups (witnesses)."""
    node_by_id, out_edges, _in = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    results: dict[str, list[list[str]]] = {name: [] for name in SUBGRAPH_NAMES}

    # refutation_driven_belief_revision
    for h1 in _nodes_of_type("H", node_type_map):
        for t in _neighbours(h1, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    for h2 in _neighbours(
                        h1, "updates_to", "H", out_edges, node_type_map
                    ):
                        if h2 != h1:
                            results["refutation_driven_belief_revision"].append(
                                sorted({h1, t, ev, j, h2})
                            )

    # fixed_hypothesis_test_tuning
    for h in _nodes_of_type("H", node_type_map):
        for t1 in _neighbours(h, "tests", "T", out_edges, node_type_map):
            for ev in _neighbours(t1, "observes", "E", out_edges, node_type_map):
                for j in _nodes_of_type("J", node_type_map):
                    if not _has_informs_link(ev, j, out_edges):
                        continue
                    j_to_Ts = _neighbours(j, "tests", "T", out_edges, node_type_map)
                    if not j_to_Ts:
                        continue
                    h_to_Hs = _neighbours(
                        h, "updates_to", "H", out_edges, node_type_map
                    )
                    if h_to_Hs:
                        continue
                    for t2 in j_to_Ts:
                        results["fixed_hypothesis_test_tuning"].append(
                            sorted({h, t1, ev, j, t2})
                        )

    # explore_then_test_transition
    for t0 in _nodes_of_type("T", node_type_map):
        t0_time = node_time(node_by_id[t0])
        evs = _neighbours(t0, "observes", "E", out_edges, node_type_map)
        if not evs:
            continue
        for h1 in _nodes_of_type("H", node_type_map):
            if node_time(node_by_id[h1]) <= t0_time:
                continue
            h_tests = _neighbours(h1, "tests", "T", out_edges, node_type_map)
            if h_tests:
                results["explore_then_test_transition"].append(
                    sorted({t0, *evs, h1, *h_tests})
                )

    # hypothesis_reranking
    Hs = _nodes_of_type("H", node_type_map)
    for i, h1 in enumerate(Hs):
        for h2 in Hs[i + 1 :]:
            competes = _has_edge(h1, h2, "competes_with", out_edges) or _has_edge(
                h2, h1, "competes_with", out_edges
            )
            if not competes:
                continue
            if _neighbours(h1, "tests", "T", out_edges, node_type_map) and _neighbours(
                h2, "tests", "T", out_edges, node_type_map
            ):
                results["hypothesis_reranking"].append(sorted({h1, h2}))

    # evidence_led_hypothesis_generation
    for e0 in _nodes_of_type("E", node_type_map):
        e0_time = node_time(node_by_id[e0])
        for j0 in _nodes_of_type("J", node_type_map):
            if not _has_informs_link(e0, j0, out_edges):
                continue
            for h1 in _nodes_of_type("H", node_type_map):
                if node_time(node_by_id[h1]) <= e0_time:
                    continue
                h_tests = _neighbours(h1, "tests", "T", out_edges, node_type_map)
                if h_tests:
                    results["evidence_led_hypothesis_generation"].append(
                        sorted({e0, j0, h1, *h_tests})
                    )

    # convergent_multi_test_evidence
    for h in _nodes_of_type("H", node_type_map):
        test_targets = _neighbours(h, "tests", "T", out_edges, node_type_map)
        test_with_ev = [
            t
            for t in test_targets
            if _neighbours(t, "observes", "E", out_edges, node_type_map)
        ]
        if len(set(test_with_ev)) >= 3:
            involved = {h}
            for t in test_with_ev:
                involved.add(t)
                involved.update(
                    _neighbours(t, "observes", "E", out_edges, node_type_map)
                )
            results["convergent_multi_test_evidence"].append(sorted(involved))

    # evidence_guided_test_redesign
    for j in _nodes_of_type("J", node_type_map):
        for t in _neighbours(j, "tests", "T", out_edges, node_type_map):
            evs = _neighbours(t, "observes", "E", out_edges, node_type_map)
            if evs:
                results["evidence_guided_test_redesign"].append(sorted({j, t, *evs}))

    return results


def _antipattern_nodes_local(
    nodes: list[dict], edges: list[dict]
) -> dict[str, list[list[str]]]:
    """Return, for each antipattern, a list of node-ID groups (witnesses)."""
    node_by_id, out_edges, in_edges = build_index(nodes, edges)
    node_type_map = {nid: n.get("type") for nid, n in node_by_id.items()}
    results: dict[str, list[list[str]]] = {name: [] for name in ANTIPATTERN_NAMES}

    # untested_claim
    for h in _nodes_of_type("H", node_type_map):
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            results["untested_claim"].append([h])

    # evidence_non_uptake
    for ev in _nodes_of_type("E", node_type_map):
        used = any(
            e.get("relation") == "informs"
            and node_type_map.get(e.get("dst")) in {"J", "H"}
            for e in out_edges.get(ev, [])
        )
        if not used:
            used = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) in {"J", "H"}
                for e in in_edges.get(ev, [])
            )
        if not used:
            results["evidence_non_uptake"].append([ev])

    # unsupported_judgment
    for j in _nodes_of_type("J", node_type_map):
        has_e = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(j, [])
        )
        if not has_e:
            has_e = any(
                e.get("relation") == "informs"
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(j, [])
            )
        if not has_e:
            results["unsupported_judgment"].append([j])

    # stalled_revision
    for h in _nodes_of_type("H", node_type_map):
        is_revised = any(e.get("relation") == "updates_to" for e in in_edges.get(h, []))
        if not is_revised:
            continue
        has_test = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not has_test:
            results["stalled_revision"].append([h])

    # contradiction_without_repair
    for nid in list(node_type_map):
        for e in out_edges.get(nid, []):
            if e.get("relation") != "contradicts":
                continue
            dst = e.get("dst")
            if dst is None or node_type_map.get(dst) != "H":
                continue
            h_t = node_time(node_by_id.get(dst, {}))
            resolved = False
            for ie in in_edges.get(dst, []):
                src = ie.get("src")
                if (
                    ie.get("relation") == "updates_to"
                    and node_type_map.get(src) == "H"
                    and node_time(node_by_id.get(src, {})) >= h_t
                ):
                    resolved = True
                    break
            if not resolved:
                for oe in out_edges.get(dst, []):
                    if oe.get("relation") == "competes_with":
                        resolved = True
                        break
                if not resolved:
                    for ie in in_edges.get(dst, []):
                        if ie.get("relation") == "competes_with":
                            resolved = True
                            break
            if not resolved:
                results["contradiction_without_repair"].append(sorted({nid, dst}))

    # premature_commitment
    h_linked_to_c: dict[str, set[str]] = {}
    for j in _nodes_of_type("J", node_type_map):
        informs_c = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(j, [])
        )
        if not informs_c:
            continue
        c_nodes = [
            e["dst"]
            for e in out_edges.get(j, [])
            if e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
        ]
        for e in out_edges.get(j, []):
            if (
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) == "H"
            ):
                h_linked_to_c.setdefault(e["dst"], set()).add(j)
                h_linked_to_c[e["dst"]].update(c_nodes)
    for h, context_nodes in h_linked_to_c.items():
        tested = any(e.get("relation") == "tests" for e in out_edges.get(h, []))
        if not tested:
            tested = any(e.get("relation") == "tests" for e in in_edges.get(h, []))
        if not tested:
            results["premature_commitment"].append(sorted({h, *context_nodes}))

    # uninformative_test
    for t in _nodes_of_type("T", node_type_map):
        has_ev = any(
            e.get("relation") in ("observes", "tests")
            and node_type_map.get(e.get("dst")) == "E"
            for e in out_edges.get(t, [])
        )
        if not has_ev:
            has_ev = any(
                e.get("relation") in ("observes", "tests")
                and node_type_map.get(e.get("src")) == "E"
                for e in in_edges.get(t, [])
            )
        if not has_ev:
            results["uninformative_test"].append([t])

    # fixed_belief_trace  (trace-level; no specific nodes)
    has_update = any(
        e.get("relation") == "updates_to"
        for edges_list in out_edges.values()
        for e in edges_list
    )
    if not has_update:
        results["fixed_belief_trace"].append(["__trace__"])

    # disconnected_evidence
    for ev in _nodes_of_type("E", node_type_map):
        if not out_edges.get(ev) and not in_edges.get(ev):
            results["disconnected_evidence"].append([ev])

    # one_sided_confirmation
    h_linked_to_c2: dict[str, set[str]] = {}
    for j in _nodes_of_type("J", node_type_map):
        informs_c = any(
            e.get("relation") == "informs" and node_type_map.get(e.get("dst")) == "C"
            for e in out_edges.get(j, [])
        )
        if not informs_c:
            continue
        for e in out_edges.get(j, []):
            if (
                e.get("relation") == "informs"
                and node_type_map.get(e.get("dst")) == "H"
            ):
                h_linked_to_c2.setdefault(e["dst"], set()).add(j)
    for h, j_set in h_linked_to_c2.items():
        has_support = False
        for ie in in_edges.get(h, []):
            src = ie.get("src")
            if ie.get("relation") == "informs" and node_type_map.get(src) == "J":
                for je in in_edges.get(src, []):
                    if (
                        je.get("relation") == "informs"
                        and node_type_map.get(je.get("src")) == "E"
                    ):
                        has_support = True
                        break
            if ie.get("relation") == "informs" and node_type_map.get(src) == "E":
                has_support = True
            if has_support:
                break
        if not has_support:
            continue
        has_contradiction = any(
            ie.get("relation") == "contradicts" for ie in in_edges.get(h, [])
        )
        if not has_contradiction:
            has_contradiction = any(
                oe.get("relation") == "contradicts" for oe in out_edges.get(h, [])
            )
        if not has_contradiction:
            results["one_sided_confirmation"].append(sorted({h, *j_set}))

    # precommitted_test_plan
    Cs = _nodes_of_type("C", node_type_map)
    Es = _nodes_of_type("E", node_type_map)
    for h in _nodes_of_type("H", node_type_map):
        for t in _neighbours(h, "tests", "T", out_edges, node_type_map):
            t_time = node_time(node_by_id[t])
            matched = False
            for c in Cs:
                c_time = node_time(node_by_id[c])
                if c_time > t_time:
                    continue
                for ev in Es:
                    if node_time(node_by_id[ev]) > c_time:
                        results["precommitted_test_plan"].append(sorted({h, t, c, ev}))
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break

    return results


def _parse_path(fp: Path) -> dict[str, str]:
    """Extract model, env, level from a path like
    reasoning_reports/<model>/<env>/<level>/annotated/<file>.annotated.json
    """
    parts = fp.relative_to(_REPO_ROOT / "reasoning_reports").parts
    model = parts[0]
    env = parts[1]
    level = parts[2]
    return {"model": model, "env": env, "level": level}


def build_record(
    fp: Path,
    v2_index: dict[str, tuple[str | None, str | None]],
) -> dict[str, Any]:
    """Load an annotated JSON and build a dataset record with pattern node info."""
    with fp.open(encoding="utf-8") as f:
        data = json.load(f)

    meta = _parse_path(fp)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # Resolve task_id, timestamp, and messages from the original trace via reports_v2.
    input_rel = data.get("input_file")
    task_id: str | None = None
    timestamp: str | None = None
    messages: list[dict] = []
    if input_rel:
        input_fp = _REPO_ROOT / "reasoning_reports" / input_rel
        if input_fp.exists():
            with input_fp.open(encoding="utf-8") as f:
                inp = json.load(f)
            messages = inp.get("messages", [])
            h = _msg_hash(messages)
            task_id, timestamp = v2_index.get(h, (None, None))

    subgraph_nodes = _subgraph_nodes_local(nodes, edges)
    antipattern_nodes = _antipattern_nodes_local(nodes, edges)

    # Build count dicts (for convenience)
    subgraph_counts = {name: len(groups) for name, groups in subgraph_nodes.items()}
    antipattern_counts = {
        name: len(groups) for name, groups in antipattern_nodes.items()
    }

    return {
        "source_file": str(fp.relative_to(_REPO_ROOT)),
        "model": meta["model"],
        "env": meta["env"],
        "level": meta["level"],
        "task_id": task_id,
        "timestamp": timestamp,
        "input_file": input_rel,
        "provenance": json.dumps(data.get("provenance", {})),
        "messages": json.dumps(messages),
        "nodes": json.dumps(nodes),
        "edges": json.dumps(edges),
        "qc": json.dumps(data.get("qc", {})),
        "n_messages": len(messages),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "subgraph_counts": json.dumps(subgraph_counts),
        "subgraph_nodes": json.dumps(subgraph_nodes),
        "antipattern_counts": json.dumps(antipattern_counts),
        "antipattern_nodes": json.dumps(antipattern_nodes),
    }


def collect_annotated_files(model_dir: str) -> list[Path]:
    """Find all *.annotated.json in reasoning_reports/<model_dir>/...
    Only looks in 'annotated/' subdirs, skips 'old_annotated/'.
    """
    base = _REPO_ROOT / "reasoning_reports" / model_dir
    files = sorted(base.rglob("*.annotated.json"))
    # Keep only files in directories named exactly 'annotated'
    return [f for f in files if f.parent.name == "annotated"]


def main(dry_run: bool = False) -> None:
    from datasets import Dataset

    logger.info("Building reports_v2 index for task_id / timestamp lookup ...")
    v2_index = build_v2_index()

    for model_dir in MODEL_DIRS:
        files = collect_annotated_files(model_dir)
        if not files:
            logger.warning(f"No annotated files found for {model_dir}")
            continue

        logger.info(f"[{model_dir}] Found {len(files)} annotated files")

        records = []
        for fp in files:
            try:
                rec = build_record(fp, v2_index)
                records.append(rec)
            except Exception as exc:
                logger.warning(f"Skipping {fp}: {exc}")
                continue

        logger.info(f"[{model_dir}] Built {len(records)} records")

        if dry_run:
            logger.info(
                f"[DRY RUN] Would push {len(records)} records as config '{model_dir}'"
            )
            if records:
                logger.info(f"  Sample keys: {list(records[0].keys())}")
                logger.info(
                    f"  Sample task_id={records[0]['task_id']}, timestamp={records[0]['timestamp']}"
                )
                n_with_ts = sum(1 for r in records if r["timestamp"])
                logger.info(f"  Records with timestamp: {n_with_ts}/{len(records)}")
                sample_ap = json.loads(records[0]["antipattern_nodes"])
                non_empty = {k: v for k, v in sample_ap.items() if v}
                if non_empty:
                    logger.info(f"  Sample antipattern_nodes (non-empty): {non_empty}")
                sample_sg = json.loads(records[0]["subgraph_nodes"])
                non_empty_sg = {k: v for k, v in sample_sg.items() if v}
                if non_empty_sg:
                    logger.info(f"  Sample subgraph_nodes (non-empty): {non_empty_sg}")
            continue

        ds = Dataset.from_list(records)
        config_name = model_dir
        logger.info(f"[{model_dir}] Pushing config '{config_name}' ({len(ds)} rows)")
        ds.push_to_hub(
            HF_REPO,
            config_name=config_name,
            token=HF_TOKEN,
            private=False,
        )
        logger.info(f"[{model_dir}] Done")

    logger.info("All done.")


if __name__ == "__main__":
    fire.Fire(main)
