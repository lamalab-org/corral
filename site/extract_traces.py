"""Extract curated reasoning traces from HuggingFace and output traces.js for the landing page.

Pulls annotated epistemological graphs from jablonkagroup/corral-reasoning-annotations,
selects representative traces for pattern diversity, and outputs a JS constant for the
Epistemic Trace Explorer tab.
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
log = logging.getLogger(__name__)

HF_DATASET = "jablonkagroup/corral-reasoning-annotations"
CONFIGS = ["claude_sonnet_45", "gpt_4o", "gpt_oss_120b"]
SITE_DIR = Path(__file__).resolve().parent

MODEL_DISPLAY = {
    "claude_sonnet_45": "Claude 4.5 Sonnet",
    "gpt_4o": "GPT-4o",
    "gpt_oss_120b": "GPT-OSS-120B",
}

MAX_TRACES_PER_CONFIG = 10
MAX_PATTERN_INSTANCES = 5
MIN_NODES = 5
MAX_NODES = 80


def score_trace(row: dict) -> float:
    """Score a trace for curation based on pattern diversity and graph richness."""
    sc = _parse_json(row.get("subgraph_counts", "{}"))
    ac = _parse_json(row.get("antipattern_counts", "{}"))

    productive = sum(v for v in sc.values() if isinstance(v, int))
    breakdowns = sum(v for v in ac.values() if isinstance(v, int))
    n_productive_types = sum(1 for v in sc.values() if isinstance(v, int) and v > 0)
    n_breakdown_types = sum(1 for v in ac.values() if isinstance(v, int) and v > 0)

    return (
        n_productive_types * 10
        + n_breakdown_types * 8
        + min(productive, 50)
        + min(breakdowns, 30)
        + row.get("n_nodes", 0) * 0.1
    )


def _parse_json(val):
    if isinstance(val, str):
        return json.loads(val)
    return val


def build_trace_record(row: dict, idx: int, config: str) -> dict:
    """Build a trace record for the JS output."""
    nodes = _parse_json(row["nodes"])
    edges = _parse_json(row["edges"])
    subgraph_counts = _parse_json(row.get("subgraph_counts", "{}"))
    subgraph_nodes = _parse_json(row.get("subgraph_nodes", "{}"))
    antipattern_counts = _parse_json(row.get("antipattern_counts", "{}"))
    antipattern_nodes = _parse_json(row.get("antipattern_nodes", "{}"))

    for key in subgraph_nodes:
        if (
            isinstance(subgraph_nodes[key], list)
            and len(subgraph_nodes[key]) > MAX_PATTERN_INSTANCES
        ):
            subgraph_nodes[key] = subgraph_nodes[key][:MAX_PATTERN_INSTANCES]
    for key in antipattern_nodes:
        if (
            isinstance(antipattern_nodes[key], list)
            and len(antipattern_nodes[key]) > MAX_PATTERN_INSTANCES
        ):
            antipattern_nodes[key] = antipattern_nodes[key][:MAX_PATTERN_INSTANCES]

    return {
        "id": f"{config}_{row['task_id']}_{idx}",
        "model": MODEL_DISPLAY.get(row["model"], row["model"]),
        "env": row["env"],
        "level": row["level"],
        "task_id": row["task_id"],
        "n_nodes": row["n_nodes"],
        "n_edges": row["n_edges"],
        "nodes": nodes,
        "edges": edges,
        "subgraph_counts": subgraph_counts,
        "subgraph_nodes": subgraph_nodes,
        "antipattern_counts": antipattern_counts,
        "antipattern_nodes": antipattern_nodes,
    }


def curate_traces(ds, config: str) -> list[dict]:
    """Select representative traces guaranteeing environment coverage."""
    from collections import defaultdict

    env_level_traces = defaultdict(list)
    for i, row in enumerate(ds):
        if row["n_nodes"] < MIN_NODES or row["n_nodes"] > MAX_NODES:
            continue
        s = score_trace(row)
        env_level_traces[(row["env"], row["level"])].append((s, i, row))

    # First pass: 1 best trace per environment (not per env+level) for coverage
    selected_ids = set()
    selected = []
    env_traces = defaultdict(list)
    for (_env, _level), traces in sorted(env_level_traces.items()):
        env_traces[_env].extend(traces)

    for _env in sorted(env_traces):
        best = max(env_traces[_env], key=lambda x: x[0])
        selected.append(best)
        selected_ids.add(best[1])

    # Second pass: fill remaining slots by score from all (env, level) pairs
    remaining = [
        item
        for (_env, _level), traces in sorted(env_level_traces.items())
        for item in traces
        if item[1] not in selected_ids
    ]
    remaining.sort(key=lambda x: x[0], reverse=True)

    for item in remaining:
        if len(selected) >= MAX_TRACES_PER_CONFIG:
            break
        selected.append(item)

    records = []
    for _s, i, row in selected:
        records.append(build_trace_record(row, i, config))

    return records


def main():
    from datasets import load_dataset

    all_traces = []

    for config in CONFIGS:
        log.info(f"Loading {config}...")
        ds = load_dataset(HF_DATASET, config, split="train")
        log.info(f"  {len(ds)} traces available")

        records = curate_traces(ds, config)
        log.info(f"  Selected {len(records)} traces")

        for r in records:
            env_level = f"{r['env']}/{r['level']}"
            sc = r["subgraph_counts"]
            ac = r["antipattern_counts"]
            prod = sum(v for v in sc.values() if isinstance(v, int))
            brk = sum(v for v in ac.values() if isinstance(v, int))
            log.info(
                f"    {env_level:25s} nodes={r['n_nodes']:3d} "
                f"edges={r['n_edges']:3d} prod={prod:3d} break={brk:3d}"
            )

        all_traces.extend(records)

    all_traces.sort(key=lambda t: (t["model"], t["env"], t["level"]))

    json_str = json.dumps(all_traces, separators=(",", ":"), ensure_ascii=False)
    js_content = f"const TRACE_DATA = {json_str};\n"

    output_path = SITE_DIR / "traces.js"
    output_path.write_text(js_content)

    size_kb = output_path.stat().st_size / 1024
    log.info(f"\nWrote {output_path} ({size_kb:.1f} KB, {len(all_traces)} traces)")


if __name__ == "__main__":
    main()
