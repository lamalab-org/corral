"""Generate site/blog_traces.js for the Explainer deep-links used by the blog post.

docs/blog_18_05/index.md references six specific reasoning traces. Three of them
are present in the frozen HF annotations dataset (jablonkagroup/corral-reasoning-
annotations) and are pulled directly. The other three were never annotated (they
were sampled from the bulk corral-traces set but not selected for annotation), so
they are annotated locally with the exact pipeline and params the paper used
(analyze.process_file_async, anthropic/claude-sonnet-4-6, window 20 / overlap 5).

The HF dataset is left untouched. Locally generated annotations are cached under
site/blog_annotations/ so re-running without --force is free and deterministic.

Output: site/blog_traces.js  ->  const BLOG_TRACE_DATA = [ ...6 records... ];

Usage:
  python site/extract_blog_traces.py            # reuse cached annotations
  python site/extract_blog_traces.py --force    # re-annotate the 3 local traces
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

SITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SITE_DIR.parent
TRACES_DIR = REPO_ROOT / "reasoning_reports" / "blog_to_annotate"
# analyze.py --files writes <input_dir>/annotated/<stem>.annotated.json; read from there.
CACHE_DIR = TRACES_DIR / "annotated"

sys.path.insert(0, str(REPO_ROOT / "reasoning_reports"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze import process_file_async  # noqa: E402
from push_reasoning_annotations_to_hf import (  # noqa: E402
    _antipattern_nodes_local,
    _subgraph_nodes_local,
)

# Secrets live outside the repo, at the path analyze.py targets via
# load_dotenv("../../.env") (i.e. the repo's parent directory). Load it by
# absolute path so litellm can authenticate regardless of CWD. The repo-local
# .env is a non-overriding fallback (HF_TOKEN, etc.).
load_dotenv(REPO_ROOT.parent / ".env", override=True)
load_dotenv(REPO_ROOT / ".env", override=False)

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
log = logging.getLogger(__name__)

MODEL_DISPLAY = {
    "claude_sonnet_45": "Claude 4.5 Sonnet",
    "gpt_4o": "GPT-4o",
    "gpt_oss_120b": "GPT-OSS-120B",
}

HF_DATASET = "jablonkagroup/corral-reasoning-annotations"
ANNOTATE_MODEL = "anthropic/claude-sonnet-4-6"
WINDOW, OVERLAP, MAX_NODES = 20, 5, 100
MAX_PATTERN_INSTANCES = 5

# The six blog traces. `source` is "hf" (pull the exact annotated run from HF) or
# "annotate" (annotate the raw trace locally). `id` is the stable deep-link slug
# referenced from docs/blog_18_05/index.md.
BLOG_TRACES = [
    {
        "id": "blog_formula_fixation",
        "task_id": "22_22222_orgsyn_222_2222",
        "model_dir": "claude_sonnet_45",
        "env": "spectra",
        "level": "level_1",
        "timestamp": "20251217_194546",
        "raw": "22_22222_orgsyn_222_2222_20251217_194546.json",
        "source": "annotate",
    },
    {
        "id": "blog_resistor_sampling",
        "task_id": "task_3",
        "model_dir": "claude_sonnet_45",
        "env": "resistor",
        "level": "level_1",
        "timestamp": "20251111_112807",
        "raw": "task_3_20251111_112807.json",
        "source": "annotate",
    },
    {
        "id": "blog_spectra_sampling",
        "task_id": "10_15227_orgsyn_102_0001",
        "model_dir": "claude_sonnet_45",
        "env": "spectra",
        "level": "level_2",
        "timestamp": "20251218_212824",
        "raw": "10_15227_orgsyn_102_0001_20251218_212824.json",
        "source": "annotate",
    },
    {
        "id": "blog_wetlab_contradiction",
        "task_id": "qualysis_lvl1_02",
        "model_dir": "claude_sonnet_45",
        "env": "wetlab",
        "level": "level_1",
        "timestamp": "20251204_212358",
        "raw": "qualysis_lvl1_02_20251204_212358.json",
        "source": "hf",
    },
    {
        "id": "blog_afm_scaling",
        "task_id": "afm_experiment_level_4",
        "model_dir": "gpt_4o",
        "env": "afm",
        "level": "level_4",
        "timestamp": "20251111_143615",
        "raw": "afm_experiment_level_4_20251111_143615.json",
        "source": "hf",
    },
    {
        "id": "blog_md_rounding",
        "task_id": "aluminum_surface_energy_2",
        "model_dir": "claude_sonnet_45",
        "env": "md",
        "level": "level_2",
        "timestamp": "20260213_042422",
        "raw": "aluminum_surface_energy_2_20260213_042422.json",
        "source": "hf",
    },
]


def _truncate_groups(groups_by_pattern: dict) -> dict:
    out = {}
    for key, groups in groups_by_pattern.items():
        if isinstance(groups, list) and len(groups) > MAX_PATTERN_INSTANCES:
            out[key] = groups[:MAX_PATTERN_INSTANCES]
        else:
            out[key] = groups
    return out


def _record(spec: dict, nodes: list, edges: list) -> dict:
    subgraph_nodes = _subgraph_nodes_local(nodes, edges)
    antipattern_nodes = _antipattern_nodes_local(nodes, edges)
    subgraph_counts = {k: len(v) for k, v in subgraph_nodes.items()}
    antipattern_counts = {k: len(v) for k, v in antipattern_nodes.items()}
    return {
        "id": spec["id"],
        "model": MODEL_DISPLAY.get(spec["model_dir"], spec["model_dir"]),
        "env": spec["env"],
        "level": spec["level"],
        "task_id": spec["task_id"],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "subgraph_counts": subgraph_counts,
        "subgraph_nodes": _truncate_groups(subgraph_nodes),
        "antipattern_counts": antipattern_counts,
        "antipattern_nodes": _truncate_groups(antipattern_nodes),
    }


def _from_hf(spec: dict, hf_rows: list) -> dict:
    matches = [
        r
        for r in hf_rows
        if r["task_id"] == spec["task_id"]
        and r["model"] == spec["model_dir"]
        and str(r["timestamp"]) == spec["timestamp"]
    ]
    if not matches:
        raise SystemExit(
            f"[{spec['id']}] expected exact HF run not found "
            f"(task={spec['task_id']} model={spec['model_dir']} ts={spec['timestamp']})"
        )
    row = matches[0]
    nodes = json.loads(row["nodes"]) if isinstance(row["nodes"], str) else row["nodes"]
    edges = json.loads(row["edges"]) if isinstance(row["edges"], str) else row["edges"]
    log.info(f"[{spec['id']}] from HF: {len(nodes)} nodes, {len(edges)} edges")
    return _record(spec, nodes, edges)


async def _annotate(spec: dict, force: bool) -> dict:
    raw_path = TRACES_DIR / spec["raw"]
    if not raw_path.exists():
        raise SystemExit(f"[{spec['id']}] raw trace missing: {raw_path}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{raw_path.stem}.annotated.json"

    if cached.exists() and not force:
        doc = json.loads(cached.read_text())
        log.info(f"[{spec['id']}] cached annotation: {cached.name}")
    else:
        log.info(f"[{spec['id']}] annotating {raw_path.name} with {ANNOTATE_MODEL} ...")
        doc = await process_file_async(
            in_path=raw_path,
            out_dir=CACHE_DIR,
            model=ANNOTATE_MODEL,
            window=WINDOW,
            overlap=OVERLAP,
            max_nodes_per_window=MAX_NODES,
            strict_support=False,
            dry_run=False,
        )
    nodes, edges = doc["nodes"], doc["edges"]
    log.info(f"[{spec['id']}] annotated: {len(nodes)} nodes, {len(edges)} edges")
    return _record(spec, nodes, edges)


async def _build(force: bool) -> tuple[list[dict], list[str]]:
    needs_hf = any(s["source"] == "hf" for s in BLOG_TRACES)
    hf_rows = []
    if needs_hf:
        from datasets import load_dataset

        log.info(f"Loading {HF_DATASET} ...")
        hf_rows = list(load_dataset(HF_DATASET, split="train"))

    records: list[dict] = []
    missing: list[str] = []
    for spec in BLOG_TRACES:
        if spec["source"] == "hf":
            records.append(_from_hf(spec, hf_rows))
            continue
        try:
            records.append(await _annotate(spec, force))
        except Exception as exc:  # - report and keep going
            missing.append(spec["id"])
            log.warning(
                f"[{spec['id']}] annotation unavailable ({type(exc).__name__}: {exc}). "
                f"Skipping; provide ANTHROPIC_API_KEY or a cached annotation and re-run."
            )
    return records, missing


def main(force: bool = False, env_file: str | None = None) -> None:
    if env_file:
        loaded = load_dotenv(env_file, override=True)
        log.info(f"Loaded env file {env_file}: {loaded}")
    records, missing = asyncio.run(_build(force))
    json_str = json.dumps(records, separators=(",", ":"), ensure_ascii=False)
    out_path = SITE_DIR / "blog_traces.js"
    out_path.write_text(f"const BLOG_TRACE_DATA = {json_str};\n")
    size_kb = out_path.stat().st_size / 1024
    log.info(f"\nWrote {out_path} ({size_kb:.1f} KB, {len(records)} traces)")
    for r in records:
        prod = sum(v for v in r["subgraph_counts"].values())
        brk = sum(v for v in r["antipattern_counts"].values())
        log.info(
            f"  {r['id']:28s} {r['env']}/{r['level']:8s} "
            f"nodes={r['n_nodes']:3d} edges={r['n_edges']:3d} prod={prod} break={brk}"
        )
    if missing:
        log.warning(
            f"\n{len(missing)} trace(s) still need annotation: {', '.join(missing)}"
        )


if __name__ == "__main__":
    fire.Fire(main)
