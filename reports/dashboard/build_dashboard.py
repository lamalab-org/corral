"""Build a standalone HTML dashboard of the benchmark reports.

Reads every ``<reports>/<environment>/trials.csv`` (AFM is left out by
default; ``<reports>`` is this repository's ``reports/`` unless ``--reports``
names another folder, such as one filled by ``fetch_trials.py``) and writes one self-contained page with the trial data embedded: leaderboard,
pass@k and pass^k curves, consistency, cost, per-level heatmap, native agent
vs ReAct, and passing vs failing trials. The page loads d3 from a CDN; nothing
else needs a network.

The page code is the same as the claude.ai dashboard's: ``page.html``,
``calcs.js`` and ``charts.js``, run through ``dash_shim.js``, which stands in
for that dashboard's runtime.

Run from the repository root:
    python reports/dashboard/build_dashboard.py
    python reports/dashboard/build_dashboard.py --out /tmp/dashboard.html
    python reports/dashboard/build_dashboard.py --exclude afm stargazer
    python reports/dashboard/fetch_trials.py --dest _data/reports
    python reports/dashboard/build_dashboard.py --reports _data/reports
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE.parent
D3_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"

PAIR_LABELS = {
    ("ai-scientist", "claude-opus-5-5"): "AI Scientist · Opus 5.5",
    ("claude-code", "claude-opus-5-5"): "Claude Code · Opus 5.5",
    ("codex", "gpt-5.6-sol"): "Codex · GPT-5.6 Sol",
    ("openhands", "kimi-k3"): "OpenHands · Kimi K3",
    ("react", "claude-opus-5-5"): "ReAct · Opus 5.5",
    ("react", "gpt-5.6-sol"): "ReAct · GPT-5.6 Sol",
    ("react", "kimi-k3"): "ReAct · Kimi K3",
}

# USD per 1M tokens: input, output, cache read. Same as cost_estimate.md.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "gpt-5.6-sol": (4.00, 20.00, 0.40),
    "kimi-k3": (2.60, 13.00, 0.29),
}

TRIAL_COLUMNS = [
    "environment",
    "level",
    "task_id",
    "trial",
    "agent",
    "model",
    "pair",
    "success",
    "errored",
    "duration_s",
    "input_tokens",
    "output_tokens",
    "tool_calls",
]


def read_trials(reports_dir: Path, exclude: set[str]) -> list[list]:
    paths = sorted(reports_dir.glob("*/trials.csv"))
    if not paths:
        sys.exit(
            f"No <environment>/trials.csv under {reports_dir}; run fetch_trials.py first"
        )
    rows = []
    for path in paths:
        if path.parent.name in exclude:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["agent"], r["model"])
                rows.append(
                    [
                        r["environment"],
                        r["level"],
                        r["task_id"],
                        int(r["trial"]),
                        r["agent"],
                        r["model"],
                        PAIR_LABELS.get(key, f"{r['agent']} · {r['model']}"),
                        1 if r["success"] == "True" else 0,
                        1 if r["error"].strip() else 0,
                        round(float(r["duration_s"] or 0), 1),
                        int(r["input_tokens"] or 0),
                        int(r["output_tokens"] or 0),
                        int(r["tool_calls"] or 0),
                    ]
                )
    return rows


def build(reports_dir: Path, exclude: set[str]) -> str:
    data = {
        "trials": {"columns": TRIAL_COLUMNS, "rows": read_trials(reports_dir, exclude)},
        "prices": {
            "columns": [
                "model",
                "input_usd_per_m",
                "output_usd_per_m",
                "cache_read_usd_per_m",
            ],
            "rows": [[model, *price] for model, price in PRICES.items()],
        },
    }
    params = {"env": "All", "metric": "pass_at_1", "level": "both"}
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )
    parts = {
        name: (HERE / name).read_text(encoding="utf-8")
        for name in ("theme.css", "page.html", "dash_shim.js", "calcs.js", "charts.js")
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CORRAL agent benchmark results</title>
<style>
{parts["theme.css"]}
</style>
<script src="{D3_URL}"></script>
</head>
<body>
<div id="dash-root">
{parts["page.html"]}
</div>
<script>
window.CORRAL_DATA = {payload};
window.CORRAL_PARAMS = {json.dumps(params)};
</script>
<script>
{parts["dash_shim.js"]}
</script>
<script>
{parts["calcs.js"]}
</script>
<script>
{parts["charts.js"]}
</script>
<script>dash.start();</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "corral_dashboard.html",
        help="output HTML file",
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=REPORTS_DIR,
        help="folder with <environment>/trials.csv (default: reports/)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["afm"],
        help="environments to leave out (default: afm)",
    )
    args = parser.parse_args()
    html = build(args.reports, set(args.exclude))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    sys.stdout.write(f"Wrote {args.out} ({len(html) / 1e6:.1f} MB)\n")


if __name__ == "__main__":
    main()
