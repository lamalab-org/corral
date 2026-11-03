"""Tabulate the scores of every environment, agent, model and level.

Run from the repository root:
    python reports/scores.py                      # every environment
    python reports/scores.py afm corral_md        # some environments
    python reports/scores.py --markdown out.md    # tables for a PR comment

Reads only `reports/<environment>/runs.csv`, so it needs no Git LFS, network
access or packages beyond the standard library. The overview shows pass@k
(share of tasks solved in at least one of k trials) and pass^k (share of tasks
solved in all k trials) per level, with k the trials per task (5); the
per-environment tables add passes, average score, pass@1, error trials and
token usage.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent
# The agent and model pairs every environment runs, in display order.
PAIRS = (
    ("ai-scientist", "claude-opus-5-5"),
    ("claude-code", "claude-opus-5-5"),
    ("codex", "gpt-5.6-sol"),
    ("openhands", "kimi-k3"),
    ("react", "claude-opus-5-5"),
    ("react", "gpt-5.6-sol"),
    ("react", "kimi-k3"),
)
MISSING = "—"


def environments() -> list[Path]:
    return sorted(p.parent for p in REPORTS_DIR.glob("*/runs.csv"))


def read_runs(env_dir: Path) -> list[dict]:
    with (env_dir / "runs.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pairs(runs: list[dict]) -> list[tuple[str, str]]:
    """The standard pairs, then any other pair found in the runs."""
    found = {(r["agent"], r["model"]) for r in runs}
    return [*PAIRS, *sorted(found - set(PAIRS))]


def levels(runs: list[dict]) -> list[str]:
    return sorted({r["level"] for r in runs}, key=int)


def score(value: str) -> str:
    return f"{float(value):.2f}" if value else MISSING


def tokens(value: str) -> str:
    if not value:
        return MISSING
    n = int(value)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def trials_per_task(results: list[tuple[str, list[dict]]]) -> str:
    """k for pass@k and pass^k: the trials per task, or `k` if runs differ."""
    ks = {run["trials_per_task"] for _, runs in results for run in runs}
    return ks.pop() if len(ks) == 1 else "k"


def overview(results: list[tuple[str, list[dict]]], column: str) -> list[str]:
    """One metric per agent and model (rows) and environment (columns)."""
    all_runs = [run for _, runs in results for run in runs]
    header = ["Agent / model"]
    for env, runs in results:
        header.append(f"{env} (L{'/L'.join(levels(runs))})")
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|" + "---:|" * len(results),
    ]
    for agent, model in pairs(all_runs):
        cells = [f"{agent} / {model}"]
        for _, runs in results:
            by_level = {
                r["level"]: r
                for r in runs
                if (r["agent"], r["model"]) == (agent, model)
            }
            cells.append(
                " / ".join(
                    score(by_level[level][column]) if level in by_level else MISSING
                    for level in levels(runs)
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def details(env: str, runs: list[dict], k: str) -> list[str]:
    lines = [
        f"<details><summary>{env}: {len(runs)} runs</summary>",
        "",
        f"| Level | Agent | Model | Passes | Average score | Pass@1 | Pass@{k} | "
        f"Pass^{k} | Error trials | Input tokens | Output tokens |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {pair: i for i, pair in enumerate(pairs(runs))}
    lines.extend(
        f"| {r['level']} | {r['agent']} | {r['model']} "
        f"| {r['passes']}/{r['trials']} | {score(r['average_score'])} "
        f"| {score(r['pass_at_1'])} | {score(r['pass_at_k'])} "
        f"| {score(r['pass_hat_k'])} | {r['error_trials']} "
        f"| {tokens(r['input_tokens'])} | {tokens(r['output_tokens'])} |"
        for r in sorted(
            runs, key=lambda r: (int(r["level"]), order[(r["agent"], r["model"])])
        )
    )
    return [*lines, "", "</details>"]


def markdown(results: list[tuple[str, list[dict]]]) -> str:
    runs = sum(len(r) for _, r in results)
    trials = sum(int(run["trials"]) for _, r in results for run in r)
    k = trials_per_task(results)
    lines = [
        f"{len(results)} environments, {runs} runs, {trials:,} trials. "
        f"Each cell is L1 / L2, or the levels in the column header; {MISSING} "
        "means no run.",
        "",
        f"**Pass@{k}**",
        "",
        *overview(results, "pass_at_k"),
        "",
        f"**Pass^{k}**",
        "",
        *overview(results, "pass_hat_k"),
    ]
    for env, env_runs in results:
        lines += ["", *details(env, env_runs, k)]
    lines += [
        "",
        f"Pass@{k}: share of tasks solved in at least one of {k} trials. "
        f"Pass^{k}: share of tasks solved in all {k}. Pass@1: mean success over "
        "all trials. Error trials: trials with a saved terminal error, including "
        "iteration limits.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "environments", nargs="*", help="environment folders (default: all)"
    )
    parser.add_argument(
        "--markdown", type=Path, help="write the tables to this Markdown file"
    )
    args = parser.parse_args()

    env_dirs = environments()
    if args.environments:
        known = {p.name: p for p in env_dirs}
        unknown = sorted(set(args.environments) - set(known))
        if unknown:
            sys.stderr.write(f"unknown environments: {', '.join(unknown)}\n")
            return 1
        env_dirs = [known[name] for name in args.environments]

    results = [(d.name, read_runs(d)) for d in env_dirs]
    text = markdown(results)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
        sys.stdout.write(f"Markdown written to {args.markdown}\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
