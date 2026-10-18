"""Find trials that failed for infrastructure or environment reasons.

Run from the repository root:
    python reports/find_infra_failures.py                    # every environment
    python reports/find_infra_failures.py afm corral_md      # some environments
    python reports/find_infra_failures.py --csv failures.csv # also write findings

Each trial is checked for these signals:

- its terminal error (`error_message` and the agent's `*-error` messages)
- `evaluation_error`, set when the scorer itself failed
- no activity at all: no messages and no LLM calls, so the environment failed
  before the agent ran
- tool results, matched only against specific phrases (provider error bodies,
  connection failures, out-of-memory), since tool output also holds data and code

Infrastructure categories: credits, rate_limit, provider_error, timeout,
policy, environment_crash, evaluation_error. Agent limits (iteration, turn or
interaction budgets, no submitted answer) and context-window overflows are
counted for context but are not infrastructure failures.

Findings from the terminal error, the evaluation error or a trial with no
activity have confidence `high`, as do tool results reporting exhausted
credits. Other findings from tool results have confidence `review`: the event
happened during the trial but may not have decided its outcome, so check the
evidence. A failed trial with a `high` infrastructure
finding is a rerun candidate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import lzma
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent

INFRA = (
    "credits",
    "rate_limit",
    "provider_error",
    "timeout",
    "policy",
    "environment_crash",
    "evaluation_error",
)
NOT_INFRA = ("agent_limit", "context_window")

# Checked in order; the first match wins. Credits come before rate limits
# because providers report an empty balance as HTTP 429 or 402.
TERMINAL_PATTERNS = [
    (
        "credits",
        r"no credits remaining|insufficient_quota|credit_balance_exhausted"
        r"|insufficient (credits|funds|balance)|exceeded your current quota"
        r"|payment required|requires more credits",
    ),
    ("policy", r"invalid_prompt|usage policy|content[_ ]policy|content_filter"),
    (
        "context_window",
        r"context (window|length)|maximum context|prompt is too long"
        r"|Input tokens exceed the configured limit",
    ),
    ("rate_limit", r"RateLimitError|Too Many Requests|rate.?limit"),
    (
        "provider_error",
        r"InternalServerError|ServiceUnavailable|APIConnectionError|Bad Gateway"
        r"|Gateway Time-?out|overloaded|stream disconnected|upstream connect error"
        r"|Error code: 5\d\d|Server disconnected|RemoteProtocolError"
        # The provider rejected a request the harness built.
        r"|400 Bad Request|Invalid 'messages\[",
    ),
    (
        "timeout",
        r"timed out|TimeoutError|ReadTimeout|ConnectTimeout|socket timeout"
        r"|Timeout on reading"
        r"|deadline exceeded|status='?timeout",
    ),
    (
        "environment_crash",
        r"Connection refused|Max retries exceeded|NewConnectionError|McpError"
        r"|MCP error|ClosedResourceError|BrokenResourceError|\bKilled\b|OOMKilled"
        r"|out of memory|MemoryError|No space left on device|sandbox|container",
    ),
    (
        "agent_limit",
        r"maximum iterations|interaction budget|LLM-call budget|error_max_turns"
        r"|without calling submit_answer|max_iterations",
    ),
]

# Tool results also contain data and code, so only match phrases that come from
# error bodies or failing infrastructure, never bare status codes.
TOOL_PATTERNS = [
    (
        "credits",
        r"no credits remaining|insufficient_quota|credit_balance_exhausted"
        r"|insufficient credits|exceeded your current quota",
    ),
    ("rate_limit", r"RateLimitError|Too Many Requests|Error code: 429"),
    (
        "provider_error",
        r"InternalServerError|ServiceUnavailable|APIConnectionError|Bad Gateway"
        r"|overloaded_error|Error code: 5\d\d|upstream connect error",
    ),
    (
        "timeout",
        r"ReadTimeout|ConnectTimeout|socket timeout|deadline exceeded"
        r"|Sandbox exceeded its \d+-second wall-clock timeout",
    ),
    (
        "environment_crash",
        r"Connection refused|NewConnectionError|McpError|MCP error"
        r"|ClosedResourceError|OOMKilled|No space left on device"
        r"|SandboxTerminatedError",
    ),
]

# Messages harnesses also write on passing trials, so they say nothing about
# why a trial failed.
BENIGN = re.compile(
    r"harness returned no answer|completed without a FinishAction", re.IGNORECASE
)

TERMINAL = [(c, re.compile(p, re.IGNORECASE)) for c, p in TERMINAL_PATTERNS]
TOOL = [(c, re.compile(p, re.IGNORECASE)) for c, p in TOOL_PATTERNS]

FIELDS = [
    "environment",
    "level",
    "agent",
    "model",
    "run_id",
    "task_id",
    "trial",
    "execution_id",
    "score",
    "success",
    "category",
    "infrastructure",
    "confidence",
    "source",
    "evidence",
    "langfuse_trace_url",
]


def environments() -> list[Path]:
    return sorted(p.parent for p in REPORTS_DIR.glob("*/trials.csv"))


def load_report(path: Path) -> dict:
    opener = lzma.open if path.suffix == ".xz" else gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def snippet(text: str, match: re.Match, width: int = 160) -> str:
    start = max(0, match.start() - width)
    return " ".join(text[start : match.end() + width].split())


def classify(text: str, patterns: list) -> tuple[str, str] | None:
    for category, pattern in patterns:
        match = pattern.search(text)
        if match:
            return category, snippet(text, match)
    return None


def terminal_errors(trial: dict) -> list[str]:
    errors = [str(trial["error_message"])] if trial.get("error_message") else []
    errors.extend(
        str(message["content"])
        for message in trial.get("messages") or []
        if "error" in (message.get("name") or "") and message.get("content")
    )
    return [e for e in errors if not BENIGN.search(e)]


def trial_findings(trial: dict) -> list[dict]:
    """Return one finding per category found in this trial."""
    findings: dict[str, dict] = {}

    def add(category: str, confidence: str, source: str, evidence: str) -> None:
        current = findings.get(category)
        if current is None or (
            confidence == "high" and current["confidence"] != "high"
        ):
            findings[category] = {
                "category": category,
                "infrastructure": category in INFRA,
                "confidence": confidence,
                "source": source,
                "evidence": evidence[:500],
            }

    usage = trial.get("token_usage") or {}
    if not trial.get("messages") and not usage.get("llm_calls"):
        add(
            "environment_crash",
            "high",
            "no_activity",
            str(trial.get("error_message") or "trial ended before the agent ran"),
        )
    if trial.get("evaluation_error"):
        add("evaluation_error", "high", "evaluation", str(trial["evaluation_error"]))
    errors = terminal_errors(trial)
    for error in errors:
        found = classify(error, TERMINAL)
        if found:
            add(found[0], "high", "error", found[1])
    if errors and not findings:
        # An error no pattern knows: report it so new failure modes are seen.
        add("unclassified_error", "review", "error", " ".join(errors[0].split()))
    for call in trial.get("tool_calls") or []:
        text = " ".join(str(call.get(key) or "") for key in ("error_message", "result"))
        found = classify(text, TOOL)
        if found:
            # A tool whose provider has no credits left is unusable for the
            # rest of the trial, whatever the agent does.
            confidence = "high" if found[0] == "credits" else "review"
            add(found[0], confidence, f"tool:{call.get('tool_name')}", found[1])
    return list(findings.values())


def scan(env_dir: Path) -> list[dict]:
    with (env_dir / "trials.csv").open(newline="", encoding="utf-8") as f:
        index = {row["execution_id"]: row for row in csv.DictReader(f)}
    rows = []
    for report in sorted(env_dir.glob("level_*/*/report.json.[gx]z")):
        data = load_report(report)
        for task in data["task_results"].values():
            for trial in task["trials"]:
                row = index[trial["trial_id"]]
                rows.extend(
                    {key: row.get(key, "") for key in FIELDS} | finding
                    for finding in trial_findings(trial)
                )
        del data
    return rows


def summarize(env: str, rows: list[dict], trials: int) -> None:
    failed = [r for r in rows if r["success"] != "True"]
    rerun = {
        r["execution_id"]
        for r in failed
        if r["infrastructure"] and r["confidence"] == "high"
    }
    sys.stdout.write(
        f"\n{env}: {trials} trials, {len(rerun)} failed with a high-confidence "
        "infrastructure cause (rerun candidates)\n"
    )
    counts = collections.Counter(
        (r["category"], r["confidence"], f"L{r['level']} {r['agent']}/{r['model']}")
        for r in failed
    )
    order = {c: i for i, c in enumerate((*INFRA, "unclassified_error", *NOT_INFRA))}
    for (category, confidence, run), n in sorted(
        counts.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2])
    ):
        sys.stdout.write(f"  {category:18} {confidence:6} {n:4}  {run}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "environments", nargs="*", help="environment folders (default: all)"
    )
    parser.add_argument("--csv", type=Path, help="write every finding to this CSV")
    args = parser.parse_args()

    env_dirs = environments()
    if args.environments:
        known = {p.name: p for p in env_dirs}
        unknown = sorted(set(args.environments) - set(known))
        if unknown:
            sys.stderr.write(f"unknown environments: {', '.join(unknown)}\n")
            return 1
        env_dirs = [known[name] for name in args.environments]

    sys.stdout.write(
        "Failed trials by category; `review` findings come from tool results\n"
    )
    all_rows = []
    for env_dir in env_dirs:
        rows = scan(env_dir)
        with (env_dir / "trials.csv").open(newline="", encoding="utf-8") as f:
            trials = sum(1 for _ in csv.DictReader(f))
        summarize(env_dir.name, rows, trials)
        all_rows.extend(rows)

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(all_rows)
        sys.stdout.write(f"\n{len(all_rows)} findings written to {args.csv}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
