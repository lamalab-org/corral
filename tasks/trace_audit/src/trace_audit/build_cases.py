"""Freeze the 135-case audit sample from reports and historical Git objects."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import lzma
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "reports"
DATA = ROOT / "tasks/trace_audit/data"
SEED = 20261109
ENVIRONMENTS = (
    "afm", "corral_md", "inference_opt", "psychometrics",
    "resistor_network", "retrosynthesis", "spectra_elucidation",
    "stargazer", "wetlab",
)
HISTORICAL_RESISTOR = "fca3048c4^"
HISTORICAL_RETRO = "eca3dc717"


def read_report(source: str) -> dict:
    if source.startswith("git:"):
        ref, path = source[4:].split(":", 1)
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)
    else:
        raw = (ROOT / source).read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"Git LFS report missing: {source}")
    return json.loads(lzma.decompress(raw) if source.endswith(".xz") else gzip.decompress(raw))


def old_sources() -> dict[str, list[str]]:
    tree = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", HISTORICAL_RESISTOR,
         "tasks/resistor_network/reports/v4_terminal"], cwd=ROOT, text=True
    ).splitlines()
    resistor = [f"git:{HISTORICAL_RESISTOR}:{p}" for p in tree if p.endswith("report.json.gz")]
    retro = []
    for path in sorted((REPORTS / "retrosynthesis").glob("level_*/*/report.json.*")):
        relative = path.relative_to(ROOT).as_posix()
        retro.append(f"git:{HISTORICAL_RETRO}:{relative}")
    return {"resistor_network": resistor, "retrosynthesis": retro}


def historical_rows(env: str, sources: list[str]) -> list[dict]:
    rows = []
    for source in sources:
        report = read_report(source)
        level = int(re.search(r"level_(\d+)", source).group(1))
        for task_id, task in report["task_results"].items():
            runtime = report["metadata"]["agent"]["by_task"][task_id]
            agent = runtime["id"]
            model = runtime["runtime"]["model"].split("/")[-1]
            for trial_index, trial in enumerate(task["trials"]):
                rows.append(dict(environment=env, level=level, agent=agent,
                                 model=model, task_id=task_id, trial=trial_index,
                                 score=float(trial.get("score") or 0),
                                 execution_id=trial["trial_id"], source=source))
    return rows


def current_rows(env: str) -> list[dict]:
    with (REPORTS / env / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [dict(environment=env, level=int(r["level"]), agent=r["agent"],
                 model=r["model"], task_id=r["task_id"], trial=int(r["trial"]),
                 score=float(r["score"]), execution_id=r["execution_id"],
                 source=f"reports/{env}/{r['report_file']}") for r in rows]


def key(row: dict) -> tuple:
    return (row["level"], row["agent"], row["model"], row["task_id"], row["trial"])


def flagged_keys() -> dict[str, set[tuple]]:
    result = defaultdict(set)
    environment = None
    for line in (REPORTS / "flagged_trials.md").read_text().splitlines():
        if line.startswith("## "):
            environment = line[3:].strip()
        if environment not in ENVIRONMENTS or not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or "__" not in cells[1]:
            continue
        agent, model = cells[1].split("__", 1)
        result[environment].add((int(cells[0]), agent, model,
                                 cells[2].strip("`"), int(cells[3])))
    return result


def select_rows() -> list[tuple[dict, str]]:
    rng = random.Random(SEED)
    flags = flagged_keys()
    historical = old_sources()
    selected = []
    for env in ENVIRONMENTS:
        audit_pool = historical_rows(env, historical[env]) if env in historical else current_rows(env)
        flagged = [r for r in audit_pool if key(r) in flags[env]]
        rng.shuffle(flagged)
        count = 2 if env == "stargazer" else 4
        if len(flagged) < count:
            raise RuntimeError(f"Only {len(flagged)} recoverable flags for {env}; need {count}")
        chosen_flags = []
        for candidate in flagged:
            if candidate["agent"] not in {r["agent"] for r in chosen_flags}:
                chosen_flags.append(candidate)
            if len(chosen_flags) == count:
                break
        chosen_flags += [r for r in flagged if r not in chosen_flags][:count - len(chosen_flags)]
        remaining = [r for r in audit_pool if key(r) not in flags[env]]
        controls = []
        for target in chosen_flags:
            available = [r for r in remaining if r["level"] == target["level"]
                         and r["agent"] == target["agent"]
                         and (r["score"] > 0) == (target["score"] > 0)]
            if not available:
                available = [r for r in remaining if r["level"] == target["level"]
                             and (r["score"] > 0) == (target["score"] > 0)]
            if not available:
                raise RuntimeError(f"No control for {env}: {key(target)}")
            control = rng.choice(available)
            controls.append(control)
            remaining.remove(control)
        current = current_rows(env)
        excluded = {r["execution_id"] for r in chosen_flags + controls}
        random_pool = [r for r in current if key(r) not in flags[env]
                       and r["execution_id"] not in excluded
                       and r["agent"] in {"codex", "claude-code", "openhands", "ai-scientist"}]
        target_random = 15 - 2 * count
        random_cases = []
        agents = ("codex", "claude-code", "openhands", "ai-scientist")
        for agent in agents:
            random_cases.append(rng.choice([r for r in random_pool if r["agent"] == agent]))
        while len(random_cases) < target_random:
            totals = {agent: sum(r["agent"] == agent for r, _ in selected)
                      + sum(r["agent"] == agent for r in chosen_flags + controls + random_cases)
                      for agent in agents}
            agent = min(agents, key=lambda a: (totals[a], a))
            options = [r for r in random_pool if r["agent"] == agent
                       and r not in random_cases]
            random_cases.append(rng.choice(options))
        selected += [(r, "flagged") for r in chosen_flags]
        selected += [(r, "matched") for r in controls]
        selected += [(r, "random") for r in random_cases]
    rng.shuffle(selected)
    return selected


def build() -> None:
    cases_dir = DATA / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows()
    cache = {}
    private = []
    public = {env: [] for env in ENVIRONMENTS}
    for number, (row, stratum) in enumerate(rows, 1):
        source = row["source"]
        if source not in cache:
            cache[source] = read_report(source)
        trial = cache[source]["task_results"][row["task_id"]]["trials"][row["trial"]]
        if trial["trial_id"] != row["execution_id"]:
            raise RuntimeError(f"Trial position mismatch: {source}: {key(row)}")
        case_id = f"case-{number:03d}"
        case = {
            "trial_id": case_id,
            "environment": row["environment"],
            "level": row["level"],
            "task_id": row["task_id"],
            "source_agent": row["agent"],
            "source_model": row["model"],
            "messages": trial.get("messages", []),
            "output": trial.get("output"),
            "recorded_score": row["score"],
            "error_message": trial.get("error_message"),
            "evaluation_error": trial.get("evaluation_error"),
            "evaluation": trial.get("evaluation"),
        }
        payload = json.dumps(case, ensure_ascii=False, separators=(",", ":")).encode()
        (cases_dir / f"{case_id}.json.gz").write_bytes(
            gzip.compress(payload, compresslevel=6, mtime=0)
        )
        private.append({**row, "case_id": case_id, "stratum": stratum,
                        "case_sha256": hashlib.sha256(payload).hexdigest()})
        public[row["environment"]].append(case_id)
    (DATA / "manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    (DATA / "private_manifest.json").write_text(json.dumps(private, indent=2) + "\n")
    print(f"Built {len(rows)} cases: {dict((e, len(v)) for e, v in public.items())}")


if __name__ == "__main__":
    build()
