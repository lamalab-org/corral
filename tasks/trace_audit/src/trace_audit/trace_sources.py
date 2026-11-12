"""Read the historical reports used to build the frozen audit pilots."""

from __future__ import annotations

import csv
import gzip
import json
import lzma
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "reports"
DATA = ROOT / "tasks/trace_audit/data"
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
    resistor = [f"git:{HISTORICAL_RESISTOR}:{path}" for path in tree
                if path.endswith("report.json.gz")]
    retro = []
    for path in sorted((REPORTS / "retrosynthesis").glob("level_*/*/report.json.*")):
        relative = path.relative_to(ROOT).as_posix()
        retro.append(f"git:{HISTORICAL_RETRO}:{relative}")
    return {"resistor_network": resistor, "retrosynthesis": retro}


def historical_rows(environment: str, sources: list[str]) -> list[dict]:
    rows = []
    for source in sources:
        report = read_report(source)
        level = int(re.search(r"level_(\d+)", source).group(1))
        for task_id, task in report["task_results"].items():
            runtime = report["metadata"]["agent"]["by_task"][task_id]
            agent = runtime["id"]
            model = runtime["runtime"]["model"].split("/")[-1]
            for trial_index, trial in enumerate(task["trials"]):
                rows.append(dict(environment=environment, level=level, agent=agent,
                                 model=model, task_id=task_id, trial=trial_index,
                                 score=float(trial.get("score") or 0),
                                 execution_id=trial["trial_id"], source=source))
    return rows


def current_rows(environment: str) -> list[dict]:
    with (REPORTS / environment / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [dict(environment=environment, level=int(row["level"]), agent=row["agent"],
                 model=row["model"], task_id=row["task_id"], trial=int(row["trial"]),
                 score=float(row["score"]), execution_id=row["execution_id"],
                 source=f"reports/{environment}/{row['report_file']}") for row in rows]


def key(row: dict) -> tuple:
    return (row["level"], row["agent"], row["model"], row["task_id"], row["trial"])


def flagged_keys() -> dict[str, set[tuple]]:
    from collections import defaultdict

    result = defaultdict(set)
    environment = None
    for line in (REPORTS / "flagged_trials.md").read_text().splitlines():
        if line.startswith("## "):
            environment = line[3:].strip()
        if environment not in ENVIRONMENTS or not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or "__" not in cells[1]:
            continue
        agent, model = cells[1].split("__", 1)
        result[environment].add((int(cells[0]), agent, model,
                                 cells[2].strip("`"), int(cells[3])))
    return result
