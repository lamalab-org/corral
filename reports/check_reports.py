"""Check that reports/ follows the report convention and is published on Hugging Face.

Run from the repository root:
    python reports/check_reports.py           # convention checks only
    python reports/check_reports.py --hf      # also check the Hugging Face datasets

Convention checks (see reports/README.md) fail on what matters for the data:
where reports live and how run folders are named, CSV columns, consistency
between runs.csv, trials.csv and the run folders, Langfuse IDs and URLs, and
that both CSVs match what build_index.py produces from the reports. Other files
(READMEs, notes, figures) are only reported as warnings. Reports stored as Git
LFS pointers cannot be read and are skipped by the rebuild check; --hf covers
them through report_sha256.

The --hf checks need huggingface_hub and pyarrow, and HF_TOKEN for the private
datasets. They require a subset for every environment and level, row counts
that match the CSVs, the same report_sha256 per run, and a trials table that
matches trials.csv trial by trial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORTS_DIR))

import build_index  # noqa: E402

FULL_REPO = "jablonkagroup/corral-reports-october-runs-v1"
SUMMARY_REPO = "jablonkagroup/corral-reports-october-runs-summary-v1"

AGENTS = sorted(set(build_index.AGENTS.values()))
MODELS = sorted(slug for slug, _ in build_index.MODELS.values())
ENV_FILES = {"runs.csv", "trials.csv"}
REPORT_FILES = {"report.json.gz", "report.json.xz"}
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


class Problems(list):
    def add(self, where: str, message: str) -> None:
        self.append(f"{where}: {message}")


def environments() -> list[Path]:
    return sorted(
        p
        for p in REPORTS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(len(LFS_POINTER)) == LFS_POINTER


def as_csv(value: object) -> str:
    return "" if value is None else str(value)


def check_layout(env_dir: Path, problems: Problems, warnings: Problems) -> list[Path]:
    """Check where reports live and how run folders are named.

    Only the reports and the two index files are required. Anything else, such
    as a README, notes or figures, is reported as a warning, not an error.
    """
    where = f"reports/{env_dir.name}"
    names = {p.name for p in env_dir.iterdir()}
    for missing in sorted(ENV_FILES - names):
        problems.add(where, f"missing {missing}")
    for extra in sorted(names - ENV_FILES - {"README.md"}):
        if not re.fullmatch(r"level_\d+", extra) or not (env_dir / extra).is_dir():
            warnings.add(where, f"'{extra}' is not part of the convention")

    reports = []
    for level_dir in sorted(env_dir.glob("level_*")):
        if not level_dir.is_dir():
            continue
        for run_dir in sorted(level_dir.iterdir()):
            run_where = f"{where}/{level_dir.name}/{run_dir.name}"
            if not run_dir.is_dir():
                warnings.add(run_where, "is not part of the convention")
                continue
            agent, _, model = run_dir.name.partition("__")
            if agent not in AGENTS or model not in MODELS:
                problems.add(
                    run_where,
                    f"run folders must be <agent>__<model> with agent in {AGENTS} "
                    f"and model in {MODELS}",
                )
                continue
            files = {p.name for p in run_dir.iterdir()}
            for extra in sorted(files - REPORT_FILES - {"run-metadata.json"}):
                warnings.add(f"{run_where}/{extra}", "is not part of the convention")
            found = sorted(files & REPORT_FILES)
            if len(found) != 1:
                problems.add(
                    run_where, "needs exactly one report.json.gz or report.json.xz"
                )
                continue
            reports.append(run_dir / found[0])
    return reports


def check_indexes(env_dir: Path, reports: list[Path], problems: Problems) -> None:
    """Check runs.csv and trials.csv against each other and the reports."""
    where = f"reports/{env_dir.name}"
    run_fields, runs = read_csv(env_dir / "runs.csv")
    trial_fields, trials = read_csv(env_dir / "trials.csv")
    if run_fields != build_index.RUN_FIELDS:
        problems.add(f"{where}/runs.csv", f"columns must be {build_index.RUN_FIELDS}")
        return
    if trial_fields != build_index.TRIAL_FIELDS:
        problems.add(
            f"{where}/trials.csv", f"columns must be {build_index.TRIAL_FIELDS}"
        )
        return

    report_files = {r.relative_to(env_dir).as_posix() for r in reports}
    run_files = [r["report_file"] for r in runs]
    if sorted(run_files) != sorted(report_files):
        problems.add(
            f"{where}/runs.csv",
            f"rows do not match the run folders (run build_index.py): "
            f"missing {sorted(report_files - set(run_files))}, "
            f"stale {sorted(set(run_files) - report_files)}",
        )

    trials_by_report: dict[str, list[dict[str, str]]] = {}
    seen: set[str] = set()
    for row in trials:
        execution_id = row["execution_id"]
        row_where = f"{where}/trials.csv {execution_id}"
        if execution_id in seen:
            problems.add(row_where, "duplicate execution_id")
        seen.add(execution_id)
        session = execution_id.rsplit(":", 2)[0]
        trace = hashlib.sha256(f"corral:{execution_id}".encode()).hexdigest()[:32]
        if row["langfuse_session_id"] != session:
            problems.add(
                row_where, "langfuse_session_id must be the run part of execution_id"
            )
        if row["langfuse_trace_id"] != trace:
            problems.add(
                row_where,
                'langfuse_trace_id must be sha256("corral:" + execution_id)[:32]',
            )
        if row["langfuse_trace_url"] != build_index.trace_url(trace):
            problems.add(row_where, "langfuse_trace_url does not match the trace ID")
        trials_by_report.setdefault(row["report_file"], []).append(row)

    for run in runs:
        run_where = f"{where}/runs.csv {run['run_id']}"
        expected = int(run["tasks"]) * int(run["trials_per_task"])
        count = len(trials_by_report.get(run["report_file"], []))
        if not int(run["trials"]) == count == expected:
            problems.add(
                run_where,
                f"{run['tasks']} tasks x {run['trials_per_task']} trials needs {expected} "
                f"trials; runs.csv says {run['trials']}, trials.csv has {count}",
            )
        if run["langfuse_session_url"] != build_index.session_url(run["run_id"]):
            problems.add(run_where, "langfuse_session_url does not match run_id")

    # The CSVs must be exactly what build_index.py produces from the reports.
    runs_by_file = {r["report_file"]: r for r in runs}
    for report in reports:
        rel = report.relative_to(env_dir).as_posix()
        if rel not in runs_by_file or is_lfs_pointer(report):
            continue
        try:
            run, run_trials = build_index.index_run(env_dir, report, env_dir.name)
        except Exception as error:
            problems.add(f"{where}/{rel}", f"cannot be indexed: {error}")
            continue
        if {k: as_csv(v) for k, v in run.items()} != runs_by_file[rel]:
            problems.add(
                f"{where}/runs.csv",
                f"row for {rel} is out of date (run build_index.py)",
            )
        rebuilt = [{k: as_csv(v) for k, v in t.items()} for t in run_trials]
        if rebuilt != trials_by_report.get(rel, []):
            problems.add(
                f"{where}/trials.csv",
                f"rows for {rel} are out of date (run build_index.py)",
            )


def token_from_dotenv() -> str | None:
    """Read HF_TOKEN from the repository's .env, as the push scripts do."""
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    if not dotenv.is_file():
        return None
    for line in dotenv.read_text().splitlines():
        key, _, value = line.removeprefix("export ").partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip("\"'") or None
    return None


def check_hf(env_dirs: list[Path], problems: Problems) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import DatasetCard, HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or token_from_dotenv()
    if not token:
        problems.add("huggingface", "HF_TOKEN is not set; the datasets are private")
        return

    def config_rows(repo: str) -> dict[str, int]:
        info = (
            DatasetCard.load(repo, repo_type="dataset", token=token).data.get(
                "dataset_info"
            )
            or []
        )
        info = info if isinstance(info, list) else [info]
        return {
            c.get("config_name", "default"): sum(
                s["num_examples"] for s in c.get("splits", [])
            )
            for c in info
        }

    def table(repo: str, config: str) -> list[dict]:
        files = [
            f
            for f in HfApi(token=token).list_repo_files(repo, repo_type="dataset")
            if f.startswith(f"{config}/") and f.endswith(".parquet")
        ]
        rows: list[dict] = []
        for name in sorted(files):
            path = hf_hub_download(repo, name, repo_type="dataset", token=token)
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    hint = "run scripts/push_october_reports_to_hf.py and scripts/push_october_summary_to_hf.py"
    full = config_rows(FULL_REPO)
    summary = config_rows(SUMMARY_REPO)
    all_runs, all_trials = [], []
    for env_dir in env_dirs:
        _, runs = read_csv(env_dir / "runs.csv")
        _, trials = read_csv(env_dir / "trials.csv")
        all_runs += runs
        all_trials += trials
        for level in sorted({r["level"] for r in trials}):
            config = f"{env_dir.name}-level_{level}"
            expected = sum(r["level"] == level for r in trials)
            if full.get(config) != expected:
                problems.add(
                    FULL_REPO,
                    f"config {config} has {full.get(config)} rows, expected {expected} ({hint})",
                )
    for config, expected in (
        ("runs", len(all_runs)),
        ("tasks", None),
        ("trials", len(all_trials)),
    ):
        if config not in summary or (
            expected is not None and summary[config] != expected
        ):
            problems.add(
                SUMMARY_REPO,
                f"config {config} has {summary.get(config)} rows, expected {expected} ({hint})",
            )
    if full.get("runs") != len(all_runs):
        problems.add(
            FULL_REPO,
            f"config runs has {full.get('runs')} rows, expected {len(all_runs)} ({hint})",
        )
    if problems:
        return

    hf_runs = {(r["environment"], r["run_id"]): r for r in table(SUMMARY_REPO, "runs")}
    for env_dir in env_dirs:
        for run in read_csv(env_dir / "runs.csv")[1]:
            hf = hf_runs.get((env_dir.name, run["run_id"]))
            if hf is None or hf["report_sha256"] != run["report_sha256"]:
                problems.add(
                    SUMMARY_REPO,
                    f"run {run['run_id']} is missing or built from a different report ({hint})",
                )
    hf_trials = {r["execution_id"]: r for r in table(SUMMARY_REPO, "trials")}
    for row in all_trials:
        hf = hf_trials.get(row["execution_id"])
        if (
            hf is None
            or as_csv(hf["score"]) != row["score"]
            or str(hf["success"]) != row["success"]
            or hf["langfuse_trace_id"] != row["langfuse_trace_id"]
        ):
            problems.add(
                SUMMARY_REPO,
                f"trial {row['execution_id']} is missing or differs from trials.csv ({hint})",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hf", action="store_true", help="also check the Hugging Face datasets"
    )
    args = parser.parse_args()

    problems = Problems()
    warnings = Problems()
    env_dirs = environments()
    if not env_dirs:
        problems.add("reports", "no environment folders found")
    for env_dir in env_dirs:
        reports = check_layout(env_dir, problems, warnings)
        if not (env_dir / "runs.csv").exists() or not (env_dir / "trials.csv").exists():
            continue
        check_indexes(env_dir, reports, problems)
        skipped = sum(is_lfs_pointer(r) for r in reports)
        sys.stdout.write(
            f"{env_dir.name}: {len(reports)} reports checked"
            + (f" ({skipped} Git LFS pointers not rebuilt)" if skipped else "")
            + "\n"
        )
    if args.hf and not problems:
        check_hf(env_dirs, problems)
        if not problems:
            sys.stdout.write("Hugging Face datasets match the reports\n")

    for warning in warnings:
        sys.stdout.write(f"WARNING {warning}\n")
    for problem in problems:
        sys.stdout.write(f"ERROR {problem}\n")
    if problems:
        sys.stdout.write(
            f"{len(problems)} problem(s); see reports/README.md for the convention\n"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
