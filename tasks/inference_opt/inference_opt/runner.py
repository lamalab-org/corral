"""Run a policy on one split: jailed host, controller gateway, then grading.

Without Docker permission enforcement the host is a plain subprocess, so
metering holds but the labels are not protected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corral.runtime import permissions
from corral.runtime.service_channel import ServiceServer, socket_pair
from inference_opt import datasets
from inference_opt.gateway import Backend, StudentGateway, backend_for
from inference_opt.grading import Grader, GradeSpec
from inference_opt.host import PolicyHostTool
from inference_opt.outcomes import read_outcomes

__all__ = [
    "DEFAULT_RUN_TIME_LIMIT_S",
    "DEFAULT_STUDENT_CONCURRENCY",
    "MAX_TOKENS_CAP",
    "ModelRun",
    "private_root",
    "run_policy",
]

#: Concurrent student requests per run when the task config does not set
#: ``student_concurrency``.
DEFAULT_STUDENT_CONCURRENCY = 16
#: Wall clock for one run; answers submitted before it still count.
DEFAULT_RUN_TIME_LIMIT_S = 3600
#: Ceiling on ``max_tokens`` for any one student call.
MAX_TOKENS_CAP = 32768


@dataclass
class ModelRun:
    """What one policy run on one student produced."""

    model: str
    split: str
    question_ids: list[str]
    answers: dict[str, str] = field(default_factory=dict)
    correct: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    logs: dict[str, list[str]] = field(default_factory=dict)
    per_question: dict[str, dict[str, int]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    calls_used: int = 0
    calls_reserved: int = 0
    output_tokens: int = 0
    budget_exhausted: bool = False
    manifest: dict[str, Any] = field(default_factory=dict)
    #: The policy could not be imported; nothing ran.
    load_error: str = ""
    #: The run stopped early: a crash in ``run()``, or the time limit.
    error: str = ""
    #: The student never answered; the environment, not the policy, failed.
    infrastructure_error: str = ""
    grading_error: str = ""

    @property
    def n_items(self) -> int:
        return len(self.question_ids)

    @property
    def n_correct(self) -> int:
        return sum(self.correct.get(item, False) for item in self.question_ids)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_items if self.n_items else 0.0

    @property
    def n_crashed(self) -> int:
        return sum(
            1 for text in self.errors.values() if not text.startswith("budget_exhausted")
        )


def private_root(work_dir: Path) -> Path:
    """Controller-only storage beside the workspace, never inside it.

    In Docker the jail mounts only the task's own folder, and the agent's file
    tools are bound to it, so a sibling folder is out of reach of both.
    """
    work_dir = Path(work_dir).resolve()
    return work_dir.parent / ".inference-opt-private" / work_dir.name


def run_policy(
    *,
    config: dict[str, Any],
    work_dir: Path,
    run_id: str,
    policy_dir: Path,
    split: str,
    models: list[str],
    budget_per_model: int,
    examples: list[dict[str, Any]],
    item_ids: list[str] | None = None,
    write_copy: bool = True,
    max_tokens_cap: int = MAX_TOKENS_CAP,
    time_limit_s: int | None = None,
    backends: dict[str, Backend] | None = None,
) -> dict[str, ModelRun]:
    """Run ``policy_dir`` on one split for each model, side by side, and grade it."""
    work_dir = Path(work_dir).resolve()
    benchmark = str(config["benchmark"])
    items = datasets.load_items(benchmark, split)  # type: ignore[arg-type]
    if item_ids:
        wanted = set(item_ids)
        items = [item for item in items if item.item_id in wanted]
    records = [
        {**datasets.public_record(item), "index": index}
        for index, item in enumerate(items)
    ]
    targets = datasets.load_targets(benchmark, split)  # type: ignore[arg-type]
    private = private_root(work_dir) / run_id
    shutil.rmtree(private, ignore_errors=True)
    workspace_copy = work_dir / "runs" / run_id
    if write_copy:
        shutil.rmtree(workspace_copy, ignore_errors=True)
    limit = int(time_limit_s or config.get("run_time_limit_s", DEFAULT_RUN_TIME_LIMIT_S))
    concurrency = int(config.get("student_concurrency", DEFAULT_STUDENT_CONCURRENCY))

    def one(model: str) -> ModelRun:
        model_spec = str((config.get("model_specs") or {}).get(model, model))
        backend = (backends or {}).get(model) or backend_for(
            model_spec,
            (config.get("base_urls") or {}).get(model),
            os.environ.get("VLLM_API_KEY"),
            float(limit),
        )
        gateway = StudentGateway(
            backend=backend,
            budget=budget_per_model,
            max_tokens_cap=max_tokens_cap,
            question_ids=[record["item_id"] for record in records],
            policy_api=str(config.get("policy_api", "primitive")),
        )
        request = {
            "policy_dir": "",
            "questions": records,
            "examples": examples,
            "benchmark": benchmark,
            "policy_api": str(config.get("policy_api", "primitive")),
            "budget": budget_per_model,
        }
        result = _execute_host(
            request, gateway, work_dir, Path(policy_dir), limit, concurrency
        )
        run = ModelRun(
            model=model,
            split=split,
            question_ids=[record["item_id"] for record in records],
            answers=dict(gateway.answers),
            errors=dict(gateway.errors),
            logs={key: list(lines) for key, lines in gateway.logs.items()},
            per_question={key: dict(value) for key, value in gateway.per_question.items()},
            calls=list(gateway.calls),
            calls_used=gateway.used,
            calls_reserved=budget_per_model,
            output_tokens=gateway.output_tokens,
            budget_exhausted=gateway.exhausted,
            manifest=dict(result.get("manifest") or {}),
            load_error=str(result.get("load_error") or ""),
            error=str(result.get("error") or ""),
        )
        if gateway.backend_failures and not gateway.backend_successes:
            failed = next((call["error"] for call in gateway.calls if call["error"]), "")
            run.infrastructure_error = f"the student never answered: {failed}"
        if not (run.load_error or run.error or run.answers or gateway.errors):
            run.error = (
                "the policy finished without submitting any answer; call "
                "ctx.submit(question_id, answer) or return {question_id: answer} "
                "from run()"
            )
        if not run.load_error:
            _grade(run, records, targets, private / model, benchmark)
        if write_copy:
            _write_copy(run, workspace_copy / model, run_id)
        return run

    with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
        runs = list(pool.map(one, models))
    return {run.model: run for run in runs}


def _execute_host(
    request: dict[str, Any],
    gateway: StudentGateway,
    work_dir: Path,
    policy_dir: Path,
    time_limit_s: int,
    concurrency: int,
) -> dict[str, Any]:
    if permissions.enabled():
        relative = Path(policy_dir).resolve().relative_to(work_dir)
        request = {**request, "policy_dir": str(Path("/workspace") / relative)}
        cancel = threading.Event()
        timer = threading.Timer(time_limit_s, cancel.set)
        timer.start()
        try:
            reply = permissions.run_worker(
                "tool",
                (PolicyHostTool(), {"request": json.dumps(request)}),
                str(work_dir),
                cancel=cancel,
                workspace_access="read_write",
                service=gateway.handle,
                service_workers=concurrency,
            )
            return json.loads(reply["content"])
        except RuntimeError as exc:
            if cancel.is_set():
                return {"error": f"time limit of {time_limit_s}s reached"}
            return {"error": str(exc)[-4000:]}
        finally:
            timer.cancel()
    return _execute_local(
        {**request, "policy_dir": str(Path(policy_dir).resolve())},
        gateway,
        work_dir,
        time_limit_s,
        concurrency,
    )


def _execute_local(
    request: dict[str, Any],
    gateway: StudentGateway,
    work_dir: Path,
    time_limit_s: int,
    concurrency: int,
) -> dict[str, Any]:
    """The same host as a plain subprocess, for runs without Docker."""
    scratch = Path(tempfile.mkdtemp(prefix="inference-opt-host-"))
    controller_end, worker_end = socket_pair()
    server = ServiceServer(controller_end, gateway.handle, workers=concurrency).start()
    try:
        request_path = scratch / "request.json"
        result_path = scratch / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        # The host gets no credentials and no labels path, only module paths.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join(sys.path),
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
        }
        with (scratch / "host.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "inference_opt.host",
                    str(request_path),
                    str(result_path),
                    str(worker_end.fileno()),
                ],
                cwd=work_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(worker_end.fileno(),),
            )
            worker_end.close()
            try:
                process.wait(timeout=time_limit_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return {"error": f"time limit of {time_limit_s}s reached"}
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        tail = (scratch / "host.log").read_text(errors="replace")[-4000:]
        return {"error": f"policy host exited with code {process.returncode}: {tail}"}
    finally:
        worker_end.close()
        server.close()
        shutil.rmtree(scratch, ignore_errors=True)


def _grade(
    run: ModelRun,
    records: list[dict[str, Any]],
    targets: dict[str, str],
    out_dir: Path,
    benchmark: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = out_dir / "questions.jsonl"
    answers = out_dir / "answers.json"
    datasets.write_jsonl(questions, records)
    answers.write_text(json.dumps(run.answers), encoding="utf-8")
    spec = GradeSpec(
        run_id=f"{run.split}-{run.model}",
        questions_path=str(questions),
        answers_path=str(answers),
        out_dir=str(out_dir),
        benchmark=benchmark,
        split=run.split,
    )
    summary = Grader().grade(spec, targets)
    if not summary.ok:
        run.grading_error = summary.error[-4000:]
        return
    run.correct = {item.item_id: item.correct for item in read_outcomes(spec.log_dir)}
    # Student calls and graded logs stay here, not in the workspace.
    (out_dir / "student_calls.jsonl").write_text(
        "".join(json.dumps(call, ensure_ascii=False) + "\n" for call in run.calls),
        encoding="utf-8",
    )


def _write_copy(run: ModelRun, folder: Path, run_id: str) -> None:
    """What the agent may read about a train run: no targets anywhere."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item_id in run.question_ids:
            usage = run.per_question.get(item_id, {})
            row = {
                "item_id": item_id,
                "answer": run.answers.get(item_id, ""),
                "submitted": item_id in run.answers,
                "correct": run.correct.get(item_id) if run.correct else None,
                "calls": usage.get("calls", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "error": run.errors.get(item_id, ""),
                "log": run.logs.get(item_id, []),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (folder / "student_calls.jsonl").open("w", encoding="utf-8") as stream:
        for call in run.calls:
            stream.write(json.dumps(call, ensure_ascii=False) + "\n")
    summary = {
        "run_id": run_id,
        "model": run.model,
        "split": run.split,
        "n_items": run.n_items,
        "n_submitted": len(run.answers),
        "n_correct": run.n_correct,
        "accuracy": round(run.accuracy, 4),
        "calls_used": run.calls_used,
        "calls_reserved": run.calls_reserved,
        "output_tokens": run.output_tokens,
        "budget_exhausted": run.budget_exhausted,
        "manifest": run.manifest,
        "load_error": run.load_error,
        "error": run.error,
        "infrastructure_error": run.infrastructure_error,
        "run_log": run.logs.get("_run", []),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
