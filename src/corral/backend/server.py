import contextlib
import shutil
import threading
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from loguru import logger

from corral.backend.env import Environment
from corral.backend.mcp_server import (
    DEFAULT_MCP_THREAD_POOL_SIZE,
    attach_task_mcp_servers,
    attach_trial_mcp_server,
    install_mcp_thread_capacity,
)
from corral.backend.schema import (
    ToLatexRequest,
    ToolRequest,
    TrialArtifactPromotionRequest,
    TrialArtifactPromotionResponse,
    TrialCompletionResponse,
    TrialCreatedResponse,
    TrialCreateRequest,
)
from corral.backend.trial_runtime import (
    InProcessTrialRuntime,
    TrialRuntime,
    build_mcp_tools_payload,
    build_openai_tools_payload,
)
from corral.backend.trial_worker import (
    DEFAULT_TRIAL_WORKER_POOL_SIZE,
    BuildEnvs,
    TrialWorkerPool,
    WorkerHandle,
    WorkerTrialRuntime,
)
from corral.router.verbosity import (
    ToolVerbosity,
    get_tools_guide_with_verbosity,
)

if TYPE_CHECKING:
    from corral.backend.state import TaskRunState


def _finalize_trial(
    env: Environment, score: float, surrendered: bool = False
) -> TrialCompletionResponse:
    """
    Finalize a trial by capturing state and resetting environment.

    This centralizes the 3-step finalization process:
    1. Capture completed trial data (before reset)
    2. Reset environment state for next trial
    3. Return structured response

    Args:
        env: The environment instance
        score: Trial score
        surrendered: Whether trial was surrendered

    Returns:
        TrialCompletionResponse with all trial completion data
    """
    # Get completed trial data before reset
    completed_trial = env.get_completed_trial_data()

    # Reset state for next trial
    finished_trial_id = env.reset_state()

    return TrialCompletionResponse(
        score=score,
        state=completed_trial["state"],
        trial_id=finished_trial_id,
        surrendered=surrendered,
    )


def create_benchmark_server(
    environments: dict[str, Environment],
    *,
    build_envs: BuildEnvs | None = None,
    trial_worker_pool_size: int = DEFAULT_TRIAL_WORKER_POOL_SIZE,
    trial_worker_start_method: str | None = None,
    mcp_thread_pool_size: int = DEFAULT_MCP_THREAD_POOL_SIZE,
) -> FastAPI:
    app = FastAPI()

    # Process-worker backend (Environment Concurrency Isolation, Phase 2). Envs
    # that keep process-global state (`concurrency="process"`, e.g. wetlab) can
    # only be run concurrently by giving each trial its own OS process. That
    # needs a `build_envs` the worker can call to reconstruct the environments
    # (a live `Environment` won't pickle — reaktoro's C++ objects don't), so the
    # pool exists only when a builder is supplied *and* some env actually asks
    # for it. Without it, "process" envs fall back to the in-process backend and
    # rely on the scheduler's Phase-0 serial gate — unchanged behaviour.
    needs_workers = any(
        getattr(env, "concurrency", "thread") == "process"
        for env in environments.values()
    )
    trial_worker_pool: TrialWorkerPool | None = None
    if build_envs is not None and needs_workers:
        trial_worker_pool = TrialWorkerPool(
            build_envs,
            trial_worker_pool_size,
            start_method=trial_worker_start_method,
        )
    elif needs_workers:
        logger.warning(
            "Env(s) declare concurrency='process' but no build_envs was given; "
            "running them in-process (the scheduler serialises them). Pass "
            "build_envs to enable the process-worker backend."
        )

    # Worker pinned to each in-flight *episode* so a dependency chain's tasks
    # share one worker (and thus one dependency-output store, which cannot cross
    # a process boundary). Guarded by a lock because concurrent `create_trial`s
    # for sibling tasks of one episode may race to pin it.
    episode_workers: dict[str, WorkerHandle] = {}
    episode_workers_lock = threading.Lock()

    # Active per-trial runtimes, keyed by `trial_runtime_id`. Populated by
    # `POST /tasks/{task_id}/trials` and drained by `DELETE /trials/{id}`. Each
    # value is a `TrialRuntime` — an `InProcessTrialRuntime` or a
    # `WorkerTrialRuntime` depending on the env's `concurrency` mode — so the
    # handlers below never depend on which backend executes a trial.
    trial_registry: dict[str, TrialRuntime] = {}

    # Dependency-output stores keyed by `episode_id`. One trial round of a
    # dependency chain is an *episode*: every trial runtime created for that
    # episode shares this store, so a downstream task reads its upstream
    # siblings' outputs while keeping its own isolated state and workspace.
    # Created lazily on the first `create_trial` for an episode and freed by
    # `DELETE /episodes/{episode_id}`. Different rounds get different episode
    # ids, so outputs are never shared across trial rounds.
    episode_registry: dict[str, dict[str, TaskRunState]] = {}

    @app.get("/tasks")
    def get_available_tasks():
        """Get list of available task IDs"""
        return list(environments.keys())

    @app.get("/dependency_chain")
    def get_dependency_chain_setting():
        # The graph lives on the task definitions; every environment exposes
        # its component's tasks through `group_tasks`.
        has_chained_tasks = any(
            task.dependencies()
            for env in environments.values()
            for task in env.group_tasks.values()
        )
        return {"dependency_chain": has_chained_tasks}

    @app.get("/dependency_graph")
    def get_dependency_graph():
        """Expose the dependency graph so the runner can order/close the run."""
        return {
            task_id: sorted(env.current_task.dependencies())
            for task_id, env in environments.items()
        }

    @app.get("/concurrency")
    def get_concurrency_modes():
        """Expose each task's *declared* env concurrency mode for the scheduler.

        The concurrent scheduler reads this to serialise envs that keep
        process-global state (`"process"`/`"serial"`) so they never overlap
        another globally-stateful trial (Environment Concurrency Isolation,
        Phase 0). Envs without the attribute predate the flag and are reported as
        the safe-to-overlap default, `"thread"`.

        This reports the *declared* mode only; whether a `"process"` env is
        actually isolated in its own worker process (and so may overlap) also
        depends on a live worker pool, which the scheduler reads separately from
        `/trial_worker` (Phase 3).
        """
        return {
            task_id: getattr(env, "concurrency", "thread")
            for task_id, env in environments.items()
        }

    @app.get("/trial_worker")
    def get_trial_worker_status():
        """Report whether the process-worker backend is live on this server.

        `"process"` envs only overlap safely when each trial runs in its own
        worker process, which requires a live pool (`build_envs` was supplied and
        some env is `"process"`). The concurrent scheduler reads this to decide
        whether to route independent `"process"` trials through the worker pool
        and stop serialising them (Environment Concurrency Isolation, Phase 3).
        When inactive, `"process"` envs fall back to the in-process backend and
        the scheduler keeps serialising them (Phase-0 behaviour).
        """
        return {
            "active": trial_worker_pool is not None,
            "size": trial_worker_pool.size if trial_worker_pool is not None else None,
        }

    @app.get("/tasks/{task_id}/prompt")
    def get_task_prompt(task_id: str):
        """Get the task prompt for the agent"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        env = environments[task_id]
        return {"prompt": env.get_task_prompt()}

    @app.get("/tasks/{task_id}/guide")
    def get_environment_guide(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        """Get the complete environment guide with specified tool verbosity"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        task_prompt = env.get_task_prompt()
        tools_guide = get_tools_guide_with_verbosity(env, verbosity)

        return {"prompt": f"Task: {task_prompt}\n\n{tools_guide}"}

    @app.get("/tasks/{task_id}/tools/guide")
    def get_tools_guide(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        """Get the tools guide with specified verbosity level"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        tools_guide = get_tools_guide_with_verbosity(env, verbosity)

        return {"prompt": tools_guide}

    @app.get("/tasks/{task_id}/tools")
    def get_available_tools(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        """
        Get available tools for a task in OpenAI function-calling format.

        Returns tool definitions directly usable with LLM function-calling APIs.
        Descriptions are filtered based on the requested verbosity level.
        """
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        return build_openai_tools_payload(environments[task_id], verbosity)

    @app.get("/tasks/{task_id}/tools/mcp")
    def get_mcp_tool_schema(
        task_id: str,
        verbosity: ToolVerbosity | None = None,
    ):
        """Return the task's tools in the *MCP* representation, plus a digest.

        This is the exact schema an MCP client (Codex, Claude Code, ...) receives
        from tools/list — produced by :meth:`Tool.to_mcp` — so a harness can
        record a hash of what it actually saw, rather than the REST/OpenAI
        function-calling schema (which is only logically equivalent).
        """
        if verbosity is None:
            verbosity = Query(
                ToolVerbosity.FULL, description="Tool description verbosity level"
            )
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        return build_mcp_tools_payload(environments[task_id], verbosity)

    @app.post("/tasks/{task_id}/tools/execute")
    def execute_tool(task_id: str, request: ToolRequest):
        """Execute a tool in the environment"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        try:
            result = environments[task_id].call_tool(
                request.tool_name, request.arguments
            )
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # Trial runtimes: isolated per-trial executions of a task.
    #
    # `POST /tasks/{task_id}/trials` mints a fresh, isolated runtime (its own
    # CorralState + workspace) and registers it under a `trial_runtime_id`. All
    # mutable operations then address `/trials/{trial_runtime_id}/...`, so two
    # concurrent trials — including repeated trials of the *same* task — never
    # collide. The `/tasks/{task_id}/...` routes above remain the sequential
    # (single-runtime-per-task) compatibility surface.
    def _require_trial(trial_runtime_id: str) -> TrialRuntime:
        runtime = trial_registry.get(trial_runtime_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Trial runtime not found")
        return runtime

    def _copy_trial_artifacts(source: str, destination: str) -> list[str]:
        """Overlay regular files from one registered workspace onto another."""
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if not source_path.is_dir():
            raise HTTPException(status_code=409, detail="Source workspace is missing")
        if source_path == destination_path:
            return []
        if (
            source_path in destination_path.parents
            or destination_path in source_path.parents
        ):
            raise HTTPException(
                status_code=409,
                detail="Artifact workspaces must not contain one another",
            )

        destination_path.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for item in source_path.rglob("*"):
            # Branches are model-controlled. Do not promote symlinks that could
            # point outside either registered trial workspace.
            if item.is_symlink():
                continue
            relative = item.relative_to(source_path)
            target = destination_path / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                copied.append(relative.as_posix())
        return copied

    def _acquire_worker(episode_id: str | None) -> tuple[WorkerHandle, bool]:
        """Lease a worker for a process-trial, reusing an episode's pinned one.

        Returns `(handle, pinned)`. A standalone trial (`episode_id is None`)
        leases a fresh worker and owns it until it closes. An episode's first
        task pins a worker so its later tasks reuse it — and thus the worker's
        dependency-output store; the lease is not returned until the episode
        closes. The lock is dropped across the (blocking) `lease()` so sibling
        opens for *other* episodes aren't stalled, and a lost pin race hands the
        surplus worker straight back.
        """
        assert trial_worker_pool is not None
        if episode_id is None:
            return trial_worker_pool.lease(), False
        with episode_workers_lock:
            handle = episode_workers.get(episode_id)
            if handle is not None:
                return handle, True
        handle = trial_worker_pool.lease()
        with episode_workers_lock:
            existing = episode_workers.get(episode_id)
            if existing is not None:  # a sibling pinned first; return ours
                trial_worker_pool.release(handle)
                return existing, True
            episode_workers[episode_id] = handle
            return handle, True

    @app.post("/tasks/{task_id}/trials")
    def create_trial(task_id: str, request: TrialCreateRequest) -> TrialCreatedResponse:
        """Open a fresh, isolated runtime for one trial of `task_id`."""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        trial_runtime_id = f"tr_{uuid.uuid4().hex[:16]}"

        # Route the trial to the process-worker backend when its env keeps
        # process-global state and a worker pool exists; otherwise run it
        # in-process. Both satisfy the `TrialRuntime` protocol, so every handler
        # downstream is oblivious to which backend executed the trial.
        use_worker = (
            trial_worker_pool is not None
            and getattr(environments[task_id], "concurrency", "thread") == "process"
        )
        if use_worker:
            handle, pinned = _acquire_worker(request.episode_id)
            worker_runtime = WorkerTrialRuntime(
                trial_worker_pool,
                handle,
                task_id,
                # Episode-pinned workers stay leased until the episode closes;
                # a standalone trial returns its worker to the pool on close.
                release_on_close=not pinned,
            )
            workspace = worker_runtime.open(
                trial_runtime_id=trial_runtime_id,
                episode_id=request.episode_id,
                benchmark_run_id=request.benchmark_run_id,
                tool_jobs_per_trial=request.tool_jobs_per_trial,
            )
            runtime: TrialRuntime = worker_runtime
        else:
            # Resolve (or open) the episode's shared dependency-output store so
            # tasks chained within one trial round see each other's outputs.
            # `setdefault` is a single atomic dict op, so concurrent sibling
            # `create_trial`s for the same episode share one store without a lock.
            episode_store = (
                episode_registry.setdefault(request.episode_id, {})
                if request.episode_id is not None
                else None
            )
            env = environments[task_id].for_trial(
                trial_runtime_id,
                episode_task_runs=episode_store,
                # Size the runtime's background-job pool from the request when the
                # caller pins it (ConcurrencyConfig.tool_jobs_per_trial); `None`
                # keeps the environment's own default.
                max_job_concurrency=request.tool_jobs_per_trial,
            )
            # Bind the benchmark-run/episode provenance to the runtime's state.
            env.state.run_id = request.benchmark_run_id
            env.state.episode_id = request.episode_id
            runtime = InProcessTrialRuntime(env)
            workspace = runtime.workspace

        trial_registry[trial_runtime_id] = runtime

        return TrialCreatedResponse(
            trial_runtime_id=trial_runtime_id,
            task_id=task_id,
            workspace=workspace,
            mcp_url=f"/trials/{trial_runtime_id}/mcp",
        )

    @app.delete("/trials/{trial_runtime_id}")
    def close_trial(trial_runtime_id: str):
        """Discard a trial runtime and free its slot in the registry."""
        runtime = trial_registry.pop(trial_runtime_id, None)
        if runtime is None:
            raise HTTPException(status_code=404, detail="Trial runtime not found")
        # Release any background-job threads so nothing outlives the trial.
        try:
            runtime.close()
        except Exception as exc:  # cleanup must not fail the close
            logger.warning(
                f"Failed to shut down jobs for trial {trial_runtime_id}: {exc}"
            )
        return {"status": "closed", "trial_runtime_id": trial_runtime_id}

    @app.post("/trials/{trial_runtime_id}/artifacts/promote")
    def promote_trial_artifacts(
        trial_runtime_id: str,
        request: TrialArtifactPromotionRequest,
    ) -> TrialArtifactPromotionResponse:
        """Overlay a branch workspace onto its canonical scored workspace."""
        source = _require_trial(trial_runtime_id)
        destination = _require_trial(request.destination_trial_runtime_id)
        if source.task_id != destination.task_id:
            raise HTTPException(
                status_code=409,
                detail="Artifacts can only be promoted between trials of one task",
            )
        if source.workspace is None or destination.workspace is None:
            raise HTTPException(
                status_code=409,
                detail="Both trials must expose filesystem workspaces",
            )
        files = _copy_trial_artifacts(source.workspace, destination.workspace)
        return TrialArtifactPromotionResponse(
            source_trial_runtime_id=trial_runtime_id,
            destination_trial_runtime_id=request.destination_trial_runtime_id,
            files=files,
        )

    @app.delete("/episodes/{episode_id}")
    def close_episode(episode_id: str):
        """Free an episode's shared dependency-output store.

        Called once every runtime in the episode has finished, so its accumulated
        task outputs never outlive the trial round that produced them. For a
        worker-backed episode the store lives in the pinned worker, so this also
        drops it there and returns the worker to the pool.
        """
        had_store = episode_registry.pop(episode_id, None) is not None

        with episode_workers_lock:
            handle = episode_workers.pop(episode_id, None)
        had_worker = handle is not None
        if handle is not None and trial_worker_pool is not None:
            # Drop the episode's store inside the worker, then free the worker
            # regardless of whether that RPC succeeded (a wedged worker must not
            # leak a pool slot).
            try:
                with handle.lock:
                    handle.conn.send(("drop_episode", {"episode_id": episode_id}))
                    handle.conn.recv()
            except Exception as exc:
                logger.warning(
                    f"Failed to drop episode store for {episode_id} in worker: {exc}"
                )
            finally:
                trial_worker_pool.release(handle)

        if not had_store and not had_worker:
            raise HTTPException(status_code=404, detail="Episode not found")
        return {"status": "closed", "episode_id": episode_id}

    @app.get("/trials/{trial_runtime_id}/prompt")
    def get_trial_prompt(trial_runtime_id: str):
        return {"prompt": _require_trial(trial_runtime_id).get_task_prompt()}

    @app.get("/trials/{trial_runtime_id}/guide")
    def get_trial_guide(
        trial_runtime_id: str, verbosity: ToolVerbosity = ToolVerbosity.FULL
    ):
        return {"prompt": _require_trial(trial_runtime_id).guide(verbosity)}

    @app.get("/trials/{trial_runtime_id}/tools")
    def get_trial_tools(
        trial_runtime_id: str, verbosity: ToolVerbosity = ToolVerbosity.FULL
    ):
        return _require_trial(trial_runtime_id).tools_payload(verbosity)

    @app.get("/trials/{trial_runtime_id}/tools/mcp")
    def get_trial_mcp_tools(
        trial_runtime_id: str, verbosity: ToolVerbosity = ToolVerbosity.FULL
    ):
        return _require_trial(trial_runtime_id).mcp_tools_payload(verbosity)

    @app.post("/trials/{trial_runtime_id}/tools/execute")
    def execute_trial_tool(trial_runtime_id: str, request: ToolRequest):
        runtime = _require_trial(trial_runtime_id)
        try:
            result = runtime.call_tool(request.tool_name, request.arguments)
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/trials/{trial_runtime_id}/state")
    def get_trial_snapshot(trial_runtime_id: str):
        return _require_trial(trial_runtime_id).snapshot()

    @app.get("/trials/{trial_runtime_id}/status")
    def get_trial_status(trial_runtime_id: str):
        return _require_trial(trial_runtime_id).status()

    @app.post("/trials/{trial_runtime_id}/configure")
    def configure_trial_apps(trial_runtime_id: str):
        result = _require_trial(trial_runtime_id).configure()
        return {**result, "trial_runtime_id": trial_runtime_id}

    @app.post("/trials/{trial_runtime_id}/submit")
    def submit_trial_answer(
        trial_runtime_id: str, answer: dict
    ) -> TrialCompletionResponse:
        return _require_trial(trial_runtime_id).submit(answer["answer"])

    @app.post("/trials/{trial_runtime_id}/surrender")
    def surrender_trial(trial_runtime_id: str) -> TrialCompletionResponse:
        return _require_trial(trial_runtime_id).surrender()

    @app.get("/tasks/{task_id}/state")
    def get_state(task_id: str):
        """Get the current state of the task"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        return environments[task_id].state.snapshot()

    @app.post("/tasks/{task_id}/submit")
    def submit_answer(task_id: str, answer: dict) -> TrialCompletionResponse:
        """Submit an answer and finalize the trial"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        score = env.submit_answer(answer["answer"])

        return _finalize_trial(env, score, surrendered=False)

    @app.post("/tasks/{task_id}/surrender")
    def surrender_task(task_id: str) -> TrialCompletionResponse:
        """Surrender from a task without submitting an answer"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        score = env.surrender()

        return _finalize_trial(env, score, surrendered=True)

    @app.get("/tasks/{task_id}/status")
    def get_task_status(task_id: str):
        """Get task completion status"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        return {
            "is_attempted": env.state.is_attempted,
            "score": env.state.score,
            "submitted_answer": env.state.submitted_answer,
            "tool_statistics": env.state.tool_statistics(),
        }

    @app.get("/tasks/{task_id}/last_score")
    def get_last_score(task_id: str):
        """Get the score from the most recent trial submission"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        # Get the most recent completed trial
        trial_ids = sorted(env.state.trials.keys(), key=int)
        if not trial_ids:
            raise HTTPException(status_code=404, detail="No trials completed yet")

        latest_trial_id = trial_ids[-1]
        latest_trial = env.state.trials[latest_trial_id]

        return {
            "task_id": task_id,
            "trial_id": latest_trial_id,
            "score": latest_trial["score"],
        }

    @app.get("/tasks/{task_id}/trials")
    def get_all_trials(task_id: str):
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        env = environments[task_id]
        return {"trials": env.state.trials}

    @app.get("/tasks/{task_id}/trials/{trial_id}")
    def get_trial_state(task_id: str, trial_id: str):
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")
        env = environments[task_id]
        trial_state = env.state.trials.get(trial_id)
        if trial_state is None:
            raise HTTPException(status_code=404, detail="Trial not found")
        return {"trial_state": trial_state}

    @app.post("/tasks/{task_id}/configure")
    def configure_additional_apps(task_id: str):
        """Configure external apps/services for this specific task at the beginning of each trail"""
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]
        status = env.configure_additional_apps()

        return {
            "status": status,
            "task_id": task_id,
            "trial_id": env.state.trial_id,
        }

    @app.post("/tasks/{task_id}/latex")
    def generate_latex(task_id: str, request: ToLatexRequest):
        """Generate LaTeX documentation for this task

        Args:
            task_id: The task identifier
            request: LaTeX generation parameters including:
                - output_dir: Directory for output .tex files
                - level: Task level identifier (e.g., 1, 2, "advanced")
                - env_name: Optional environment name (e.g., "afm", "catalyst")
                - task_name: Optional custom name for the task

        Returns:
            Paths to the generated .tex files (task_path, tools_path, and scoring_path)
        """
        if task_id not in environments:
            raise HTTPException(status_code=404, detail="Task not found")

        env = environments[task_id]

        try:
            task_path, tools_path, scoring_path = env.to_latex(
                output_dir=request.output_dir,
                level=request.level,
                env_name=request.env_name,
                task_name=request.task_name,
                verbosity=request.verbosity,
            )
            return {
                "status": "success",
                "task_id": task_id,
                "output_path": task_path,
                "tools_output_path": tools_path,
                "scoring_output_path": scoring_path,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate LaTeX: {e!s}"
            ) from e

    # Expose each task's tools over MCP (Streamable HTTP) at
    # `/tasks/{task_id}/mcp`, alongside the REST API, so generic MCP clients
    # can connect directly without a local proxy. Schemas and execution reuse
    # `Tool.to_mcp` / `Environment.call_tool` (no duplicated tool system).
    attach_task_mcp_servers(app, environments)

    # A single MCP endpoint for *all* trial runtimes, mounted at `/trials`. The
    # `trial_runtime_id` is read from the request path per call and resolved
    # against `trial_registry`, so runtimes created after startup are served
    # without dynamically (un)mounting an endpoint each trial.
    attach_trial_mcp_server(app, trial_registry)

    # Raise the server's anyio worker-thread pool so many concurrent trials
    # firing parallel MCP tool calls (each offloaded to a worker thread) do not
    # serialise on a starved pool (Section 9). Installed last so it wraps the
    # MCP session-manager lifespans and runs its raise at startup.
    install_mcp_thread_capacity(app, mcp_thread_pool_size)

    # Tear the trial-worker pool's processes down with the server so they never
    # outlive it (they are daemonic, but an explicit shutdown is clean and lets
    # tests reclaim processes deterministically).
    if trial_worker_pool is not None:
        _install_pool_shutdown(app, trial_worker_pool)

    return app


def _install_pool_shutdown(app: FastAPI, pool: TrialWorkerPool) -> None:
    """Chain a lifespan onto `app` that shuts the worker pool down on exit."""
    previous = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            async with previous(app):
                yield
        finally:
            pool.shutdown()

    app.router.lifespan_context = lifespan


def run_server(
    environments: Mapping[str, Environment],
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    build_envs: BuildEnvs | None = None,
    trial_worker_pool_size: int = DEFAULT_TRIAL_WORKER_POOL_SIZE,
    trial_worker_start_method: str | None = None,
    mcp_thread_pool_size: int = DEFAULT_MCP_THREAD_POOL_SIZE,
):
    """Run the benchmark server with the provided environments

    Args:
        environments: dictionary of environments
        host: Server host
        port: Server port
        build_envs: Zero-argument, picklable builder that reconstructs the
            environments map inside a worker process. Supply it to enable the
            process-worker backend for `concurrency="process"` envs (wetlab);
            the environments won't pickle, so the worker rebuilds them from this
            (e.g. `functools.partial(create_qualysis_environments, level=2)`).
            When omitted, "process" envs run in-process and the scheduler
            serialises them (Phase-0 behaviour).
        trial_worker_pool_size: Number of worker processes (only spawned when
            `build_envs` is set and some env is "process"). Caps how many
            process-trials run at once; leasing blocks past it.
        trial_worker_start_method: multiprocessing start method for the worker
            pool (`"spawn"`, `"forkserver"`, ...). Defaults to a safe choice for
            the platform.
        mcp_thread_pool_size: Ceiling for the anyio worker-thread pool that
            services offloaded MCP tool calls. Raise it above the default for
            very high trial concurrency (Section 9); it only permits growth and
            threads are still spawned lazily.
    """
    app = create_benchmark_server(
        dict(environments),
        build_envs=build_envs,
        trial_worker_pool_size=trial_worker_pool_size,
        trial_worker_start_method=trial_worker_start_method,
        mcp_thread_pool_size=mcp_thread_pool_size,
    )
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
