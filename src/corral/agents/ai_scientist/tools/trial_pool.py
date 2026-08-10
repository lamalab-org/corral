"""Isolated branch runtimes and replay-on-fork orchestration."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger

from corral.agents.ai_scientist.search.nodes import (
    ExecutedAction,
    ExperimentNode,
    Observation,
)
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools.corral_executor import (
    CorralExecutor,
    ResearchBudget,
    ToolCallBudgetExceeded,
)


class ReplayDiverged(RuntimeError):
    """Raised when replay cannot reconstruct the parent's physical state."""


ReplayEquivalence = Callable[[Observation, Observation], bool]


@dataclass
class ReplayResult:
    """Comparison between an original physical action and its replay."""

    branch_id: str
    exact: bool
    equivalent: bool
    original: ExecutedAction
    replayed: ExecutedAction


@dataclass(frozen=True)
class ArtifactPromotion:
    """Files copied from a winning branch into the scored workspace."""

    branch_id: str
    source_workspace: str
    destination_workspace: str
    files: tuple[str, ...]


@dataclass
class BranchRuntime:
    """The isolated physical realization of one logical tree trajectory."""

    branch_id: str
    trial_runtime_id: str | None
    router: Any
    workspace: str | None
    executor: CorralExecutor
    owns_trial: bool = True
    action_history: list[ExecutedAction] = field(default_factory=list)
    head_node_id: str | None = None
    replay_results: list[ReplayResult] = field(default_factory=list)
    inheritance_method: str = "fresh"


class TrialPool:
    """Create, reuse, reconstruct, and close AI Scientist branch trials."""

    def __init__(
        self,
        *,
        interface: Any,
        task_id: str,
        tools: dict[str, Any] | list[dict[str, Any]],
        max_tool_calls: int,
        max_observation_chars: int | None = 12_000,
        stop_on_error: bool = True,
        replay_equivalence: ReplayEquivalence | None = None,
    ) -> None:
        self.interface = interface
        self.task_id = task_id
        self.tools = tools
        self.max_observation_chars = max_observation_chars
        self.stop_on_error = stop_on_error
        self.replay_equivalence = replay_equivalence
        self.budget = ResearchBudget(max_tool_calls)
        self.active: dict[str, BranchRuntime] = {}
        self.replay_results: list[ReplayResult] = []
        self.trials_created = 0
        self.peak_simultaneous_trials = 0
        self.trials_cloned = 0
        self._next_branch_number = 1
        self._tool_calls: list[dict[str, Any]] = []
        self._tool_calls_lock = Lock()
        required_methods = ("create_trial", "for_trial", "close_trial")
        missing = [
            name
            for name in required_methods
            if not callable(getattr(interface, name, None))
        ]
        if missing:
            raise TypeError(
                "AI Scientist branch isolation requires the Corral trial API; "
                f"missing: {', '.join(missing)}"
            )
        self.isolated = True

    @property
    def remaining_calls(self) -> int:
        return self.budget.remaining

    @property
    def call_count(self) -> int:
        return self.budget.used

    @property
    def supports_cloning(self) -> bool:
        """Whether the environment can checkpoint/clone a live trial."""
        return callable(getattr(self.interface, "clone_trial", None))

    def can_clone(self, parent: ExperimentNode) -> bool:
        """Whether an active runtime is still parked at this checkpoint."""
        return self.supports_cloning and any(
            branch.head_node_id == parent.id and branch.trial_runtime_id is not None
            for branch in self.active.values()
        )

    def create(self) -> BranchRuntime:
        descriptor: dict[str, Any] = self.interface.create_trial(self.task_id)
        return self._runtime_from_descriptor(descriptor)

    def _runtime_from_descriptor(
        self,
        descriptor: dict[str, Any],
        *,
        action_history: list[ExecutedAction] | None = None,
        head_node_id: str | None = None,
        inheritance_method: str = "fresh",
    ) -> BranchRuntime:
        branch_id = f"branch_{self._next_branch_number:04d}"
        self._next_branch_number += 1

        trial_runtime_id = str(descriptor["trial_runtime_id"])
        workspace = descriptor.get("workspace")
        self.trials_created += 1
        try:
            router = self.interface.for_trial(
                trial_runtime_id,
                self.task_id,
                verbosity=getattr(self.interface, "current_verbosity", None),
                workspace=workspace,
            )
            # The top-level benchmark runner configures its own trial before
            # starting the agent, but AI Scientist creates additional isolated
            # runtimes for experiment branches. Those runtimes need the same
            # per-trial setup (for example, spectra tasks bind the sample as a
            # hidden ``h_smiles`` argument) before any branch tool executes.
            configure = getattr(router, "configure_additional_apps", None)
            if callable(configure):
                configure(self.task_id)
        except Exception:
            try:
                self.interface.close_trial(trial_runtime_id)
            except Exception as close_error:  # pragma: no cover - transport cleanup
                logger.warning(
                    "Failed to close unscoped AI Scientist trial {}: {}",
                    trial_runtime_id,
                    close_error,
                )
            raise

        history = list(action_history or [])
        executor = CorralExecutor(
            interface=router,
            task_id=self.task_id,
            tools=self.tools,
            max_tool_calls=self.budget.max_physical_tool_calls,
            max_observation_chars=self.max_observation_chars,
            stop_on_error=self.stop_on_error,
            budget=self.budget,
            on_executed=history.append,
            on_tool_call=self._record_tool_call,
        )
        branch = BranchRuntime(
            branch_id=branch_id,
            trial_runtime_id=trial_runtime_id,
            router=router,
            workspace=workspace,
            executor=executor,
            owns_trial=True,
            action_history=history,
            head_node_id=head_node_id,
            inheritance_method=inheritance_method,
        )
        self.active[branch_id] = branch
        self.peak_simultaneous_trials = max(
            self.peak_simultaneous_trials,
            sum(runtime.owns_trial for runtime in self.active.values()),
        )
        return branch

    def acquire(
        self,
        parent: ExperimentNode | None,
        tree: ExperimentTree,
        *,
        prefer_existing: bool,
        prefer_clone: bool = False,
    ) -> BranchRuntime:
        """Get a runtime at ``parent``, cloning or replaying when needed."""
        if parent is None:
            return self.create()
        if prefer_existing and parent.branch_id is not None:
            existing = self.active.get(parent.branch_id)
            if existing is not None and existing.head_node_id == parent.id:
                return existing
        if prefer_clone and self.can_clone(parent):
            return self._clone(parent, tree)
        return self._fork(parent, tree)

    def replay_cost(
        self,
        parent: ExperimentNode | None,
        tree: ExperimentTree,
        *,
        prefer_existing: bool,
        prefer_clone: bool = False,
    ) -> int:
        """Return physical calls needed to place a runtime at ``parent``."""
        if parent is None:
            return 0
        if prefer_existing and parent.branch_id is not None:
            existing = self.active.get(parent.branch_id)
            if existing is not None and existing.head_node_id == parent.id:
                return 0
        if prefer_clone and self.can_clone(parent):
            return 0
        return len(tree.executed_trajectory(parent.id))

    def commit(self, branch: BranchRuntime, node: ExperimentNode, start: int) -> None:
        """Attach newly executed actions to a node and advance the branch head."""
        node.executed_actions = list(branch.action_history[start:])
        branch.head_node_id = node.id

    def close(self, branch_id: str) -> None:
        branch = self.active[branch_id]
        if branch.owns_trial and branch.trial_runtime_id is not None:
            self.interface.close_trial(branch.trial_runtime_id)
        self.active.pop(branch_id)

    def close_all(self) -> list[Exception]:
        """Best-effort cleanup of every runtime created by this pool."""
        errors: list[Exception] = []
        for branch_id in list(self.active):
            try:
                self.close(branch_id)
            except Exception as exc:  # pragma: no cover - transport cleanup
                errors.append(exc)
                logger.warning(
                    "Failed to close AI Scientist branch {}: {}", branch_id, exc
                )
        return errors

    def stats(self) -> dict[str, int]:
        return {
            "scientific_tool_calls": self.budget.scientific_calls,
            "replay_tool_calls": self.budget.replay_calls,
            "physical_tool_calls": self.budget.used,
            "trial_runtimes_created": self.trials_created,
            "trial_runtimes_cloned": self.trials_cloned,
            "peak_simultaneous_trials": self.peak_simultaneous_trials,
            "nonexact_replays": sum(not item.exact for item in self.replay_results),
            "replay_divergences": sum(
                not item.equivalent for item in self.replay_results
            ),
        }

    def _record_tool_call(self, call: dict[str, Any]) -> None:
        """Record one physical branch call exactly once across all runtimes."""
        with self._tool_calls_lock:
            self._tool_calls.append(call)

    def tool_statistics(self) -> dict[str, Any]:
        """Return report-compatible statistics for all physical branch calls."""
        with self._tool_calls_lock:
            calls = [dict(call) for call in self._tool_calls]
        successful = sum(call.get("status") == "success" for call in calls)
        error_statuses = ("invalid_tool", "invalid_args", "execution_error")
        return {
            "total_calls": len(calls),
            "successful_calls": successful,
            "failed_calls": len(calls) - successful,
            "tools_used": sorted(
                {str(call["tool_name"]) for call in calls if call.get("tool_name")}
            ),
            "error_types": {
                status: sum(call.get("status") == status for call in calls)
                for status in error_statuses
            },
            "tool_calls": calls,
        }

    def visual_artifacts(
        self,
        branch: BranchRuntime,
        *,
        limit: int,
        max_bytes: int,
    ) -> list[str]:
        """Return safe local plot/image artifacts produced in a branch.

        Remote trial workspaces are intentionally tolerated: if the advertised
        path is not mounted in this process there is simply no local artifact
        to attach to the evaluator.
        """
        if limit <= 0 or not branch.workspace:
            return []
        workspace = Path(branch.workspace).resolve()
        if not workspace.is_dir():
            return []
        suffixes = {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".tif",
            ".tiff",
            ".bmp",
        }
        artifacts: list[str] = []
        for item in sorted(workspace.rglob("*")):
            if len(artifacts) >= limit:
                break
            if item.is_symlink() or not item.is_file():
                continue
            if item.suffix.casefold() not in suffixes:
                continue
            resolved = item.resolve()
            if workspace not in resolved.parents:
                continue
            try:
                if resolved.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            artifacts.append(str(resolved))
        return artifacts

    def promote_artifacts(
        self,
        nodes: list[ExperimentNode],
        tree: ExperimentTree,
        *,
        destination_workspace: str | None,
        destination_trial_runtime_id: str | None = None,
    ) -> ArtifactPromotion | None:
        """Promote the highest-ranked physical branch into the scored trial.

        Corral's server-side promotion API is preferred because client and
        server filesystems need not be the same. A local copy is retained as a
        compatibility path for in-process/custom routers.
        """
        if not destination_workspace:
            return None
        branch = self._first_physical_branch(nodes, tree)
        if branch is None or not branch.workspace or not branch.trial_runtime_id:
            return None

        promote = getattr(self.interface, "promote_trial_artifacts", None)
        if callable(promote) and destination_trial_runtime_id:
            response = promote(
                branch.trial_runtime_id,
                destination_trial_runtime_id,
            )
            files = tuple(str(item) for item in response.get("files", []))
        else:
            files = tuple(self._copy_workspace(branch.workspace, destination_workspace))
        return ArtifactPromotion(
            branch_id=branch.branch_id,
            source_workspace=branch.workspace,
            destination_workspace=destination_workspace,
            files=files,
        )

    def _first_physical_branch(
        self,
        nodes: list[ExperimentNode],
        tree: ExperimentTree,
    ) -> BranchRuntime | None:
        for candidate in nodes:
            node = candidate
            while True:
                if node.branch_id is not None:
                    branch = self.active.get(node.branch_id)
                    if (
                        branch is not None
                        and branch.workspace
                        and branch.trial_runtime_id
                    ):
                        return branch
                if node.parent_id is None:
                    break
                node = tree.get(node.parent_id)
        return None

    @staticmethod
    def _copy_workspace(source: str, destination: str) -> list[str]:
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if not source_path.is_dir():
            return []
        if source_path == destination_path:
            return []
        if (
            source_path in destination_path.parents
            or destination_path in source_path.parents
        ):
            raise ValueError("Artifact workspaces must not contain one another")

        destination_path.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for item in source_path.rglob("*"):
            # Never let a branch-created symlink escape either workspace.
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

    def _fork(self, parent: ExperimentNode, tree: ExperimentTree) -> BranchRuntime:
        trajectory = tree.executed_trajectory(parent.id)
        if len(trajectory) > self.remaining_calls:
            raise ToolCallBudgetExceeded(
                "Insufficient physical tool-call budget to replay parent trajectory."
            )

        branch = self.create()
        branch.inheritance_method = "replay"
        try:
            for original in trajectory:
                before = len(branch.action_history)
                observations = branch.executor.execute_plan(
                    [original.action], replay=True
                )
                if not observations or len(branch.action_history) == before:
                    raise ReplayDiverged("Replay produced no observation")
                replayed = branch.action_history[before]
                exact = original.observation.model_dump(
                    exclude={"action_index"}
                ) == replayed.observation.model_dump(exclude={"action_index"})
                equivalent = exact
                if (
                    not exact
                    and original.observation.success == replayed.observation.success
                    and self.replay_equivalence is not None
                ):
                    try:
                        equivalent = bool(
                            self.replay_equivalence(
                                original.observation,
                                replayed.observation,
                            )
                        )
                    except Exception as exc:
                        raise ReplayDiverged(
                            "Replay equivalence comparison failed."
                        ) from exc
                result = ReplayResult(
                    branch_id=branch.branch_id,
                    exact=exact,
                    equivalent=equivalent,
                    original=original,
                    replayed=replayed,
                )
                branch.replay_results.append(result)
                self.replay_results.append(result)
                if original.observation.success != replayed.observation.success:
                    raise ReplayDiverged(
                        "Replay changed the success status of a parent action."
                    )
                if not equivalent:
                    raise ReplayDiverged(
                        "Replay observation differed from the parent trajectory; "
                        "the forked physical state is unsafe to use."
                    )
            branch.head_node_id = parent.id
            return branch
        except Exception:
            try:
                self.close(branch.branch_id)
            except Exception as close_error:  # pragma: no cover - transport cleanup
                logger.warning(
                    "Failed to close divergent AI Scientist branch {}: {}",
                    branch.branch_id,
                    close_error,
                )
            raise

    def _clone(
        self,
        parent: ExperimentNode,
        tree: ExperimentTree,
    ) -> BranchRuntime:
        """Clone a parent runtime without replaying its physical trajectory.

        Optional routers expose ``clone_trial(source_trial_runtime_id, task_id)``
        and return the same descriptor shape as ``create_trial``. The logical
        action history is retained for later provenance and fallback replay,
        but cloned actions do not consume the physical tool-call budget.
        """
        source = next(
            (
                branch
                for branch in self.active.values()
                if branch.head_node_id == parent.id
                and branch.trial_runtime_id is not None
            ),
            None,
        )
        source_id = source.trial_runtime_id if source is not None else None
        if source_id is None:
            return self._fork(parent, tree)
        descriptor: dict[str, Any] = self.interface.clone_trial(
            source_id,
            self.task_id,
        )
        self.trials_cloned += 1
        return self._runtime_from_descriptor(
            descriptor,
            action_history=tree.executed_trajectory(parent.id),
            head_node_id=parent.id,
            inheritance_method="clone",
        )
