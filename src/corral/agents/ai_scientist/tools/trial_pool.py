"""Isolated branch runtimes and replay-on-fork orchestration."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from corral.agents.ai_scientist.search.nodes import ExecutedAction, ExperimentNode
from corral.agents.ai_scientist.search.tree import ExperimentTree
from corral.agents.ai_scientist.tools.corral_executor import (
    CorralExecutor,
    ResearchBudget,
    ToolCallBudgetExceeded,
)


class ReplayDiverged(RuntimeError):
    """Raised when replay changes whether a physical action succeeds."""


@dataclass
class ReplayResult:
    """Comparison between an original physical action and its replay."""

    branch_id: str
    exact: bool
    original: ExecutedAction
    replayed: ExecutedAction


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
    ) -> None:
        self.interface = interface
        self.task_id = task_id
        self.tools = tools
        self.max_observation_chars = max_observation_chars
        self.stop_on_error = stop_on_error
        self.budget = ResearchBudget(max_tool_calls)
        self.active: dict[str, BranchRuntime] = {}
        self.replay_results: list[ReplayResult] = []
        self.trials_created = 0
        self.peak_simultaneous_trials = 0
        self._next_branch_number = 1
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

    def create(self) -> BranchRuntime:
        branch_id = f"branch_{self._next_branch_number:04d}"
        self._next_branch_number += 1

        descriptor: dict[str, Any] = self.interface.create_trial(self.task_id)
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

        history: list[ExecutedAction] = []
        executor = CorralExecutor(
            interface=router,
            task_id=self.task_id,
            tools=self.tools,
            max_tool_calls=self.budget.max_physical_tool_calls,
            max_observation_chars=self.max_observation_chars,
            stop_on_error=self.stop_on_error,
            budget=self.budget,
            on_executed=history.append,
        )
        branch = BranchRuntime(
            branch_id=branch_id,
            trial_runtime_id=trial_runtime_id,
            router=router,
            workspace=workspace,
            executor=executor,
            owns_trial=True,
            action_history=history,
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
    ) -> BranchRuntime:
        """Get a runtime at ``parent``, replaying its trajectory when needed."""
        if parent is None:
            return self.create()
        if prefer_existing and parent.branch_id is not None:
            existing = self.active.get(parent.branch_id)
            if existing is not None and existing.head_node_id == parent.id:
                return existing
        return self._fork(parent, tree)

    def replay_cost(
        self,
        parent: ExperimentNode | None,
        tree: ExperimentTree,
        *,
        prefer_existing: bool,
    ) -> int:
        """Return physical calls needed to place a runtime at ``parent``."""
        if parent is None:
            return 0
        if prefer_existing and parent.branch_id is not None:
            existing = self.active.get(parent.branch_id)
            if existing is not None and existing.head_node_id == parent.id:
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
            "peak_simultaneous_trials": self.peak_simultaneous_trials,
            "replay_divergences": sum(not item.exact for item in self.replay_results),
        }

    def _fork(self, parent: ExperimentNode, tree: ExperimentTree) -> BranchRuntime:
        trajectory = tree.executed_trajectory(parent.id)
        if len(trajectory) > self.remaining_calls:
            raise ToolCallBudgetExceeded(
                "Insufficient physical tool-call budget to replay parent trajectory."
            )

        branch = self.create()
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
                result = ReplayResult(
                    branch_id=branch.branch_id,
                    exact=exact,
                    original=original,
                    replayed=replayed,
                )
                branch.replay_results.append(result)
                self.replay_results.append(result)
                if original.observation.success != replayed.observation.success:
                    raise ReplayDiverged(
                        "Replay changed the success status of a parent action."
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
