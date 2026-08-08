"""Deterministic progressive experiment orchestration."""

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

from loguru import logger

from corral.agents.ai_scientist.config import AIScientistConfig
from corral.agents.ai_scientist.journal import JSONLTraceWriter
from corral.agents.ai_scientist.search.nodes import (
    ExperimentNode,
    NodeEvaluation,
    NodeProposal,
    NodeStatus,
    NodeType,
    Recommendation,
    ResearchStage,
)
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.state import ScientistState
from corral.agents.ai_scientist.tools.corral_executor import ToolCallBudgetExceeded
from corral.agents.ai_scientist.tools.trial_pool import (
    BranchRuntime,
    ReplayDiverged,
    TrialPool,
)
from corral.agents.ai_scientist.workers.base import StructuredModel
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner


class ExperimentManager:
    """Own the stage, tree, journal, selection policy, and global budgets."""

    def __init__(
        self,
        *,
        config: AIScientistConfig,
        model: StructuredModel,
        planner: NodePlanner,
        experimenter: Experimenter,
        critic: ScientificCritic,
        selector: TreeSelector,
        trace: JSONLTraceWriter,
        initial_llm_calls: int = 0,
        initial_llm_tokens: int = 0,
    ) -> None:
        self.config = config
        self.model = model
        self.planner = planner
        self.experimenter = experimenter
        self.critic = critic
        self.selector = selector
        self.trace = trace
        self.initial_llm_calls = initial_llm_calls
        self.initial_llm_tokens = initial_llm_tokens

    def run(self, state: ScientistState, pool: TrialPool) -> ScientistState:
        self._transition(state, ResearchStage.PRELIMINARY)
        self._preliminary(state, pool)

        if (
            state.formulation.has_tunable_parameters
            and self.config.tuning_node_budget > 0
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            self._transition(state, ResearchStage.TUNING)
            self._iterative_stage(
                state,
                pool,
                stage=ResearchStage.TUNING,
                budget=self.config.tuning_node_budget,
                default_type=NodeType.PARAMETER_SEARCH,
            )

        self._transition(state, ResearchStage.RESEARCH)
        self._iterative_stage(
            state,
            pool,
            stage=ResearchStage.RESEARCH,
            budget=self.config.research_node_budget,
            default_type=NodeType.RESEARCH,
        )

        self._transition(state, ResearchStage.VERIFICATION)
        self._verification(state, pool)
        self._transition(state, ResearchStage.COMPLETE)
        self._update_usage(state, pool)
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return state

    def _preliminary(self, state: ScientistState, pool: TrialPool) -> None:
        # Independent roots may be planned/evaluated concurrently. Fit all those
        # logical calls while reserving final synthesis for the top-level agent.
        root_limit = min(
            self.config.initial_drafts,
            self.config.preliminary_node_budget,
            self.config.max_nodes - len(state.tree),
            pool.remaining_calls,
        )
        root_count = self._affordable_candidate_count(root_limit)
        if root_count <= 0:
            return
        proposals = self._propose(
            state,
            stage=ResearchStage.PRELIMINARY,
            node_type=NodeType.DRAFT,
            count=root_count,
            parent=None,
        )
        self._execute_candidates(state, pool, proposals, parent=None)

        while (
            len(state.tree.by_stage(ResearchStage.PRELIMINARY))
            < self.config.preliminary_node_budget
            and not self._stage_complete(state, ResearchStage.PRELIMINARY)
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            parent = self.selector.select(state.tree)
            if parent is None:
                break
            node_type = self._continuation_type(parent, NodeType.REFINE)
            count = self._affordable_candidate_count(
                min(
                    1
                    if node_type == NodeType.CONTINUE
                    else self.config.candidates_per_expansion,
                    self.config.candidates_per_expansion,
                    self.selector.remaining_child_slots(state.tree, parent),
                    self.config.preliminary_node_budget
                    - len(state.tree.by_stage(ResearchStage.PRELIMINARY)),
                    self.config.max_nodes - len(state.tree),
                    pool.remaining_calls,
                )
            )
            if count <= 0:
                break
            proposals = self._propose(
                state,
                stage=ResearchStage.PRELIMINARY,
                node_type=node_type,
                count=count,
                parent=parent,
            )
            if not proposals:
                break
            self._execute_candidates(state, pool, proposals, parent=parent)

    def _iterative_stage(
        self,
        state: ScientistState,
        pool: TrialPool,
        *,
        stage: ResearchStage,
        budget: int,
        default_type: NodeType,
    ) -> None:
        while (
            len(state.tree.by_stage(stage)) < budget
            and not self._stage_complete(state, stage)
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            parent = self.selector.select(state.tree)
            if parent is None:
                break
            node_type = self._continuation_type(parent, default_type)
            count = self._affordable_candidate_count(
                min(
                    1
                    if node_type == NodeType.CONTINUE
                    else self.config.candidates_per_expansion,
                    self.config.candidates_per_expansion,
                    self.selector.remaining_child_slots(state.tree, parent),
                    budget - len(state.tree.by_stage(stage)),
                    self.config.max_nodes - len(state.tree),
                    pool.remaining_calls,
                )
            )
            if count <= 0:
                break
            proposals = self._propose(
                state,
                stage=stage,
                node_type=node_type,
                count=count,
                parent=parent,
            )
            if not proposals:
                break
            self._execute_candidates(state, pool, proposals, parent=parent)

    def _verification(self, state: ScientistState, pool: TrialPool) -> None:
        verification_types = cycle(
            [
                NodeType.ABLATION,
                NodeType.REPLICATION,
                NodeType.COUNTERFACTUAL,
                NodeType.AGGREGATION,
            ]
        )
        while (
            len(state.tree.by_stage(ResearchStage.VERIFICATION))
            < self.config.verification_node_budget
        ):
            count = len(state.tree.by_stage(ResearchStage.VERIFICATION))
            if count >= self.config.verification_min_nodes and self._stage_complete(
                state, ResearchStage.VERIFICATION
            ):
                break
            node_type = (
                NodeType.AGGREGATION
                if pool.remaining_calls <= 0
                else next(verification_types)
            )
            if node_type == NodeType.AGGREGATION and any(
                node.node_type == NodeType.AGGREGATION
                for node in state.tree.by_stage(ResearchStage.VERIFICATION)
            ):
                break
            if not self._can_create(
                state,
                pool,
                llm_calls=2
                if node_type == NodeType.AGGREGATION
                else self._calls_per_node,
                requires_tool_budget=node_type != NodeType.AGGREGATION,
            ):
                break
            parent = self.selector.select(
                state.tree,
                allow_failures=False,
                allow_partial=False,
            )
            if parent is None:
                break
            proposals = self._propose(
                state,
                stage=ResearchStage.VERIFICATION,
                node_type=node_type,
                count=1,
                parent=parent,
            )
            if not proposals:
                break
            node_id, proposal = proposals[0]
            self._execute_candidates(state, pool, [(node_id, proposal)], parent=parent)

    def _propose(
        self,
        state: ScientistState,
        *,
        stage: ResearchStage,
        node_type: NodeType,
        count: int,
        parent: ExperimentNode | None,
    ) -> list[tuple[str, NodeProposal]]:
        candidate_ids = [state.tree.next_id() for _ in range(count)]
        # Every proposal will execute in a separate trial workspace. Relative
        # paths therefore isolate artifacts without synthetic node directories.
        branch_workspaces = ["." for _ in candidate_ids]
        journal_context = state.journal.context(self.config.max_journal_chars)
        siblings = state.tree.children(parent.id if parent is not None else None)

        def plan(
            candidate_count: int,
            workspaces: list[str],
            *,
            proposal_offset: int | None = None,
        ) -> list[NodeProposal]:
            return self.planner.propose(
                stage=stage,
                node_type=node_type,
                count=candidate_count,
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                tools=state.tools,
                journal_context=journal_context,
                parent=parent,
                siblings=siblings,
                branch_workspaces=workspaces,
                proposal_offset=(
                    len(siblings) if proposal_offset is None else proposal_offset
                ),
            )

        if count > 1 and self.config.parallel_llm_workers > 1:

            def propose_one(index: int) -> tuple[str, NodeProposal | None]:
                proposals = plan(
                    1,
                    [branch_workspaces[index]],
                    proposal_offset=len(siblings) + index,
                )
                return candidate_ids[index], proposals[0] if proposals else None

            with ThreadPoolExecutor(
                max_workers=min(count, self.config.parallel_llm_workers)
            ) as pool:
                planned = list(pool.map(propose_one, range(count)))
            candidates = [
                (node_id, proposal) for node_id, proposal in planned if proposal
            ]
            return self._deduplicate_proposals(candidates, siblings)

        proposals = plan(count, branch_workspaces)
        return self._deduplicate_proposals(
            list(zip(candidate_ids, proposals, strict=False)), siblings
        )

    def _deduplicate_proposals(
        self,
        candidates: list[tuple[str, NodeProposal]],
        siblings: list[ExperimentNode],
    ) -> list[tuple[str, NodeProposal]]:
        """Reject exact semantic repeats of existing or simultaneous siblings."""
        fingerprints = {self._experiment_fingerprint(node) for node in siblings}
        unique: list[tuple[str, NodeProposal]] = []
        for node_id, proposal in candidates:
            fingerprint = self._experiment_fingerprint(proposal)
            if fingerprint in fingerprints:
                self.trace.write(
                    "proposal_duplicate_rejected",
                    {"node_id": node_id, "fingerprint": fingerprint},
                )
                continue
            fingerprints.add(fingerprint)
            unique.append((node_id, proposal))
        return unique

    @staticmethod
    def _experiment_fingerprint(
        experiment: ExperimentNode | NodeProposal,
    ) -> str:
        fields = [
            experiment.node_type.value,
            experiment.hypothesis,
            experiment.experiment_goal,
            *experiment.success_criteria,
        ]
        return "|".join(" ".join(item.casefold().split()) for item in fields)

    def _execute_candidates(
        self,
        state: ScientistState,
        pool: TrialPool,
        candidates: list[tuple[str, NodeProposal]],
        *,
        parent: ExperimentNode | None,
    ) -> list[ExperimentNode]:
        """Run isolated sibling experiments and evaluations concurrently."""
        node_capacity = self.config.max_nodes - len(state.tree)
        prepared: list[tuple[ExperimentNode, BranchRuntime | None, int, int]] = []
        reserved_scientific_calls = 0
        reserved_llm_calls = 0
        remaining_llm_calls = max(
            0, self.config.max_llm_calls - 1 - self._llm_calls_used
        )
        candidate_batch = candidates[:node_capacity]
        replay_costs = [
            (
                0
                if proposal.node_type == NodeType.AGGREGATION
                else pool.replay_cost(parent, state.tree, prefer_existing=index == 0)
            )
            for index, (_, proposal) in enumerate(candidate_batch)
        ]

        for index, (node_id, proposal) in enumerate(candidate_batch):
            is_aggregation = proposal.node_type == NodeType.AGGREGATION
            available = pool.remaining_calls - reserved_scientific_calls
            replay_cost = replay_costs[index]
            future_tool_minimum = sum(
                future_replay
                + (0 if future_proposal.node_type == NodeType.AGGREGATION else 1)
                for future_replay, (_, future_proposal) in zip(
                    replay_costs[index + 1 :],
                    candidate_batch[index + 1 :],
                    strict=True,
                )
            )
            if replay_cost + (0 if is_aggregation else 1) > available:
                break
            own_tool_minimum = 0 if is_aggregation else 1
            reserve_for_future_tools = (
                future_tool_minimum
                if replay_cost + own_tool_minimum + future_tool_minimum <= available
                else 0
            )
            available_after_replay = max(
                0,
                available - replay_cost - reserve_for_future_tools,
            )
            future_llm_minimum = sum(
                1 if future.node_type == NodeType.AGGREGATION else 3
                for _, future in candidate_batch[index + 1 :]
            )
            available_llm = (
                remaining_llm_calls - reserved_llm_calls - future_llm_minimum
            )
            action_limit = (
                0
                if is_aggregation
                else min(
                    self.config.max_actions_per_node,
                    available_after_replay,
                    max(0, available_llm - 2),
                )
            )
            if not is_aggregation and action_limit <= 0:
                break
            # Reserve an evaluator, one decision per possible physical action,
            # and a final zero-action completion decision. That last decision
            # prevents a successful action prefix from masquerading as a
            # complete experiment when its allocated budget is exhausted.
            candidate_llm_calls = 1 if is_aggregation else action_limit + 2
            if reserved_llm_calls + candidate_llm_calls > remaining_llm_calls:
                break
            branch: BranchRuntime | None = None
            if not is_aggregation:
                try:
                    branch = pool.acquire(
                        parent,
                        state.tree,
                        prefer_existing=index == 0,
                    )
                except (ReplayDiverged, ToolCallBudgetExceeded) as exc:
                    logger.warning("Could not prepare {}: {}", node_id, exc)
                    self.trace.write(
                        "branch_replay_failed",
                        {"node_id": node_id, "error": str(exc)},
                    )
                    break
            is_debug = proposal.node_type == NodeType.DEBUG
            reserved_llm_calls += candidate_llm_calls
            reserved_scientific_calls += action_limit
            node = ExperimentNode(
                id=node_id,
                parent_id=parent.id if parent is not None else None,
                branch_id=branch.branch_id if branch is not None else None,
                trial_runtime_id=(
                    branch.trial_runtime_id if branch is not None else None
                ),
                branch_workspace=branch.workspace if branch is not None else None,
                stage=state.current_stage,
                node_type=proposal.node_type,
                hypothesis=proposal.hypothesis,
                rationale=proposal.rationale,
                experiment_goal=proposal.experiment_goal,
                success_criteria=proposal.success_criteria,
                debug_depth=(parent.debug_depth + 1 if parent and is_debug else 0),
                depth=(parent.depth + 1 if parent else 0),
            )
            self.trace.write("node_planned", node.model_dump(mode="json"))
            if branch is not None:
                self.trace.write(
                    "branch_assigned",
                    {
                        "node_id": node.id,
                        "branch_id": branch.branch_id,
                        "trial_runtime_id": branch.trial_runtime_id,
                        "replayed_actions": len(branch.replay_results),
                    },
                )
                for replay in branch.replay_results:
                    self.trace.write(
                        "action_replayed",
                        {
                            "node_id": node.id,
                            "branch_id": branch.branch_id,
                            "exact": replay.exact,
                            "equivalent": replay.equivalent,
                            "original": replay.original.model_dump(mode="json"),
                            "replayed": replay.replayed.model_dump(mode="json"),
                        },
                    )
            else:
                self.trace.write(
                    "aggregation_without_branch",
                    {"node_id": node.id, "parent_id": node.parent_id},
                )
            prepared.append(
                (
                    node,
                    branch,
                    len(branch.action_history) if branch is not None else 0,
                    action_limit,
                )
            )

        if not prepared:
            return []

        def execute(
            item: tuple[ExperimentNode, BranchRuntime | None, int, int],
        ) -> ExperimentNode:
            node, branch, history_start, action_limit = item
            executed = self.experimenter.execute(
                node,
                branch.executor if branch is not None else None,
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                tools=state.tools,
                journal_context=state.journal.context(self.config.max_journal_chars),
                action_limit=action_limit,
                previous_node=(parent if node.node_type == NodeType.CONTINUE else None),
            )
            if branch is not None:
                pool.commit(branch, executed, history_start)
            return executed

        experiment_workers = (
            min(len(prepared), self.config.parallel_experiment_workers)
            if pool.isolated
            else 1
        )
        if experiment_workers > 1:
            with ThreadPoolExecutor(max_workers=experiment_workers) as executor:
                pending = list(executor.map(execute, prepared))
        else:
            pending = [execute(item) for item in prepared]

        journal_context = state.journal.context(self.config.max_journal_chars)

        def evaluate(node: ExperimentNode) -> NodeEvaluation:
            return self.critic.evaluate(
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                node=node,
                journal_context=journal_context,
            )

        if len(pending) > 1 and self.config.parallel_llm_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(pending), self.config.parallel_llm_workers)
            ) as llm_pool:
                evaluations = list(llm_pool.map(evaluate, pending))
        else:
            evaluations = [evaluate(node) for node in pending]

        for node, evaluation in zip(pending, evaluations, strict=True):
            node.evaluation = evaluation
            node.conclusions = evaluation.conclusions
            node.open_questions = evaluation.open_questions
            if (
                node.status == NodeStatus.SUCCESSFUL
                and evaluation.validity < self.config.minimum_validity
            ):
                node.status = NodeStatus.INVALID
            state.tree.add(node)
            state.journal.integrate(node, self.config.minimum_validity)
            self.trace.write("node_completed", node.model_dump(mode="json"))
            self.trace.write("journal", state.journal.as_dict())

        self._update_usage(state, pool)
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return pending

    def _affordable_candidate_count(self, maximum: int) -> int:
        """Fit planning, adaptive execution, completion, and evaluation."""
        if maximum <= 0 or not self._within_token_budget:
            return 0
        remaining = self.config.max_llm_calls - 1 - self._llm_calls_used
        for count in range(maximum, 0, -1):
            planning_calls = (
                count if count > 1 and self.config.parallel_llm_workers > 1 else 1
            )
            if planning_calls + count * 3 <= remaining:
                return count
        return 0

    def _can_create(
        self,
        state: ScientistState,
        pool: TrialPool,
        *,
        llm_calls: int,
        requires_tool_budget: bool = True,
    ) -> bool:
        # Keep one LLM call in reserve for final synthesis.
        return (
            len(state.tree) < self.config.max_nodes
            and (not requires_tool_budget or pool.remaining_calls > 0)
            and self._llm_calls_used + llm_calls <= self.config.max_llm_calls - 1
            and self._within_token_budget
        )

    def _update_usage(self, state: ScientistState, pool: TrialPool) -> None:
        state.tool_calls = pool.call_count
        state.scientific_tool_calls = pool.budget.scientific_calls
        state.replay_tool_calls = pool.budget.replay_calls
        state.trial_runtimes_created = pool.trials_created
        state.peak_simultaneous_trials = pool.peak_simultaneous_trials
        state.replay_results = list(pool.replay_results)

    @property
    def _calls_per_node(self) -> int:
        """Minimum: planner, action, explicit finish, and evaluator."""
        return 4

    @property
    def _llm_calls_used(self) -> int:
        """Return calls used by this run, even for a reusable injected model."""
        return self.model.call_count - self.initial_llm_calls

    @property
    def _llm_tokens_used(self) -> int:
        """Return provider-reported tokens used by this run."""
        return getattr(self.model, "token_count", 0) - self.initial_llm_tokens

    @property
    def _within_token_budget(self) -> bool:
        limit = self.config.max_llm_tokens
        return limit is None or self._llm_tokens_used < limit

    def _continuation_type(
        self, parent: ExperimentNode, successful_type: NodeType
    ) -> NodeType:
        if parent.status == NodeStatus.PARTIAL:
            return NodeType.CONTINUE
        if parent.status in {NodeStatus.FAILED, NodeStatus.INVALID}:
            if parent.debug_depth < self.config.max_debug_depth:
                return NodeType.DEBUG
            return successful_type
        if parent.evaluation is not None:
            if (
                parent.evaluation.recommendation == Recommendation.DEBUG
                and parent.debug_depth < self.config.max_debug_depth
            ):
                return NodeType.DEBUG
            if parent.evaluation.recommendation == Recommendation.REFINE:
                return NodeType.REFINE
        return successful_type

    def _stage_complete(self, state: ScientistState, stage: ResearchStage) -> bool:
        evaluated = [
            node.evaluation
            for node in state.tree.by_stage(stage)
            if node.evaluation is not None
            and node.status == NodeStatus.SUCCESSFUL
            and node.evaluation.recommendation != Recommendation.ABANDON
        ]
        if not evaluated:
            return False
        if stage == ResearchStage.PRELIMINARY:
            return any(
                item.validity >= self.config.minimum_validity
                and item.evidence_strength >= self.config.preliminary_evidence_threshold
                for item in evaluated
            )
        if stage in {ResearchStage.TUNING, ResearchStage.RESEARCH}:
            return any(
                item.recommendation == Recommendation.FINALIZE
                and item.task_progress >= self.config.stage_completion_threshold
                and item.evidence_strength >= self.config.preliminary_evidence_threshold
                for item in evaluated
            )
        if stage == ResearchStage.VERIFICATION:
            return any(
                self._confidence(item) >= self.config.confidence_threshold
                for item in evaluated
            )
        return False

    @staticmethod
    def _confidence(evaluation: NodeEvaluation) -> float:
        return (
            evaluation.validity + evaluation.evidence_strength + evaluation.consistency
        ) / 3

    def _transition(self, state: ScientistState, stage: ResearchStage) -> None:
        state.current_stage = stage
        logger.info("AI Scientist entering {} stage", stage.value)
        self.trace.write("stage_transition", {"stage": stage.value})
