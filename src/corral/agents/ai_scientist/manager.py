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
from corral.agents.ai_scientist.tools.corral_executor import CorralExecutor
from corral.agents.ai_scientist.workers.base import StructuredModel
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner


class ExperimentManager:
    """Own the stage, tree, journal, selection policy, and local budgets."""

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

    def run(self, state: ScientistState, executor: CorralExecutor) -> ScientistState:
        self._transition(state, ResearchStage.PRELIMINARY)
        self._preliminary(state, executor)

        if (
            state.formulation.has_tunable_parameters
            and self.config.tuning_node_budget > 0
            and self._can_create(state, executor, llm_calls=2)
        ):
            self._transition(state, ResearchStage.TUNING)
            self._iterative_stage(
                state,
                executor,
                stage=ResearchStage.TUNING,
                budget=self.config.tuning_node_budget,
                default_type=NodeType.PARAMETER_SEARCH,
            )

        self._transition(state, ResearchStage.RESEARCH)
        self._iterative_stage(
            state,
            executor,
            stage=ResearchStage.RESEARCH,
            budget=self.config.research_node_budget,
            default_type=NodeType.RESEARCH,
        )

        self._transition(state, ResearchStage.VERIFICATION)
        self._verification(state, executor)
        self._transition(state, ResearchStage.COMPLETE)
        state.tool_calls = executor.call_count
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return state

    def _preliminary(self, state: ScientistState, executor: CorralExecutor) -> None:
        # Independent roots may be planned/evaluated concurrently. Fit all those
        # logical calls while reserving final synthesis for the top-level agent.
        root_limit = min(
            self.config.initial_drafts,
            self.config.preliminary_node_budget,
            self.config.max_nodes - len(state.tree),
            executor.remaining_calls,
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
        self._execute_candidates(state, executor, proposals, parent=None)

        while (
            len(state.tree.by_stage(ResearchStage.PRELIMINARY))
            < self.config.preliminary_node_budget
            and not self._stage_complete(state, ResearchStage.PRELIMINARY)
            and self._can_create(state, executor, llm_calls=2)
        ):
            parent = self.selector.select(state.tree)
            if parent is None:
                break
            node_type = self._continuation_type(parent, NodeType.REFINE)
            count = self._affordable_candidate_count(
                min(
                    self.config.candidates_per_expansion,
                    self.config.preliminary_node_budget
                    - len(state.tree.by_stage(ResearchStage.PRELIMINARY)),
                    self.config.max_nodes - len(state.tree),
                    executor.remaining_calls,
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
            self._execute_candidates(state, executor, proposals, parent=parent)

    def _iterative_stage(
        self,
        state: ScientistState,
        executor: CorralExecutor,
        *,
        stage: ResearchStage,
        budget: int,
        default_type: NodeType,
    ) -> None:
        while (
            len(state.tree.by_stage(stage)) < budget
            and not self._stage_complete(state, stage)
            and self._can_create(state, executor, llm_calls=2)
        ):
            parent = self.selector.select(state.tree)
            if parent is None:
                break
            node_type = self._continuation_type(parent, default_type)
            count = self._affordable_candidate_count(
                min(
                    self.config.candidates_per_expansion,
                    budget - len(state.tree.by_stage(stage)),
                    self.config.max_nodes - len(state.tree),
                    executor.remaining_calls,
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
            self._execute_candidates(state, executor, proposals, parent=parent)

    def _verification(self, state: ScientistState, executor: CorralExecutor) -> None:
        verification_types = cycle(
            [
                NodeType.ABLATION,
                NodeType.REPLICATION,
                NodeType.COUNTERFACTUAL,
                NodeType.AGGREGATION,
            ]
        )
        while len(
            state.tree.by_stage(ResearchStage.VERIFICATION)
        ) < self.config.verification_node_budget and self._can_create(
            state, executor, llm_calls=2
        ):
            count = len(state.tree.by_stage(ResearchStage.VERIFICATION))
            if count >= self.config.verification_min_nodes and self._stage_complete(
                state, ResearchStage.VERIFICATION
            ):
                break
            best = state.tree.best(limit=1)
            parent = best[0] if best else self.selector.select(state.tree)
            if parent is None:
                break
            node_type = next(verification_types)
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
            if node_type == NodeType.AGGREGATION:
                proposal = proposal.model_copy(update={"plan": []})
            self._execute_candidates(
                state, executor, [(node_id, proposal)], parent=parent
            )

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
        branch_workspaces = [f"ai_scientist/{node_id}/" for node_id in candidate_ids]
        journal_context = state.journal.context(self.config.max_journal_chars)

        def plan(candidate_count: int, workspaces: list[str]) -> list[NodeProposal]:
            return self.planner.propose(
                stage=stage,
                node_type=node_type,
                count=candidate_count,
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                tools=state.tools,
                journal_context=journal_context,
                parent=parent,
                branch_workspaces=workspaces,
            )

        if count > 1 and self.config.parallel_llm_workers > 1:

            def propose_one(index: int) -> tuple[str, NodeProposal | None]:
                proposals = plan(1, [branch_workspaces[index]])
                return candidate_ids[index], proposals[0] if proposals else None

            with ThreadPoolExecutor(
                max_workers=min(count, self.config.parallel_llm_workers)
            ) as pool:
                planned = list(pool.map(propose_one, range(count)))
            return [(node_id, proposal) for node_id, proposal in planned if proposal]

        proposals = plan(count, branch_workspaces)
        return list(zip(candidate_ids, proposals, strict=False))

    def _execute_candidates(
        self,
        state: ScientistState,
        executor: CorralExecutor,
        candidates: list[tuple[str, NodeProposal]],
        *,
        parent: ExperimentNode | None,
    ) -> list[ExperimentNode]:
        """Run environment actions serially, then evaluate siblings in parallel."""
        evaluation_capacity = max(
            0,
            self.config.max_llm_calls - 1 - self._llm_calls_used,
        )
        node_capacity = self.config.max_nodes - len(state.tree)
        pending: list[ExperimentNode] = []

        for node_id, proposal in candidates[: min(evaluation_capacity, node_capacity)]:
            if (
                executor.remaining_calls <= 0
                and proposal.node_type != NodeType.AGGREGATION
            ):
                break
            is_debug = proposal.node_type == NodeType.DEBUG
            # Never turn a useful partial plan into a failed node merely because
            # the planner proposed more actions than the remaining local budget.
            plan = proposal.plan[: executor.remaining_calls]
            node = ExperimentNode(
                id=node_id,
                parent_id=parent.id if parent is not None else None,
                branch_workspace=f"ai_scientist/{node_id}/",
                stage=state.current_stage,
                node_type=proposal.node_type,
                hypothesis=proposal.hypothesis,
                rationale=proposal.rationale,
                plan=plan,
                debug_depth=(parent.debug_depth + 1 if parent and is_debug else 0),
                depth=(parent.depth + 1 if parent else 0),
            )
            self.trace.write("node_planned", node.model_dump(mode="json"))
            # Deliberately serial: different branches share Corral task state.
            pending.append(self.experimenter.execute(node, executor))

        if not pending:
            return []

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
            ) as pool:
                evaluations = list(pool.map(evaluate, pending))
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

        state.tool_calls = executor.call_count
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return pending

    def _affordable_candidate_count(self, maximum: int) -> int:
        """Fit planning, evaluation, and the reserved final call in the budget."""
        if maximum <= 0 or not self._within_token_budget:
            return 0
        remaining = self.config.max_llm_calls - 1 - self._llm_calls_used
        for count in range(maximum, 0, -1):
            planning_calls = (
                count if count > 1 and self.config.parallel_llm_workers > 1 else 1
            )
            if planning_calls + count <= remaining:
                return count
        return 0

    def _can_create(
        self,
        state: ScientistState,
        executor: CorralExecutor,
        *,
        llm_calls: int,
    ) -> bool:
        # Keep one LLM call in reserve for final synthesis.
        return (
            len(state.tree) < self.config.max_nodes
            and executor.remaining_calls > 0
            and self._llm_calls_used + llm_calls <= self.config.max_llm_calls - 1
            and self._within_token_budget
        )

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
