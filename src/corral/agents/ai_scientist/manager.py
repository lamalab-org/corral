"""Deterministic progressive experiment orchestration."""

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from math import sqrt
from statistics import fmean, stdev

from loguru import logger

from corral.agents.ai_scientist.config import AIScientistConfig
from corral.agents.ai_scientist.journal import JSONLTraceWriter
from corral.agents.ai_scientist.search.evaluator import (
    evaluation_priority,
    extract_measured_outcome,
    measured_improvement,
    node_ranking_key,
)
from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    ExperimentNode,
    ExperimentStep,
    ExperimentTermination,
    NodeEvaluation,
    NodeProposal,
    NodeStatus,
    NodeType,
    PlannedAction,
    Recommendation,
    ReplicationSummary,
    ResearchStage,
    SubstagePlan,
)
from corral.agents.ai_scientist.search.selector import SuccessfulRanker, TreeSelector
from corral.agents.ai_scientist.state import (
    ScientistState,
    StageProgress,
    SubstageState,
)
from corral.agents.ai_scientist.tools.corral_executor import ToolCallBudgetExceeded
from corral.agents.ai_scientist.tools.execution_pool import (
    BranchExecution,
    ExecutionPool,
    ReplayDiverged,
)
from corral.agents.ai_scientist.workers.base import StructuredModel
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner


class ExperimentManager:
    """Own the stage, tree, journal, selection policy, and global budgets.

    This is an implementation component of :class:`AIScientistAgent`, not a
    Corral agent entry point. The agent-facing lifecycle is exclusively
    ``AIScientistAgent.run_session``.
    """

    def __init__(
        self,
        *,
        config: AIScientistConfig,
        max_llm_calls: int,
        model: StructuredModel,
        planner: NodePlanner,
        experimenter: Experimenter,
        critic: ScientificCritic,
        selector: TreeSelector,
        trace: JSONLTraceWriter,
        initial_llm_calls: int = 0,
        initial_llm_tokens: int = 0,
    ) -> None:
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be at least 1")
        self.config = config
        self.max_llm_calls = max_llm_calls
        self.model = model
        self.planner = planner
        self.experimenter = experimenter
        self.critic = critic
        self.selector = selector
        self.trace = trace
        self.initial_llm_calls = initial_llm_calls
        self.initial_llm_tokens = initial_llm_tokens

    def execute_search(
        self, state: ScientistState, pool: ExecutionPool
    ) -> ScientistState:
        """Execute the deterministic experiment-search lifecycle."""
        self._transition(state, ResearchStage.PRELIMINARY)
        self._begin_stage(state, ResearchStage.PRELIMINARY, seed=None)
        self._preliminary(state, pool)
        stage_winner = self._finish_stage(state, pool, ResearchStage.PRELIMINARY)

        preliminary = state.stage_progress(ResearchStage.PRELIMINARY)
        if stage_winner is None or not preliminary.completion_criteria_met:
            state.experimental_search_terminated_reason = (
                "Preliminary investigation did not produce an executable "
                "approach with sufficient valid primary evidence."
            )
            self.trace.write(
                "experimental_search_terminated",
                {
                    "stage": ResearchStage.PRELIMINARY.value,
                    "reason": state.experimental_search_terminated_reason,
                },
            )
            self._transition(state, ResearchStage.COMPLETE)
            self._update_usage(state, pool)
            state.llm_calls = self._llm_calls_used
            state.llm_tokens = self._llm_tokens_used
            return state

        if (
            stage_winner is not None
            and (
                state.formulation.has_tunable_parameters
                or self.config.force_tuning_stage
            )
            and self.config.tuning_node_budget > 0
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            self._transition(state, ResearchStage.TUNING)
            self._begin_stage(state, ResearchStage.TUNING, seed=stage_winner)
            self._iterative_stage(
                state,
                pool,
                stage=ResearchStage.TUNING,
                budget=self.config.tuning_node_budget,
                default_type=NodeType.PARAMETER_SEARCH,
            )
            stage_winner = (
                self._finish_stage(state, pool, ResearchStage.TUNING) or stage_winner
            )

        self._transition(state, ResearchStage.RESEARCH)
        self._begin_stage(state, ResearchStage.RESEARCH, seed=stage_winner)
        if stage_winner is not None:
            self._iterative_stage(
                state,
                pool,
                stage=ResearchStage.RESEARCH,
                budget=self.config.research_node_budget,
                default_type=NodeType.RESEARCH,
            )
        stage_winner = (
            self._finish_stage(state, pool, ResearchStage.RESEARCH) or stage_winner
        )

        self._transition(state, ResearchStage.VERIFICATION)
        self._begin_stage(state, ResearchStage.VERIFICATION, seed=stage_winner)
        if stage_winner is not None:
            self._verification(state, pool)
        self._finish_stage(state, pool, ResearchStage.VERIFICATION)
        self._transition(state, ResearchStage.COMPLETE)
        self._update_usage(state, pool)
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return state

    def _preliminary(self, state: ScientistState, pool: ExecutionPool) -> None:
        # Independent roots may be planned/evaluated concurrently. Fit all those
        # logical calls while reserving final synthesis for the top-level agent.
        root_limit = min(
            self.config.initial_drafts,
            self.config.preliminary_node_budget,
            self._remaining_node_capacity(state),
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
        self._execute_candidates(
            state,
            pool,
            proposals,
            parent=None,
            substage_id=self._current_substage_id(state, ResearchStage.PRELIMINARY),
        )

        while (
            self._search_node_count(state, ResearchStage.PRELIMINARY)
            < self.config.preliminary_node_budget
            and not self._stage_complete(state, ResearchStage.PRELIMINARY)
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            self._maybe_advance_substage(state, ResearchStage.PRELIMINARY)
            if self.config.parallel_parent_selection:
                expanded = self._expand_parallel_parent_batch(
                    state,
                    pool,
                    stage=ResearchStage.PRELIMINARY,
                    budget=self.config.preliminary_node_budget,
                    default_type=NodeType.REFINE,
                )
                if not expanded:
                    break
                continue
            parent = self._select_parent(state, ResearchStage.PRELIMINARY)
            if parent is None:
                break
            node_type = self._continuation_type(parent, NodeType.REFINE)
            count = self._affordable_candidate_count(
                min(
                    1
                    if node_type == NodeType.CONTINUE
                    else self.config.candidates_per_expansion,
                    self.config.candidates_per_expansion,
                    self.selector.remaining_child_slots(
                        state.tree,
                        parent,
                        stage=ResearchStage.PRELIMINARY,
                    ),
                    self.config.preliminary_node_budget
                    - self._search_node_count(state, ResearchStage.PRELIMINARY),
                    self._remaining_node_capacity(state),
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
            self._execute_candidates(
                state,
                pool,
                proposals,
                parent=parent,
                substage_id=self._current_substage_id(state, ResearchStage.PRELIMINARY),
            )

    def _iterative_stage(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        *,
        stage: ResearchStage,
        budget: int,
        default_type: NodeType,
    ) -> None:
        while (
            self._search_node_count(state, stage) < budget
            and (
                (
                    stage == ResearchStage.RESEARCH
                    and not self.config.research_early_stopping
                )
                or not self._stage_complete(state, stage)
            )
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            self._maybe_advance_substage(state, stage)
            if (
                self.config.parallel_parent_selection
                and stage == ResearchStage.RESEARCH
            ):
                expanded = self._expand_parallel_parent_batch(
                    state,
                    pool,
                    stage=stage,
                    budget=budget,
                    default_type=default_type,
                )
                if not expanded:
                    break
                continue
            parent, node_type = self._parent_and_node_type(
                state,
                stage,
                default_type,
            )
            if parent is None:
                break
            fixed_baseline = self._uses_fixed_baseline(stage, parent, node_type, state)
            count = self._affordable_candidate_count(
                min(
                    1
                    if node_type == NodeType.CONTINUE
                    else self.config.candidates_per_expansion,
                    self.config.candidates_per_expansion,
                    (
                        budget - self._search_node_count(state, stage)
                        if fixed_baseline
                        else self.selector.remaining_child_slots(
                            state.tree, parent, stage=stage
                        )
                    ),
                    budget - self._search_node_count(state, stage),
                    self._remaining_node_capacity(state),
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
            self._execute_candidates(
                state,
                pool,
                proposals,
                parent=parent,
                substage_id=self._current_substage_id(state, stage),
            )

    def _verification(self, state: ScientistState, pool: ExecutionPool) -> None:
        verification_types = cycle(
            [NodeType.ABLATION, NodeType.COUNTERFACTUAL]
            if self.config.verification_include_counterfactual
            else [NodeType.ABLATION]
        )
        while (
            self._search_node_count(state, ResearchStage.VERIFICATION)
            < self.config.verification_node_budget
        ):
            count = self._search_node_count(state, ResearchStage.VERIFICATION)
            if (
                self.config.verification_early_stopping
                and count >= self.config.verification_min_nodes
                and self._stage_complete(state, ResearchStage.VERIFICATION)
            ):
                break
            self._maybe_advance_substage(state, ResearchStage.VERIFICATION)
            node_type = next(verification_types)
            if not self._can_create(
                state,
                pool,
                llm_calls=self._calls_per_node,
            ):
                break
            parent, selected_type = self._parent_and_node_type(
                state,
                ResearchStage.VERIFICATION,
                node_type,
            )
            if parent is None:
                break
            node_type = selected_type
            fixed_baseline = self._uses_fixed_baseline(
                ResearchStage.VERIFICATION,
                parent,
                node_type,
                state,
            )
            candidate_count = self._affordable_candidate_count(
                min(
                    self.config.candidates_per_expansion,
                    (
                        self.config.verification_node_budget - count
                        if fixed_baseline
                        else self.selector.remaining_child_slots(
                            state.tree,
                            parent,
                            stage=ResearchStage.VERIFICATION,
                        )
                    ),
                    self.config.verification_node_budget - count,
                    self._remaining_node_capacity(state),
                    pool.remaining_calls,
                )
            )
            if candidate_count <= 0:
                break
            proposals = self._propose(
                state,
                stage=ResearchStage.VERIFICATION,
                node_type=node_type,
                count=candidate_count,
                parent=parent,
            )
            if not proposals:
                break
            self._execute_candidates(
                state,
                pool,
                proposals,
                parent=parent,
                substage_id=self._current_substage_id(
                    state, ResearchStage.VERIFICATION
                ),
            )

    def _begin_stage(
        self,
        state: ScientistState,
        stage: ResearchStage,
        *,
        seed: ExperimentNode | None,
    ) -> None:
        """Create an explicit stage scope seeded by the prior stage winner."""
        goals = {
            ResearchStage.PRELIMINARY: (
                "Find an executable route that produces valid evidence for the task."
            ),
            ResearchStage.TUNING: (
                "Improve the inherited baseline through controlled parameter or "
                "procedure comparisons."
            ),
            ResearchStage.RESEARCH: (
                "Beat the inherited checkpoint with discriminating, falsifying, "
                "or uncertainty-reducing experiments."
            ),
            ResearchStage.VERIFICATION: (
                "Systematically ablate components or assumptions supporting the "
                "inherited conclusion."
            ),
        }
        criteria = {
            ResearchStage.PRELIMINARY: [
                "At least one executable approach yields valid primary evidence."
            ],
            ResearchStage.TUNING: [
                "A controlled experimental change is selected over the inherited seed."
            ],
            ResearchStage.RESEARCH: [
                "A new checkpoint is selected over the inherited seed using stronger evidence."
            ],
            ResearchStage.VERIFICATION: [
                "At least one systematic ablation has strong, valid evidence."
            ],
        }
        plan = SubstagePlan(
            goal=goals[stage],
            rationale="Initial agenda for this main stage.",
            objectives=[goals[stage]],
            completion_criteria=criteria[stage],
        )
        progress = StageProgress(
            stage=stage,
            seed_node_id=seed.id if seed is not None else None,
            substages=[SubstageState(id=f"{stage.value}.1", plan=plan)],
        )
        state.stages[stage] = progress
        self.trace.write(
            "stage_started",
            progress.model_dump(mode="json"),
        )

    def _select_parent(
        self,
        state: ScientistState,
        stage: ResearchStage,
        *,
        allow_failures: bool = True,
        allow_partial: bool = True,
    ) -> ExperimentNode | None:
        progress = state.stage_progress(stage)
        return self.selector.select(
            state.tree,
            allow_failures=allow_failures,
            allow_partial=allow_partial,
            stage=stage,
            seed_node_id=progress.seed_node_id,
            successful_ranker=self._parent_ranker(state, stage),
        )

    def _select_parent_batch(
        self,
        state: ScientistState,
        stage: ResearchStage,
        *,
        limit: int,
    ) -> list[ExperimentNode]:
        progress = state.stage_progress(stage)
        return self.selector.select_batch(
            state.tree,
            limit=limit,
            stage=stage,
            seed_node_id=progress.seed_node_id,
            prefer_distinct_roots=self.config.prefer_distinct_root_trees,
            successful_ranker=self._parent_ranker(state, stage),
        )

    def _parent_ranker(
        self,
        state: ScientistState,
        stage: ResearchStage,
    ) -> SuccessfulRanker | None:
        if self.config.parent_selection_mode != "llm" or stage not in {
            ResearchStage.PRELIMINARY,
            ResearchStage.RESEARCH,
        }:
            return None

        def rank(candidates: list[ExperimentNode]) -> ExperimentNode | None:
            if len(candidates) == 1:
                return candidates[0]
            if (
                self._llm_calls_used + 1 > self.max_llm_calls - 1
                or not self._within_token_budget
            ):
                return None
            selection = self.critic.select_best(
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                stage=stage,
                seed_node_id=state.stage_progress(stage).seed_node_id,
                candidates=candidates,
                journal_context=state.journal.context(self.config.max_journal_chars),
            )
            by_id = {node.id: node for node in candidates}
            selected = by_id.get(selection.selected_node_id)
            self.trace.write(
                "search_parent_selected",
                {
                    "stage": stage.value,
                    "selected_node_id": selection.selected_node_id,
                    "allowed_node_ids": list(by_id),
                    "selection_valid": selected is not None,
                    "reason": selection.reason,
                },
            )
            return selected

        return rank

    def _expand_parallel_parent_batch(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        *,
        stage: ResearchStage,
        budget: int,
        default_type: NodeType,
    ) -> bool:
        """Select one parent per worker slot, then run the children together."""
        maximum = min(
            self.config.parallel_experiment_workers,
            budget - self._search_node_count(state, stage),
            self._remaining_node_capacity(state),
            pool.remaining_calls,
        )
        maximum = self._affordable_candidate_count(
            maximum,
            independent_planning=True,
        )
        if maximum <= 0:
            return False
        parents = self._select_parent_batch(state, stage, limit=maximum)
        if not parents:
            return False
        # LLM parent selection itself consumes calls, so fit the resulting
        # worker batch again before starting independent planning calls.
        affordable = self._affordable_candidate_count(
            len(parents),
            independent_planning=True,
        )
        parents = parents[:affordable]
        if not parents:
            return False
        parent_requests = [
            (parent, self._continuation_type(parent, default_type))
            for parent in parents
        ]
        candidates = self._propose_for_parent_batch(
            state,
            stage=stage,
            parent_requests=parent_requests,
        )
        if not candidates:
            return False
        completed = self._execute_parent_candidates(
            state,
            pool,
            candidates,
            substage_id=self._current_substage_id(state, stage),
        )
        return bool(completed)

    def _parent_and_node_type(
        self,
        state: ScientistState,
        stage: ResearchStage,
        default_type: NodeType,
    ) -> tuple[ExperimentNode | None, NodeType]:
        """Apply stage-specific fixed-baseline semantics.

        Normal tuning experiments always compare against the Stage-1 seed and
        normal verification experiments against the Stage-3 seed. A current-
        stage node may become the parent only for a continuation or debug that
        repairs that exact experiment.
        """
        selected = self._select_parent(state, stage)
        if stage not in {ResearchStage.TUNING, ResearchStage.VERIFICATION}:
            if selected is None:
                return None, default_type
            return selected, self._continuation_type(selected, default_type)

        progress = state.stage_progress(stage)
        seed = (
            state.tree.get(progress.seed_node_id)
            if progress.seed_node_id is not None
            else None
        )
        if selected is not None and selected.id != progress.seed_node_id:
            repair_type = self._continuation_type(selected, default_type)
            if repair_type in {NodeType.CONTINUE, NodeType.DEBUG}:
                return selected, repair_type
        return seed, default_type

    @staticmethod
    def _uses_fixed_baseline(
        stage: ResearchStage,
        parent: ExperimentNode,
        node_type: NodeType,
        state: ScientistState,
    ) -> bool:
        if stage not in {ResearchStage.TUNING, ResearchStage.VERIFICATION}:
            return False
        progress = state.stage_progress(stage)
        return parent.id == progress.seed_node_id and node_type not in {
            NodeType.CONTINUE,
            NodeType.DEBUG,
        }

    def _maybe_advance_substage(
        self, state: ScientistState, stage: ResearchStage
    ) -> None:
        if not self.config.adaptive_substages:
            return
        progress = state.stage_progress(stage)
        current = progress.current_substage
        if current is None or current.completion_criteria_met:
            return
        if len(progress.substages) >= self.config.max_substages_per_stage:
            return
        if (
            len(current.node_ids) - current.last_completion_check_node_count
            < self.config.nodes_per_substage
        ):
            return
        # The new agenda is useful only if at least one experiment can still be
        # run afterwards. Reserve one critic check, one replanning call, and
        # final synthesis as usual.
        if (
            self._llm_calls_used + 2 + self._calls_per_node > self.max_llm_calls - 1
            or not self._within_token_budget
        ):
            return
        current_nodes = [state.tree.get(node_id) for node_id in current.node_ids]
        current.last_completion_check_node_count = len(current.node_ids)
        completion = self.critic.is_substage_complete(
            task_prompt=state.task_prompt,
            formulation=state.formulation,
            stage=stage,
            substage=current.plan,
            stage_nodes=current_nodes,
            journal_context=state.journal.context(self.config.max_journal_chars),
        )
        current.completion_criteria_met = completion.complete
        current.completion_reason = completion.reason
        self.trace.write(
            "substage_completion_checked",
            {
                "stage": stage.value,
                "substage_id": current.id,
                **completion.model_dump(mode="json"),
            },
        )
        if not completion.complete:
            return
        stage_nodes = state.tree.by_stage(stage, include_boundary=False)
        seed = (
            state.tree.get(progress.seed_node_id)
            if progress.seed_node_id is not None
            else None
        )
        number = len(progress.substages) + 1
        plan = self.planner.propose_substage(
            stage=stage,
            task_prompt=state.task_prompt,
            formulation=state.formulation,
            journal_context=state.journal.context(self.config.max_journal_chars),
            seed=seed,
            previous=current.plan,
            stage_nodes=stage_nodes,
            substage_number=number,
        )
        substage = SubstageState(id=f"{stage.value}.{number}", plan=plan)
        progress.substages.append(substage)
        self.trace.write(
            "substage_created",
            {
                "stage": stage.value,
                **substage.model_dump(mode="json"),
            },
        )

    def _finish_stage(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        stage: ResearchStage,
    ) -> ExperimentNode | None:
        """Comparatively select, validate, and replicate a stage winner."""
        progress = state.stage_progress(stage)
        stage_nodes = state.tree.by_stage(stage, include_boundary=False)
        candidates = [
            node for node in stage_nodes if self._eligible_stage_candidate(node)
        ]
        if progress.seed_node_id is not None:
            seed = state.tree.get(progress.seed_node_id)
            if self._eligible_stage_candidate(seed):
                candidates.insert(0, seed)
        winner, reason = self._comparative_winner(state, stage, candidates)
        if winner is None:
            progress.comparison_reason = "No valid stage candidate was available."
            self.trace.write(
                "stage_finished",
                progress.model_dump(mode="json"),
            )
            return None

        progress.best_node_id = winner.id
        progress.improved_over_seed = (
            progress.seed_node_id is None or winner.id != progress.seed_node_id
        )
        progress.comparison_reason = reason
        progress.search_budget_exhausted = self._search_node_count(
            state, stage
        ) >= self._stage_node_budget(stage)
        progress.completion_criteria_met = self._stage_completion_for_winner(
            state, stage, winner
        )
        self.trace.write(
            "stage_winner_selected",
            progress.model_dump(mode="json"),
        )
        validate_budget_exhaustion = (
            self.config.validate_on_stage_budget_exhaustion
            and stage != ResearchStage.PRELIMINARY
            and progress.search_budget_exhausted
        )
        if progress.completion_criteria_met or validate_budget_exhaustion:
            progress.boundary_validation_reason = (
                "completion_criteria_met"
                if progress.completion_criteria_met
                else "search_budget_exhausted"
            )
            self._run_stage_boundary_validation(state, pool, stage, winner)
        else:
            self.trace.write(
                "stage_incomplete",
                {
                    "stage": stage.value,
                    "seed_node_id": progress.seed_node_id,
                    "best_node_id": progress.best_node_id,
                    "reason": (
                        "No later-stage checkpoint beat its inherited seed with "
                        "the required stage-specific evidence."
                    ),
                },
            )
        self.trace.write(
            "stage_finished",
            progress.model_dump(mode="json"),
        )
        return winner

    def _stage_node_budget(self, stage: ResearchStage) -> int:
        return {
            ResearchStage.PRELIMINARY: self.config.preliminary_node_budget,
            ResearchStage.TUNING: self.config.tuning_node_budget,
            ResearchStage.RESEARCH: self.config.research_node_budget,
            ResearchStage.VERIFICATION: self.config.verification_node_budget,
        }[stage]

    def _comparative_winner(
        self,
        state: ScientistState,
        stage: ResearchStage,
        candidates: list[ExperimentNode],
    ) -> tuple[ExperimentNode | None, str]:
        if not candidates:
            return None, "No valid candidates."
        fallback = max(candidates, key=node_ranking_key)
        if len(candidates) < 2:
            return fallback, "Only one valid candidate was available."
        if (
            self._llm_calls_used + 1 > self.max_llm_calls - 1
            or not self._within_token_budget
        ):
            return fallback, "Comparative selection skipped at the LLM budget limit."
        selection = self.critic.select_best(
            task_prompt=state.task_prompt,
            formulation=state.formulation,
            stage=stage,
            seed_node_id=state.stage_progress(stage).seed_node_id,
            candidates=candidates,
            journal_context=state.journal.context(self.config.max_journal_chars),
        )
        by_id = {node.id: node for node in candidates}
        selected = by_id.get(selection.selected_node_id)
        if selected is None:
            self.trace.write(
                "invalid_stage_winner_rejected",
                {
                    "stage": stage.value,
                    "selected_node_id": selection.selected_node_id,
                    "allowed_node_ids": list(by_id),
                },
            )
            return fallback, (
                "The comparative evaluator returned an unknown id; used the "
                "deterministic critic-priority fallback."
            )
        return selected, selection.reason

    def _stage_completion_for_winner(
        self,
        state: ScientistState,
        stage: ResearchStage,
        winner: ExperimentNode,
    ) -> bool:
        evaluation = winner.evaluation
        if evaluation is None:
            return False
        if stage == ResearchStage.PRELIMINARY:
            return (
                (
                    not self.config.preliminary_require_critic_validity
                    or evaluation.validity >= self.config.minimum_validity
                )
                and evaluation.evidence_strength
                >= self.config.preliminary_evidence_threshold
            )
        if stage in {ResearchStage.TUNING, ResearchStage.RESEARCH}:
            progress = state.stage_progress(stage)
            if winner.id == progress.seed_node_id:
                return False
            if (
                evaluation.recommendation != Recommendation.FINALIZE
                or evaluation.task_progress < self.config.stage_completion_threshold
                or evaluation.evidence_strength
                < self.config.preliminary_evidence_threshold
            ):
                return False
            return self._beats_seed(state, stage, winner, inclusive=True)
        if stage == ResearchStage.VERIFICATION:
            return any(
                node.node_type == NodeType.ABLATION
                and node.status == NodeStatus.SUCCESSFUL
                and node.evaluation is not None
                and node.evaluation.validity >= self.config.minimum_validity
                and self._confidence(node.evaluation)
                >= self.config.confidence_threshold
                for node in state.tree.by_stage(stage, include_boundary=False)
            )
        return False

    def _run_stage_boundary_validation(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        stage: ResearchStage,
        winner: ExperimentNode,
    ) -> None:
        progress = state.stage_progress(stage)
        if not self._within_token_budget:
            return
        reserve_aggregation = int(
            self.config.aggregate_stage_replications
            and self.config.stage_boundary_replications > 0
        )
        if self.config.deterministic_replication:
            replications = self._run_deterministic_replications(
                state,
                pool,
                stage,
                winner,
                reserve_aggregation=reserve_aggregation,
            )
        else:
            replication_capacity = max(
                0,
                self._remaining_node_capacity(state, boundary_validation=True)
                - reserve_aggregation,
            )
            maximum = min(
                self.config.stage_boundary_replications,
                replication_capacity,
                pool.remaining_calls,
            )
            remaining_llm = self.max_llm_calls - 1 - self._llm_calls_used
            maximum = min(maximum, max(0, remaining_llm // 3))
            candidates = [
                (
                    state.tree.next_id(),
                    NodeProposal(
                        node_type=NodeType.REPLICATION,
                        hypothesis=winner.hypothesis,
                        rationale=(
                            "Independent stage-boundary repetition of the selected "
                            f"{stage.value} checkpoint."
                        ),
                        experiment_goal=winner.experiment_goal,
                        success_criteria=list(winner.success_criteria),
                        related_node_ids=[winner.id],
                    ),
                )
                for _ in range(maximum)
            ]
            replications = self._execute_candidates(
                state,
                pool,
                candidates,
                parent=winner,
                boundary_validation=True,
                inherit_parent_state=False,
            )
        progress.replication_node_ids = [node.id for node in replications]

        if (
            not replications
            or not self.config.aggregate_stage_replications
            or not self._can_create(
                state,
                pool,
                llm_calls=1,
                requires_tool_budget=False,
                boundary_validation=True,
            )
        ):
            return
        aggregation_id = state.tree.next_id()
        summary = self._summarize_replications(replications)
        aggregation = NodeProposal(
            node_type=NodeType.AGGREGATION,
            hypothesis=winner.hypothesis,
            rationale=(
                "Reconcile the stage winner with independent repetitions before "
                "the next-stage handoff."
            ),
            experiment_goal="Aggregate stage-boundary replication evidence.",
            success_criteria=[
                "Agreement, variation, and contradictions across repetitions are explicit."
            ],
            related_node_ids=[winner.id, *progress.replication_node_ids],
            visual_artifacts=list(
                dict.fromkeys(
                    artifact
                    for node in [winner, *replications]
                    for artifact in node.visual_artifacts
                )
            )[: self.config.max_visual_artifacts_per_node],
        )
        aggregated = self._execute_candidates(
            state,
            pool,
            [(aggregation_id, aggregation)],
            parent=winner,
            boundary_validation=True,
            inherit_parent_state=False,
            replication_summary=summary,
        )
        if aggregated:
            progress.aggregation_node_id = aggregated[0].id

    def _run_deterministic_replications(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        stage: ResearchStage,
        winner: ExperimentNode,
        *,
        reserve_aggregation: int,
    ) -> list[ExperimentNode]:
        """Replay exactly the winning node's realized actions in clean executions."""
        if not winner.executed_actions:
            self.trace.write(
                "deterministic_replication_skipped",
                {
                    "stage": stage.value,
                    "winner_node_id": winner.id,
                    "reason": "winner_has_no_executed_actions",
                },
            )
            return []

        action_count = len(winner.executed_actions)
        node_capacity = max(
            0,
            self._remaining_node_capacity(state, boundary_validation=True)
            - reserve_aggregation,
        )
        remaining_llm = max(
            0,
            self.max_llm_calls - 1 - self._llm_calls_used - reserve_aggregation,
        )
        maximum = min(
            self.config.stage_boundary_replications,
            node_capacity,
            pool.remaining_calls // action_count,
            remaining_llm,
        )
        prepared: list[tuple[ExperimentNode, BranchExecution, list[PlannedAction]]] = []
        progress = state.stage_progress(stage)
        for seed in range(maximum):
            branch = pool.acquire(
                None,
                state.tree,
                prefer_existing=False,
                prefer_clone=False,
            )
            plan, seed_overrides = branch.executor.prepare_replication_plan(
                winner.executed_actions,
                seed=seed,
            )
            seeded = bool(seed_overrides)
            node = ExperimentNode(
                id=state.tree.next_id(),
                parent_id=winner.id,
                branch_id=branch.branch_id,
                execution_id=branch.execution_id,
                branch_workspace=branch.workspace,
                stage=stage,
                stage_seed_id=progress.seed_node_id,
                boundary_validation=True,
                physical_state_inherited=False,
                node_type=NodeType.REPLICATION,
                hypothesis=winner.hypothesis,
                rationale=(
                    f"Exact clean-execution replay of {winner.id} with seed {seed}."
                    if seeded
                    else f"Exact independent clean-execution replay of {winner.id}."
                ),
                experiment_goal=winner.experiment_goal,
                success_criteria=list(winner.success_criteria),
                related_node_ids=[winner.id],
                allocated_action_budget=len(plan),
                replication_seed=seed if seeded else None,
                replication_seed_overrides=seed_overrides,
                depth=winner.depth + 1,
            )
            self.trace.write("node_planned", node.model_dump(mode="json"))
            self.trace.write(
                "branch_assigned",
                {
                    "node_id": node.id,
                    "branch_id": branch.branch_id,
                    "execution_id": branch.execution_id,
                    "replayed_actions": 0,
                    "inheritance_method": "fresh_exact_replication",
                },
            )
            self.trace.write(
                "replication_plan_fixed",
                {
                    "node_id": node.id,
                    "source_node_id": winner.id,
                    "seed": node.replication_seed,
                    "seed_overrides": seed_overrides,
                    "actions": [action.model_dump(mode="json") for action in plan],
                },
            )
            prepared.append((node, branch, plan))

        def execute(
            item: tuple[ExperimentNode, BranchExecution, list[PlannedAction]],
        ) -> ExperimentNode:
            node, branch, plan = item
            node.status = NodeStatus.RUNNING
            observations = branch.executor.execute_plan(plan)
            node.plan = list(plan)
            node.observations = observations
            node.trajectory = [
                ExperimentStep(
                    step_index=index,
                    decision=ExperimentDecision(
                        decision="act",
                        rationale=(
                            "Deterministic stage-boundary replay of the winning "
                            "experiment action."
                        ),
                        purpose=action.purpose,
                        tool_name=action.tool_name,
                        arguments=action.arguments,
                        expected_information=action.expected_information,
                    ),
                    observation=observation,
                )
                for index, (action, observation) in enumerate(
                    zip(plan, observations, strict=False)
                )
            ]
            pool.commit(branch, node, 0)
            if any(not observation.success for observation in observations):
                node.status = NodeStatus.FAILED
                node.termination_reason = ExperimentTermination.ACTION_FAILED
                node.worker_conclusion = "The exact replication encountered a failure."
            elif len(observations) < len(plan):
                node.status = NodeStatus.PARTIAL
                node.termination_reason = ExperimentTermination.TOOL_BUDGET_EXHAUSTED
                node.worker_conclusion = "The exact replication did not finish."
            elif observations:
                node.status = NodeStatus.SUCCESSFUL
                node.termination_reason = ExperimentTermination.REPLICATION_REPLAYED
                node.worker_conclusion = "The exact replication completed."
            else:
                node.status = NodeStatus.INVALID
                node.worker_conclusion = (
                    "The exact replication produced no observation."
                )
            return node

        workers = min(len(prepared), self.config.parallel_experiment_workers)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = list(executor.map(execute, prepared))
        else:
            pending = [execute(item) for item in prepared]
        return self._evaluate_and_record_nodes(
            state,
            pool,
            pending,
            branches=[(node, branch) for node, branch, _ in prepared],
        )

    @staticmethod
    def _summarize_replications(
        replications: list[ExperimentNode],
    ) -> ReplicationSummary:
        """Build objective replication statistics without LLM interpretation."""
        successful = [
            node
            for node in replications
            if node.status in {NodeStatus.SUCCESSFUL, NodeStatus.INVALID}
            and node.observations
            and node.observations[-1].success
            and node.termination_reason
            in {
                ExperimentTermination.WORKER_FINISHED,
                ExperimentTermination.REPLICATION_REPLAYED,
            }
        ]
        outcomes = [
            node.measured_outcome
            for node in successful
            if node.measured_outcome is not None
        ]
        common_metric = bool(outcomes) and all(
            outcome.name.casefold() == outcomes[0].name.casefold()
            and outcome.maximize == outcomes[0].maximize
            for outcome in outcomes[1:]
        )
        values = [outcome.value for outcome in outcomes] if common_metric else []
        mean = fmean(values) if values else None
        standard_deviation = (
            stdev(values) if len(values) > 1 else (0.0 if values else None)
        )
        standard_error = (
            standard_deviation / sqrt(len(values))
            if standard_deviation is not None
            else None
        )
        return ReplicationSummary(
            n_runs=len(replications),
            n_successful=len(successful),
            metric_name=outcomes[0].name if common_metric else None,
            values=values,
            mean=mean,
            std=standard_deviation,
            stderr=standard_error,
            seeds=[node.replication_seed for node in replications],
        )

    def _evaluate_and_record_nodes(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        pending: list[ExperimentNode],
        *,
        branches: list[tuple[ExperimentNode, BranchExecution | None]],
    ) -> list[ExperimentNode]:
        """Extract evidence, evaluate nodes, and persist them consistently."""
        if self.config.enable_visual_feedback:
            for node, branch in branches:
                if branch is not None:
                    node.visual_artifacts = pool.visual_artifacts(
                        branch,
                        limit=self.config.max_visual_artifacts_per_node,
                        max_bytes=self.config.max_visual_artifact_bytes,
                    )
        for node in pending:
            node.measured_outcome = extract_measured_outcome(node, state.formulation)

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
                and (
                    node.stage != ResearchStage.PRELIMINARY
                    or self.config.preliminary_require_critic_validity
                )
            ):
                node.status = NodeStatus.INVALID
            state.tree.add(node)
            if not node.boundary_validation and node.substage_id is not None:
                progress = state.stage_progress(node.stage)
                substage = next(
                    (
                        item
                        for item in progress.substages
                        if item.id == node.substage_id
                    ),
                    None,
                )
                if substage is not None:
                    substage.node_ids.append(node.id)
            state.journal.integrate(node, self.config.minimum_validity)
            self.trace.write("node_completed", node.model_dump(mode="json"))
            self.trace.write("journal", state.journal.as_dict())

        self._update_usage(state, pool)
        state.llm_calls = self._llm_calls_used
        state.llm_tokens = self._llm_tokens_used
        return pending

    @staticmethod
    def _current_substage_id(state: ScientistState, stage: ResearchStage) -> str | None:
        substage = state.stage_progress(stage).current_substage
        return substage.id if substage is not None else None

    @staticmethod
    def _search_node_count(state: ScientistState, stage: ResearchStage) -> int:
        return len(state.tree.by_stage(stage, include_boundary=False))

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
        # Every proposal will execute in a separate execution workspace. Relative
        # paths therefore isolate artifacts without synthetic node directories.
        branch_workspaces = ["." for _ in candidate_ids]
        journal_context = state.journal.context(self.config.max_journal_chars)
        siblings = [
            node
            for node in state.tree.children(parent.id if parent is not None else None)
            if node.stage == stage and not node.boundary_validation
        ]
        progress = state.stage_progress(stage)
        current_substage = progress.current_substage

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
                substage=(current_substage.plan if current_substage else None),
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

    def _propose_for_parent_batch(
        self,
        state: ScientistState,
        *,
        stage: ResearchStage,
        parent_requests: list[tuple[ExperimentNode, NodeType]],
    ) -> list[tuple[str, NodeProposal, ExperimentNode]]:
        """Plan one child for every independently selected BFTS parent."""
        candidate_ids = [state.tree.next_id() for _ in parent_requests]
        journal_context = state.journal.context(self.config.max_journal_chars)
        progress = state.stage_progress(stage)
        current_substage = progress.current_substage
        existing_siblings = {
            parent.id: [
                node
                for node in state.tree.children(parent.id)
                if node.stage == stage and not node.boundary_validation
            ]
            for parent, _ in parent_requests
        }
        batch_offsets: list[int] = []
        seen_parent_counts: dict[str, int] = {}
        for parent, _ in parent_requests:
            batch_offsets.append(
                len(existing_siblings[parent.id]) + seen_parent_counts.get(parent.id, 0)
            )
            seen_parent_counts[parent.id] = seen_parent_counts.get(parent.id, 0) + 1

        def propose_one(index: int) -> tuple[str, NodeProposal | None, ExperimentNode]:
            parent, node_type = parent_requests[index]
            proposals = self.planner.propose(
                stage=stage,
                node_type=node_type,
                count=1,
                task_prompt=state.task_prompt,
                formulation=state.formulation,
                tools=state.tools,
                journal_context=journal_context,
                parent=parent,
                siblings=existing_siblings[parent.id],
                branch_workspaces=["."],
                substage=(current_substage.plan if current_substage else None),
                proposal_offset=batch_offsets[index],
            )
            return (
                candidate_ids[index],
                proposals[0] if proposals else None,
                parent,
            )

        if len(parent_requests) > 1 and self.config.parallel_llm_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(parent_requests), self.config.parallel_llm_workers)
            ) as planning_pool:
                planned = list(
                    planning_pool.map(propose_one, range(len(parent_requests)))
                )
        else:
            planned = [propose_one(index) for index in range(len(parent_requests))]

        fingerprints = {
            parent.id: {
                self._experiment_fingerprint(sibling)
                for sibling in existing_siblings[parent.id]
            }
            for parent, _ in parent_requests
        }
        unique: list[tuple[str, NodeProposal, ExperimentNode]] = []
        for node_id, proposal, parent in planned:
            if proposal is None:
                continue
            fingerprint = self._experiment_fingerprint(proposal)
            if fingerprint in fingerprints[parent.id]:
                self.trace.write(
                    "proposal_duplicate_rejected",
                    {
                        "node_id": node_id,
                        "parent_id": parent.id,
                        "fingerprint": fingerprint,
                    },
                )
                continue
            fingerprints[parent.id].add(fingerprint)
            unique.append((node_id, proposal, parent))
        return unique

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
        pool: ExecutionPool,
        candidates: list[tuple[str, NodeProposal]],
        *,
        parent: ExperimentNode | None,
        substage_id: str | None = None,
        boundary_validation: bool = False,
        inherit_parent_state: bool | None = None,
        replication_summary: ReplicationSummary | None = None,
    ) -> list[ExperimentNode]:
        """Run isolated sibling experiments and evaluations concurrently."""
        return self._execute_parent_candidates(
            state,
            pool,
            [(node_id, proposal, parent) for node_id, proposal in candidates],
            substage_id=substage_id,
            boundary_validation=boundary_validation,
            inherit_parent_state=inherit_parent_state,
            replication_summary=replication_summary,
        )

    def _execute_parent_candidates(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        candidates: list[tuple[str, NodeProposal, ExperimentNode | None]],
        *,
        substage_id: str | None = None,
        boundary_validation: bool = False,
        inherit_parent_state: bool | None = None,
        replication_summary: ReplicationSummary | None = None,
    ) -> list[ExperimentNode]:
        """Run a mixed-parent worker batch in isolated executions concurrently."""
        node_capacity = self._remaining_node_capacity(
            state,
            boundary_validation=boundary_validation,
        )
        prepared: list[tuple[ExperimentNode, BranchExecution | None, int, int]] = []
        reserved_scientific_calls = 0
        reserved_llm_calls = 0
        remaining_llm_calls = max(0, self.max_llm_calls - 1 - self._llm_calls_used)
        candidate_batch = candidates[:node_capacity]
        inheritance_strategy = self.config.execution_state_inheritance
        if inherit_parent_state is not None:
            inheritance_strategy = "replay" if inherit_parent_state else "clean"

        def runtime_parent(
            proposal: NodeProposal,
            parent: ExperimentNode | None,
        ) -> ExperimentNode | None:
            if parent is None or proposal.node_type == NodeType.AGGREGATION:
                return None
            if proposal.node_type == NodeType.CONTINUE:
                return parent
            if inheritance_strategy == "replay" or (
                inheritance_strategy == "auto" and pool.can_clone(parent)
            ):
                return parent
            return None

        def prefer_clone(
            proposal: NodeProposal,
            parent: ExperimentNode | None,
        ) -> bool:
            return (
                parent is not None
                and runtime_parent(proposal, parent) is not None
                and pool.can_clone(parent)
            )

        def prefer_existing(
            index: int,
            proposal: NodeProposal,
            parent: ExperimentNode | None,
        ) -> bool:
            # Keep an inherited baseline runtime parked at its checkpoint so
            # later controlled alternatives can clone the same physical state.
            first_for_parent = not any(
                prior_parent is not None
                and parent is not None
                and prior_parent.id == parent.id
                for _, _, prior_parent in candidate_batch[:index]
            )
            return first_for_parent and (
                proposal.node_type == NodeType.CONTINUE
                or not prefer_clone(proposal, parent)
            )

        replay_costs = [
            (
                0
                if proposal.node_type == NodeType.AGGREGATION
                else pool.replay_cost(
                    runtime_parent(proposal, parent),
                    state.tree,
                    prefer_existing=prefer_existing(index, proposal, parent),
                    prefer_clone=prefer_clone(proposal, parent),
                )
            )
            for index, (_, proposal, parent) in enumerate(candidate_batch)
        ]

        for index, (node_id, proposal, parent) in enumerate(candidate_batch):
            is_aggregation = proposal.node_type == NodeType.AGGREGATION
            available = pool.remaining_calls - reserved_scientific_calls
            replay_cost = replay_costs[index]
            future_tool_minimum = sum(
                future_replay
                + (0 if future_proposal.node_type == NodeType.AGGREGATION else 1)
                for future_replay, (_, future_proposal, _) in zip(
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
                for _, future, _ in candidate_batch[index + 1 :]
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
            branch: BranchExecution | None = None
            if not is_aggregation:
                try:
                    branch = pool.acquire(
                        runtime_parent(proposal, parent),
                        state.tree,
                        prefer_existing=prefer_existing(index, proposal, parent),
                        prefer_clone=prefer_clone(proposal, parent),
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
                execution_id=(branch.execution_id if branch is not None else None),
                branch_workspace=branch.workspace if branch is not None else None,
                stage=state.current_stage,
                stage_seed_id=state.stage_progress(state.current_stage).seed_node_id,
                substage_id=substage_id,
                boundary_validation=boundary_validation,
                physical_state_inherited=runtime_parent(proposal, parent) is not None,
                node_type=proposal.node_type,
                hypothesis=proposal.hypothesis,
                rationale=proposal.rationale,
                experiment_goal=proposal.experiment_goal,
                success_criteria=proposal.success_criteria,
                related_node_ids=proposal.related_node_ids,
                visual_artifacts=proposal.visual_artifacts,
                replication_summary=(replication_summary if is_aggregation else None),
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
                        "execution_id": branch.execution_id,
                        "replayed_actions": len(branch.replay_results),
                        "inheritance_method": branch.inheritance_method,
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
            item: tuple[ExperimentNode, BranchExecution | None, int, int],
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
                previous_node=(
                    state.tree.get(node.parent_id)
                    if node.parent_id is not None
                    else None
                ),
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
        return self._evaluate_and_record_nodes(
            state,
            pool,
            pending,
            branches=[(node, branch) for node, branch, _, _ in prepared],
        )

    def _affordable_candidate_count(
        self,
        maximum: int,
        *,
        independent_planning: bool = False,
    ) -> int:
        """Fit planning, adaptive execution, completion, and evaluation."""
        if maximum <= 0 or not self._within_token_budget:
            return 0
        remaining = self.max_llm_calls - 1 - self._llm_calls_used
        for count in range(maximum, 0, -1):
            planning_calls = (
                count
                if independent_planning
                or (count > 1 and self.config.parallel_llm_workers > 1)
                else 1
            )
            if planning_calls + count * 3 <= remaining:
                return count
        return 0

    def _can_create(
        self,
        state: ScientistState,
        pool: ExecutionPool,
        *,
        llm_calls: int,
        requires_tool_budget: bool = True,
        boundary_validation: bool = False,
    ) -> bool:
        # Keep one LLM call in reserve for final synthesis.
        return (
            self._remaining_node_capacity(
                state,
                boundary_validation=boundary_validation,
            )
            > 0
            and (not requires_tool_budget or pool.remaining_calls > 0)
            and self._llm_calls_used + llm_calls <= self.max_llm_calls - 1
            and self._within_token_budget
        )

    def _remaining_node_capacity(
        self,
        state: ScientistState,
        *,
        boundary_validation: bool = False,
    ) -> int:
        used = sum(
            node.boundary_validation == boundary_validation for node in state.tree.nodes
        )
        budget = (
            self.config.validation_node_budget
            if boundary_validation
            else self.config.search_node_budget
        )
        return max(0, budget - used)

    def _update_usage(self, state: ScientistState, pool: ExecutionPool) -> None:
        state.tool_calls = pool.call_count
        state.scientific_tool_calls = pool.budget.scientific_calls
        state.replay_tool_calls = pool.budget.replay_calls
        state.executions_created = pool.executions_created
        state.executions_cloned = pool.executions_cloned
        state.peak_simultaneous_executions = pool.peak_simultaneous_executions
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
                and not self.config.debug_leaf_only
            ):
                return NodeType.DEBUG
            if parent.evaluation.recommendation == Recommendation.REFINE:
                return NodeType.REFINE
        return successful_type

    def _stage_complete(self, state: ScientistState, stage: ResearchStage) -> bool:
        nodes = [
            node
            for node in state.tree.by_stage(stage, include_boundary=False)
            if self._eligible_stage_candidate(node)
        ]
        if not nodes:
            return False
        if stage == ResearchStage.PRELIMINARY:
            return any(
                (
                    not self.config.preliminary_require_critic_validity
                    or node.evaluation.validity >= self.config.minimum_validity
                )
                and node.evaluation.evidence_strength
                >= self.config.preliminary_evidence_threshold
                for node in nodes
            )
        if stage in {ResearchStage.TUNING, ResearchStage.RESEARCH}:
            return any(
                node.evaluation.recommendation == Recommendation.FINALIZE
                and node.evaluation.task_progress
                >= self.config.stage_completion_threshold
                and node.evaluation.evidence_strength
                >= self.config.preliminary_evidence_threshold
                and self._provisionally_beats_seed(state, stage, node)
                for node in nodes
            )
        if stage == ResearchStage.VERIFICATION:
            return any(
                node.node_type == NodeType.ABLATION
                and node.evaluation.validity >= self.config.minimum_validity
                and self._confidence(node.evaluation)
                >= self.config.confidence_threshold
                for node in nodes
            )
        return False

    def _provisionally_beats_seed(
        self,
        state: ScientistState,
        stage: ResearchStage,
        node: ExperimentNode,
    ) -> bool:
        seed_node_id = state.stage_progress(stage).seed_node_id
        if seed_node_id is None:
            return True
        return self._beats_seed(state, stage, node, inclusive=False)

    def _beats_seed(
        self,
        state: ScientistState,
        stage: ResearchStage,
        node: ExperimentNode,
        *,
        inclusive: bool,
    ) -> bool:
        seed_node_id = state.stage_progress(stage).seed_node_id
        if seed_node_id is None:
            return True
        if node.id == seed_node_id:
            return False
        seed = state.tree.get(seed_node_id)
        objective_delta = measured_improvement(node, seed)
        if objective_delta is not None:
            threshold = self.config.minimum_measured_improvement
            if inclusive and threshold > 0:
                return objective_delta >= threshold
            return objective_delta > threshold
        if seed.evaluation is None:
            return True
        threshold = self.config.minimum_stage_improvement
        # At stage handoff, a valid listwise-selected non-seed winner is enough
        # unless the caller requested a quantitative critic-score margin. The
        # provisional early-stop check remains strict.
        if inclusive and threshold == 0:
            return True
        critic_delta = evaluation_priority(node) - evaluation_priority(seed)
        return critic_delta >= threshold if inclusive else critic_delta > threshold

    @staticmethod
    def _confidence(evaluation: NodeEvaluation) -> float:
        return (
            evaluation.validity + evaluation.evidence_strength + evaluation.consistency
        ) / 3

    def _eligible_stage_candidate(self, node: ExperimentNode) -> bool:
        """Apply the configured working-node gate before stage comparison."""
        if node.status != NodeStatus.SUCCESSFUL or node.evaluation is None:
            return False
        if (
            node.stage == ResearchStage.PRELIMINARY
            and not self.config.preliminary_require_critic_validity
        ):
            return True
        return node.evaluation.recommendation != Recommendation.ABANDON

    def _transition(self, state: ScientistState, stage: ResearchStage) -> None:
        state.current_stage = stage
        logger.info("AI Scientist entering {} stage", stage.value)
        self.trace.write("stage_transition", {"stage": stage.value})
