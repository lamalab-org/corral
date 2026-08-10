"""Deterministic progressive experiment orchestration."""

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle

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
    ExperimentNode,
    NodeEvaluation,
    NodeProposal,
    NodeStatus,
    NodeType,
    Recommendation,
    ResearchStage,
    SubstagePlan,
)
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.state import (
    ScientistState,
    StageProgress,
    SubstageState,
)
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
            and state.formulation.has_tunable_parameters
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

    def _preliminary(self, state: ScientistState, pool: TrialPool) -> None:
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
        pool: TrialPool,
        *,
        stage: ResearchStage,
        budget: int,
        default_type: NodeType,
    ) -> None:
        while (
            self._search_node_count(state, stage) < budget
            and not self._stage_complete(state, stage)
            and self._can_create(state, pool, llm_calls=self._calls_per_node)
        ):
            self._maybe_advance_substage(state, stage)
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

    def _verification(self, state: ScientistState, pool: TrialPool) -> None:
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
        )

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
            self._llm_calls_used + 2 + self._calls_per_node
            > self.config.max_llm_calls - 1
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
        pool: TrialPool,
        stage: ResearchStage,
    ) -> ExperimentNode | None:
        """Comparatively select, validate, and replicate a stage winner."""
        progress = state.stage_progress(stage)
        stage_nodes = state.tree.by_stage(stage, include_boundary=False)
        candidates = [
            node
            for node in stage_nodes
            if node.status == NodeStatus.SUCCESSFUL
            and node.evaluation is not None
            and node.evaluation.recommendation != Recommendation.ABANDON
        ]
        if progress.seed_node_id is not None:
            seed = state.tree.get(progress.seed_node_id)
            if (
                seed.status == NodeStatus.SUCCESSFUL
                and seed.evaluation is not None
                and seed.evaluation.recommendation != Recommendation.ABANDON
            ):
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
        progress.completion_criteria_met = self._stage_completion_for_winner(
            state, stage, winner
        )
        self.trace.write(
            "stage_winner_selected",
            progress.model_dump(mode="json"),
        )
        if progress.completion_criteria_met:
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
            self._llm_calls_used + 1 > self.config.max_llm_calls - 1
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
                evaluation.validity >= self.config.minimum_validity
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
        pool: TrialPool,
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
        remaining_llm = self.config.max_llm_calls - 1 - self._llm_calls_used
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
        )
        if aggregated:
            progress.aggregation_node_id = aggregated[0].id

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
        # Every proposal will execute in a separate trial workspace. Relative
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
        substage_id: str | None = None,
        boundary_validation: bool = False,
        inherit_parent_state: bool | None = None,
    ) -> list[ExperimentNode]:
        """Run isolated sibling experiments and evaluations concurrently."""
        node_capacity = self._remaining_node_capacity(
            state,
            boundary_validation=boundary_validation,
        )
        prepared: list[tuple[ExperimentNode, BranchRuntime | None, int, int]] = []
        reserved_scientific_calls = 0
        reserved_llm_calls = 0
        remaining_llm_calls = max(
            0, self.config.max_llm_calls - 1 - self._llm_calls_used
        )
        candidate_batch = candidates[:node_capacity]
        inheritance_strategy = self.config.effective_trial_state_inheritance
        if inherit_parent_state is not None:
            inheritance_strategy = "replay" if inherit_parent_state else "clean"

        def runtime_parent(proposal: NodeProposal) -> ExperimentNode | None:
            if parent is None or proposal.node_type == NodeType.AGGREGATION:
                return None
            if proposal.node_type == NodeType.CONTINUE:
                return parent
            if inheritance_strategy == "replay" or (
                inheritance_strategy == "auto" and pool.can_clone(parent)
            ):
                return parent
            return None

        def prefer_clone(proposal: NodeProposal) -> bool:
            return (
                parent is not None
                and runtime_parent(proposal) is not None
                and pool.can_clone(parent)
            )

        def prefer_existing(index: int, proposal: NodeProposal) -> bool:
            # Keep an inherited baseline runtime parked at its checkpoint so
            # later controlled alternatives can clone the same physical state.
            return index == 0 and (
                proposal.node_type == NodeType.CONTINUE or not prefer_clone(proposal)
            )

        replay_costs = [
            (
                0
                if proposal.node_type == NodeType.AGGREGATION
                else pool.replay_cost(
                    runtime_parent(proposal),
                    state.tree,
                    prefer_existing=prefer_existing(index, proposal),
                    prefer_clone=prefer_clone(proposal),
                )
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
                        runtime_parent(proposal),
                        state.tree,
                        prefer_existing=prefer_existing(index, proposal),
                        prefer_clone=prefer_clone(proposal),
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
                stage_seed_id=state.stage_progress(state.current_stage).seed_node_id,
                substage_id=substage_id,
                boundary_validation=boundary_validation,
                physical_state_inherited=runtime_parent(proposal) is not None,
                node_type=proposal.node_type,
                hypothesis=proposal.hypothesis,
                rationale=proposal.rationale,
                experiment_goal=proposal.experiment_goal,
                success_criteria=proposal.success_criteria,
                related_node_ids=proposal.related_node_ids,
                visual_artifacts=proposal.visual_artifacts,
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
                previous_node=parent,
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

        if self.config.enable_visual_feedback:
            for node, branch, _, _ in prepared:
                if branch is not None:
                    node.visual_artifacts = pool.visual_artifacts(
                        branch,
                        limit=self.config.max_visual_artifacts_per_node,
                        max_bytes=self.config.max_visual_artifact_bytes,
                    )

        for node in pending:
            node.measured_outcome = extract_measured_outcome(
                node,
                state.formulation,
            )

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
            and self._llm_calls_used + llm_calls <= self.config.max_llm_calls - 1
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

    def _update_usage(self, state: ScientistState, pool: TrialPool) -> None:
        state.tool_calls = pool.call_count
        state.scientific_tool_calls = pool.budget.scientific_calls
        state.replay_tool_calls = pool.budget.replay_calls
        state.trial_runtimes_created = pool.trials_created
        state.trial_runtimes_cloned = pool.trials_cloned
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
        nodes = [
            node
            for node in state.tree.by_stage(stage, include_boundary=False)
            if node.evaluation is not None
            and node.status == NodeStatus.SUCCESSFUL
            and node.evaluation.recommendation != Recommendation.ABANDON
        ]
        if not nodes:
            return False
        if stage == ResearchStage.PRELIMINARY:
            return any(
                node.evaluation.validity >= self.config.minimum_validity
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

    def _transition(self, state: ScientistState, stage: ResearchStage) -> None:
        state.current_stage = stage
        logger.info("AI Scientist entering {} stage", stage.value)
        self.trace.write("stage_transition", {"stage": stage.value})
