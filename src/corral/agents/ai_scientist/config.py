"""Configuration profiles for the AI Scientist scaffold."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AIScientistConfig(BaseModel):
    """Search, budget, and persistence controls.

    Search-node budgets are upper bounds for each stage, not unconditional
    iteration counts. Boundary replications and aggregations have a separate
    validation-node budget, so validating one stage can never consume the
    capacity reserved for later scientific search. `max_llm_tokens` stops
    further search after the provider-reported total reaches the ceiling; final
    synthesis is still reserved so the run can return an answer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_drafts: int = Field(default=3, ge=1)
    preliminary_node_budget: int = Field(default=5, ge=1)
    tuning_node_budget: int = Field(default=3, ge=0)
    research_node_budget: int = Field(default=6, ge=1)
    verification_node_budget: int = Field(default=4, ge=1)
    verification_min_nodes: int = Field(default=4, ge=1)

    # Ordinary expansions should expose the configured worker parallelism. A
    # value of one made the search serial after the independent root drafts.
    candidates_per_expansion: int = Field(default=3, ge=1, le=4)
    max_children_per_node: int = Field(default=3, ge=1, le=16)
    tree_exploration_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    # Corral's economical default creates several sibling proposals from one
    # best parent. The fidelity profile instead fills an experiment-worker
    # batch with independently selected parents and represents distinct root
    # trees before reusing one, as AI Scientist v2's parallel BFTS does.
    parallel_parent_selection: bool = False
    prefer_distinct_root_trees: bool = False
    parent_selection_mode: Literal["deterministic", "llm"] = "deterministic"
    # Leaving both global caps unset relies on the explicit per-stage budgets.
    max_search_nodes: int | None = Field(default=None, ge=1)
    max_validation_nodes: int | None = Field(default=None, ge=0)
    max_tool_calls: int = Field(default=32, ge=0)
    max_llm_tokens: int | None = Field(default=None, ge=1)
    max_actions_per_node: int = Field(default=3, ge=1)
    max_debug_depth: int = Field(default=2, ge=0)
    debug_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    # AI Scientist v2 only revisits failed leaves: after a DEBUG child exists,
    # the child becomes the next debuggable checkpoint instead of opening
    # several independent repairs from the same failed ancestor.
    debug_leaf_only: bool = False
    parallel_llm_workers: int = Field(default=4, ge=1, le=16)
    parallel_experiment_workers: int = Field(default=3, ge=1, le=16)

    # The manager can revise the experimental agenda within a main stage. The
    # first substage is deterministic; after this many additional search nodes
    # the critic checks its evidence-based completion criteria. A new agenda is
    # created only when that check succeeds.
    adaptive_substages: bool = True
    max_substages_per_stage: int = Field(default=3, ge=1, le=16)
    nodes_per_substage: int = Field(default=3, ge=1)

    # Repeat the winning experiment at each main-stage boundary and reconcile
    # those repetitions before seeding the next stage. Set to zero for tasks
    # that are known to be deterministic or too expensive to repeat.
    stage_boundary_replications: int = Field(default=3, ge=0, le=16)
    aggregate_stage_replications: bool = True
    # Exact replication bypasses the adaptive experiment worker and replays the
    # selected node's realized actions in clean executions. When a tool schema
    # exposes `seed` or `random_state`, only those arguments are changed.
    deterministic_replication: bool = False

    # Stage 4 is systematic ablation/assumption testing. Counterfactuals are a
    # useful Corral generalisation, but replication and aggregation belong at
    # stage boundaries rather than in the Stage-4 node cycle.
    verification_include_counterfactual: bool = True
    research_early_stopping: bool = True
    verification_early_stopping: bool = True
    # Sakana treats reaching the configured iteration count as successful
    # stage termination for Stages 2--4 and still evaluates the best node over
    # multiple seeds. Stage 1 remains a hard working-implementation gate.
    validate_on_stage_budget_exhaustion: bool = False
    # Corral normally omits Stage 2 when formulation finds no explicit tunable
    # parameter. The fidelity profile treats procedural/experimental choices
    # as tunable and therefore always attempts the stage when it has a budget.
    force_tuning_stage: bool = False

    # `auto` uses the branch-session provider's optional cloning capability and
    # otherwise starts non-continuation children clean. `replay` reconstructs
    # parent state when cloning is unavailable.
    execution_state_inheritance: Literal["auto", "clean", "replay"] = "auto"

    # Local image artifacts are sent to a multimodal evaluator when the model
    # gateway supports that call shape.
    enable_visual_feedback: bool = True
    max_visual_artifacts_per_node: int = Field(default=4, ge=0, le=32)
    max_visual_artifact_bytes: int = Field(default=5_000_000, ge=1_024)

    preliminary_evidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # The general Corral profile protects its Stage-1 gate with the critic's
    # validity threshold. Sakana's good-node gate asks only whether the
    # implementation executed successfully.
    preliminary_require_critic_validity: bool = True
    stage_completion_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    minimum_validity: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_stage_improvement: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_measured_improvement: float = Field(default=0.0, ge=0.0)

    random_seed: int = 0
    use_structured_output: bool = True
    stop_plan_on_tool_error: bool = True
    max_observation_chars: int | None = Field(default=12_000, ge=200)
    max_journal_chars: int = Field(default=30_000, ge=1_000)
    max_tool_schema_chars: int = Field(default=40_000, ge=1_000)
    trace_path: Path | None = None

    @model_validator(mode="after")
    def _consistent_budgets(self) -> "AIScientistConfig":
        if self.initial_drafts > self.preliminary_node_budget:
            raise ValueError("initial_drafts cannot exceed preliminary_node_budget")
        if self.verification_min_nodes > self.verification_node_budget:
            raise ValueError(
                "verification_min_nodes cannot exceed verification_node_budget"
            )
        return self

    @property
    def search_node_budget(self) -> int:
        """Effective global cap for ordinary scientific search nodes."""
        stage_total = (
            self.preliminary_node_budget
            + self.tuning_node_budget
            + self.research_node_budget
            + self.verification_node_budget
        )
        configured = self.max_search_nodes
        return min(stage_total, configured) if configured is not None else stage_total

    @property
    def validation_node_budget(self) -> int:
        """Effective cap for replications and aggregations at stage boundaries."""
        if self.max_validation_nodes is not None:
            return self.max_validation_nodes
        per_stage = self.stage_boundary_replications + int(
            self.aggregate_stage_replications and self.stage_boundary_replications > 0
        )
        return 4 * per_stage

    @property
    def planned_node_budget(self) -> int:
        """Maximum ordinary search nodes reachable through enabled stages."""
        return self.search_node_budget

    @property
    def planned_total_node_budget(self) -> int:
        """Maximum search plus stage-boundary validation nodes."""
        return self.search_node_budget + self.validation_node_budget


class SakanaAIScientistConfig(AIScientistConfig):
    """AI Scientist v2 fidelity profile rather than the cheaper Corral profile."""

    initial_drafts: int = Field(default=3, ge=1)
    preliminary_node_budget: int = Field(default=20, ge=1)
    tuning_node_budget: int = Field(default=12, ge=0)
    research_node_budget: int = Field(default=12, ge=1)
    verification_node_budget: int = Field(default=18, ge=1)
    verification_min_nodes: int = Field(default=18, ge=1)
    debug_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    max_debug_depth: int = Field(default=3, ge=0)
    debug_leaf_only: bool = True
    candidates_per_expansion: int = Field(default=4, ge=1, le=4)
    max_children_per_node: int = Field(default=16, ge=1, le=16)
    max_actions_per_node: int = Field(default=16, ge=1)
    parallel_experiment_workers: int = Field(default=4, ge=1, le=16)
    parallel_parent_selection: bool = True
    prefer_distinct_root_trees: bool = True
    parent_selection_mode: Literal["deterministic", "llm"] = "llm"
    tree_exploration_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_include_counterfactual: bool = False
    research_early_stopping: bool = False
    verification_early_stopping: bool = False
    validate_on_stage_budget_exhaustion: bool = True
    force_tuning_stage: bool = True
    deterministic_replication: bool = True
    preliminary_evidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    preliminary_require_critic_validity: bool = False
    max_tool_calls: int = Field(default=256, ge=0)
