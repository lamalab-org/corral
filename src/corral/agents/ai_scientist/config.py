"""Configuration for the AI Scientist scaffold."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AIScientistConfig(BaseModel):
    """Search, budget, and persistence controls.

    Node budgets are upper bounds for each stage, not unconditional iteration
    counts. The manager may transition earlier when the stage's evidence-based
    completion criterion is met. ``max_llm_tokens`` stops further search after
    the provider-reported total reaches the ceiling; final synthesis is still
    reserved so the run can return an answer.
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
    max_nodes: int = Field(default=3, ge=1)
    max_tool_calls: int = Field(default=32, ge=0)
    max_llm_calls: int = Field(default=64, ge=2)
    max_llm_tokens: int | None = Field(default=None, ge=1)
    max_actions_per_node: int = Field(default=3, ge=1)
    max_debug_depth: int = Field(default=2, ge=0)
    debug_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    parallel_llm_workers: int = Field(default=4, ge=1, le=16)
    parallel_experiment_workers: int = Field(default=3, ge=1, le=16)

    # The manager can revise the experimental agenda within a main stage. The
    # first substage is deterministic; later substages are generated from the
    # accumulated evidence after this many search nodes.
    adaptive_substages: bool = True
    max_substages_per_stage: int = Field(default=3, ge=1, le=16)
    nodes_per_substage: int = Field(default=3, ge=1)

    # Repeat the winning experiment at each main-stage boundary and reconcile
    # those repetitions before seeding the next stage. Set to zero for tasks
    # that are known to be deterministic or too expensive to repeat.
    stage_boundary_replications: int = Field(default=3, ge=0, le=16)
    aggregate_stage_replications: bool = True

    # Stage 4 is systematic ablation/assumption testing. Counterfactuals are a
    # useful Corral generalisation, but replication and aggregation belong at
    # stage boundaries rather than in the Stage-4 node cycle.
    verification_include_counterfactual: bool = True

    # Non-continuation experiments start in clean trials and inherit the
    # parent's scientific design through prompts. Opting into physical-state
    # inheritance restores replay-on-fork for environments that truly need it.
    inherit_parent_trial_state: bool = False

    # Local image artifacts are sent to a multimodal evaluator when the model
    # gateway supports that call shape.
    enable_visual_feedback: bool = True
    max_visual_artifacts_per_node: int = Field(default=4, ge=0, le=32)
    max_visual_artifact_bytes: int = Field(default=5_000_000, ge=1_024)

    preliminary_evidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    stage_completion_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    minimum_validity: float = Field(default=0.45, ge=0.0, le=1.0)
    minimum_stage_improvement: float = Field(default=0.0, ge=0.0, le=1.0)

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
    def planned_node_budget(self) -> int:
        """Maximum nodes reachable through all enabled stages."""
        return min(
            self.max_nodes,
            self.preliminary_node_budget
            + self.tuning_node_budget
            + self.research_node_budget
            + self.verification_node_budget,
        )
