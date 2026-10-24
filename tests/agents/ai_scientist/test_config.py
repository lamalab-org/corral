import pytest
from pydantic import ValidationError

from corral.agents.ai_scientist import AIScientistConfig, SakanaAIScientistConfig


def test_default_search_keeps_all_stages_within_the_reduced_budget():
    config = AIScientistConfig()

    assert config.initial_drafts == 2
    assert (
        config.preliminary_node_budget,
        config.tuning_node_budget,
        config.research_node_budget,
        config.verification_node_budget,
    ) == (3, 1, 2, 2)
    assert config.verification_min_nodes == 2
    assert config.max_search_nodes is None
    assert config.search_node_budget == 8
    assert config.stage_boundary_replications == 1
    assert config.validation_node_budget == 8
    assert config.planned_total_node_budget == 16
    assert config.max_actions_per_node == 3
    assert config.candidates_per_expansion == 2
    assert config.max_children_per_node == 3
    assert config.tree_exploration_weight == 0.1
    assert config.parallel_experiment_workers == 2


def test_config_rejects_inconsistent_stage_budgets():
    with pytest.raises(ValidationError, match="initial_drafts"):
        AIScientistConfig(initial_drafts=3, preliminary_node_budget=2)

    with pytest.raises(ValidationError, match="verification_min_nodes"):
        AIScientistConfig(verification_min_nodes=3, verification_node_budget=2)


def test_planned_node_budget_obeys_search_cap_but_keeps_validation_separate():
    config = AIScientistConfig(
        max_search_nodes=7,
        initial_drafts=1,
        preliminary_node_budget=2,
        tuning_node_budget=2,
        research_node_budget=3,
        verification_node_budget=4,
        verification_min_nodes=1,
    )

    assert config.planned_node_budget == 7
    assert config.validation_node_budget == 8


def test_sakana_fidelity_profile_uses_original_search_behavior():
    config = SakanaAIScientistConfig()

    assert config.initial_drafts == 3
    assert (
        config.preliminary_node_budget,
        config.tuning_node_budget,
        config.research_node_budget,
        config.verification_node_budget,
    ) == (20, 12, 12, 18)
    assert config.debug_probability == 0.5
    assert config.max_debug_depth == 3
    assert config.candidates_per_expansion == 4
    assert config.max_children_per_node == 16
    assert config.max_actions_per_node == 16
    assert config.parallel_experiment_workers == 4
    assert config.stage_boundary_replications == 3
    assert config.validation_node_budget == 16
    assert config.planned_total_node_budget == 78
    assert config.parallel_parent_selection is True
    assert config.prefer_distinct_root_trees is True
    assert config.parent_selection_mode == "llm"
    assert config.tree_exploration_weight == 0.0
    assert config.verification_include_counterfactual is False
    assert config.research_early_stopping is False
    assert config.verification_early_stopping is False
    assert config.validate_on_stage_budget_exhaustion is True
    assert config.debug_leaf_only is True
    assert config.force_tuning_stage is True
    assert config.deterministic_replication is True
    assert config.preliminary_evidence_threshold == 0.0
    assert config.preliminary_require_critic_validity is False


def test_execution_inheritance_strategy_is_explicit():
    assert (
        AIScientistConfig(
            execution_state_inheritance="replay"
        ).execution_state_inheritance
        == "replay"
    )
    assert (
        AIScientistConfig(
            execution_state_inheritance="clean"
        ).execution_state_inheritance
        == "clean"
    )


def test_removed_trial_inheritance_flag_is_rejected():
    with pytest.raises(ValidationError):
        AIScientistConfig(inherit_parent_trial_state=True)
