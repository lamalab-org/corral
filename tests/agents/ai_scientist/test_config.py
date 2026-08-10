import pytest
from pydantic import ValidationError

from corral.agents.ai_scientist import AIScientistConfig, SakanaAIScientistConfig


def test_default_search_is_not_capped_at_the_three_initial_drafts():
    config = AIScientistConfig()

    assert config.max_nodes is None
    assert config.max_search_nodes is None
    assert config.search_node_budget == 18
    assert config.validation_node_budget == 16
    assert config.planned_total_node_budget == 34
    assert config.max_actions_per_node == 3
    assert config.candidates_per_expansion == 3
    assert config.max_children_per_node == 3
    assert config.tree_exploration_weight == 0.1
    assert config.parallel_experiment_workers == 3


def test_config_rejects_inconsistent_stage_budgets():
    with pytest.raises(ValidationError, match="initial_drafts"):
        AIScientistConfig(initial_drafts=3, preliminary_node_budget=2)

    with pytest.raises(ValidationError, match="verification_min_nodes"):
        AIScientistConfig(verification_min_nodes=3, verification_node_budget=2)


def test_planned_node_budget_obeys_search_cap_but_keeps_validation_separate():
    config = AIScientistConfig(
        max_nodes=7,
        initial_drafts=1,
        preliminary_node_budget=2,
        tuning_node_budget=2,
        research_node_budget=3,
        verification_node_budget=4,
        verification_min_nodes=1,
    )

    assert config.planned_node_budget == 7
    assert config.validation_node_budget == 16


def test_config_rejects_both_search_cap_spellings():
    with pytest.raises(ValidationError, match="Set only one"):
        AIScientistConfig(max_nodes=2, max_search_nodes=3)


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


def test_legacy_trial_inheritance_flag_maps_to_explicit_strategy():
    assert (
        AIScientistConfig(
            inherit_parent_trial_state=True
        ).effective_trial_state_inheritance
        == "replay"
    )
    assert (
        AIScientistConfig(
            inherit_parent_trial_state=False
        ).effective_trial_state_inheritance
        == "clean"
    )
