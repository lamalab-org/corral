import pytest
from pydantic import ValidationError

from corral.agents.ai_scientist import AIScientistConfig


def test_default_search_uses_three_nodes():
    config = AIScientistConfig()

    assert config.max_nodes == 3
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


def test_planned_node_budget_obeys_global_cap():
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
