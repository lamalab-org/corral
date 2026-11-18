"""Every worker prompt shows experiment evidence without step rationales.

Replaying the model's own per-step reasoning back to it triggers provider
`reasoning_extraction` refusals. Saved nodes keep the full trajectory.
"""

import pytest

from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    ExperimentNode,
    ExperimentStep,
    NodeEvaluation,
    NodeStatus,
    NodeType,
    Observation,
    Recommendation,
    ResearchStage,
    SubstagePlan,
)
from corral.agents.ai_scientist.state import TaskFormulation
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.planner import NodePlanner
from corral.agents.ai_scientist.workers.synthesizer import FinalSynthesizer

SECRET = "private-step-rationale"
FORMULATION = TaskFormulation(objective="measure", required_answer="a number")
SUBSTAGE = SubstagePlan(goal="measure", rationale="needed")


def node_with_trajectory(node_id="node_0001"):
    observation = Observation(
        action_index=0,
        purpose="measure",
        tool_name="measure",
        success=True,
        result="measured:42",
    )
    return ExperimentNode(
        id=node_id,
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis="the value is 42",
        rationale="design rationale stays",
        status=NodeStatus.SUCCESSFUL,
        observations=[observation],
        trajectory=[
            ExperimentStep(
                step_index=0,
                decision=ExperimentDecision(
                    decision="act",
                    rationale=SECRET,
                    purpose="measure",
                    tool_name="measure",
                    arguments={"value": 1},
                    expected_information="a measurement",
                ),
                observation=observation,
            ),
            ExperimentStep(
                step_index=1,
                decision=ExperimentDecision(
                    decision="finish",
                    rationale=SECRET,
                    conclusion="the value is 42",
                ),
            ),
        ],
        evaluation=NodeEvaluation(
            validity=0.9,
            task_progress=0.9,
            evidence_strength=0.9,
            information_gain=0.9,
            consistency=0.9,
            recommendation=Recommendation.CONTINUE,
            reason="consistent",
            conclusions=["the value is 42"],
        ),
    )


class CapturingModel:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, response_model, *, model=None, purpose=None):
        self.prompts.append(prompt)
        raise _Captured


class _Captured(Exception):
    pass


def _critic(model):
    return ScientificCritic(model, evaluator_model="critic", max_journal_chars=2_000)


def _planner(model):
    return NodePlanner(
        model,
        max_actions_per_node=4,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    )


CALLS = {
    "evaluate_node": lambda m, n: _critic(m).evaluate(
        task_prompt="t", formulation=FORMULATION, node=n, journal_context="{}"
    ),
    "select_stage_winner": lambda m, n: _critic(m).select_best(
        task_prompt="t",
        formulation=FORMULATION,
        stage=ResearchStage.RESEARCH,
        seed_node_id=None,
        candidates=[n, node_with_trajectory("node_0002")],
        journal_context="{}",
    ),
    "evaluate_substage": lambda m, n: _critic(m).is_substage_complete(
        task_prompt="t",
        formulation=FORMULATION,
        stage=ResearchStage.RESEARCH,
        substage=SUBSTAGE,
        stage_nodes=[n],
        journal_context="{}",
    ),
    "planner_parent": lambda m, n: _planner(m).propose(
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.REFINE,
        count=1,
        task_prompt="t",
        formulation=FORMULATION,
        tools=[],
        journal_context="{}",
        parent=n,
        siblings=[node_with_trajectory("node_0002")],
        branch_workspaces=["/workspace/b"],
    ),
    "planner_substage": lambda m, n: _planner(m).propose_substage(
        stage=ResearchStage.RESEARCH,
        task_prompt="t",
        formulation=FORMULATION,
        journal_context="{}",
        seed=n,
        previous=SUBSTAGE,
        stage_nodes=[n],
        substage_number=2,
    ),
    "final_synthesis": lambda m, n: FinalSynthesizer(
        m, max_journal_chars=2_000
    ).generate(task_prompt="t", journal_context="{}", best_nodes=[n]),
}


@pytest.mark.parametrize("call", sorted(CALLS))
def test_worker_prompts_omit_step_rationales(call):
    model = CapturingModel()
    node = node_with_trajectory()

    with pytest.raises(_Captured):
        CALLS[call](model, node)

    (prompt,) = model.prompts
    assert SECRET not in prompt
    assert "the value is 42" in prompt
    # The saved node is unchanged.
    assert [step.decision.rationale for step in node.trajectory] == [SECRET] * 2


def test_prompt_view_keeps_actions_observations_and_design():
    view = node_with_trajectory().prompt_view()

    assert view["rationale"] == "design rationale stays"
    assert view["trajectory"][0]["decision"]["purpose"] == "measure"
    assert view["trajectory"][0]["observation"]["result"] == "measured:42"
    assert view["trajectory"][1]["decision"]["conclusion"] == "the value is 42"
    assert all("rationale" not in step["decision"] for step in view["trajectory"])
