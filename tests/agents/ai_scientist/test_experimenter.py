from dataclasses import dataclass

from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    ExperimentNode,
    ExperimentTermination,
    NodeStatus,
    NodeType,
    ResearchStage,
)
from corral.agents.ai_scientist.state import TaskFormulation
from corral.agents.ai_scientist.tools import CorralExecutor
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.core.action import Action


@dataclass
class Response:
    success: bool
    result: str


class AdaptiveActionExecutor:
    def __init__(self):
        self.calls = []

    def __call__(self, action: Action):
        self.calls.append(dict(action.arguments))
        return Response(success=True, result=f"measured:{action.arguments['value']}")


class AdaptiveModel:
    def __init__(self):
        self.call_count = 0
        self.prompts = []

    def generate(self, prompt, response_model, *, model=None, purpose=None):
        assert response_model is ExperimentDecision
        self.call_count += 1
        self.prompts.append(prompt)
        if self.call_count == 1:
            value = 1
        elif self.call_count == 2:
            assert "measured:1" in prompt
            value = 2
        else:
            assert "measured:2" in prompt
            return ExperimentDecision(
                decision="finish",
                rationale="both adaptive measurements are complete",
                conclusion="Both requested measurements succeeded.",
            )
        return ExperimentDecision(
            decision="act",
            rationale="choose the next measurement from current evidence",
            purpose="measure",
            tool_name="measure",
            arguments={"value": value},
            expected_information="a measurement",
        )


TOOLS = [
    {
        "name": "measure",
        "description": "Measure a value",
        "inputSchema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    }
]


def test_experiment_worker_chooses_each_action_after_observing_the_previous_one():
    model = AdaptiveModel()
    execute_action = AdaptiveActionExecutor()
    executor = CorralExecutor(
        execute_action=execute_action,
        tools=TOOLS,
        max_tool_calls=2,
    )
    node = ExperimentNode(
        id="node_0001",
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis="The second measurement should depend on the first",
        rationale="An adaptive comparison is informative",
        experiment_goal="make two conditional measurements",
        success_criteria=["both measurements complete"],
    )
    formulation = TaskFormulation(
        objective="measure adaptively",
        required_answer="two measurements",
    )

    result = Experimenter(
        model,
        max_actions_per_node=2,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    ).execute(
        node,
        executor,
        task_prompt="Measure twice.",
        formulation=formulation,
        tools=TOOLS,
        journal_context="{}",
    )

    assert result.status == NodeStatus.SUCCESSFUL
    assert execute_action.calls == [{"value": 1}, {"value": 2}]
    assert [
        step.observation.result
        for step in result.trajectory
        if step.observation is not None
    ] == [
        "measured:1",
        "measured:2",
    ]
    assert [observation.action_index for observation in result.observations] == [
        0,
        1,
    ]
    assert result.termination_reason == ExperimentTermination.WORKER_FINISHED


def test_budget_exhausted_action_prefix_is_partial_not_successful():
    model = AdaptiveModel()
    execute_action = AdaptiveActionExecutor()
    node = ExperimentNode(
        id="node_0001",
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis="Two measurements are required",
        rationale="A single measurement cannot satisfy the comparison",
        experiment_goal="make two measurements",
        success_criteria=["both measurements complete"],
    )

    result = Experimenter(
        model,
        max_actions_per_node=2,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    ).execute(
        node,
        CorralExecutor(
            execute_action=execute_action,
            tools=TOOLS,
            max_tool_calls=1,
        ),
        task_prompt="Measure twice.",
        formulation=TaskFormulation(
            objective="measure adaptively", required_answer="two measurements"
        ),
        tools=TOOLS,
        journal_context="{}",
        action_limit=1,
    )

    assert result.status == NodeStatus.PARTIAL
    assert result.allocated_action_budget == 1
    assert result.termination_reason == ExperimentTermination.ACTION_BUDGET_EXHAUSTED
    assert len(result.plan) == 1
    assert len(result.observations) == 1
    # The rejected second action remains inspectable but is not represented as
    # part of the realized plan.
    assert len(result.trajectory) == 2
    assert result.trajectory[-1].observation is None
    assert execute_action.calls == [{"value": 1}]


class FinishingModel:
    call_count = 0

    def generate(self, prompt, response_model, *, model=None, purpose=None):
        self.call_count += 1
        return ExperimentDecision(
            decision="finish",
            rationale="the existing evidence already resolves this experiment",
            conclusion="No further measurement is useful.",
        )


def test_non_aggregation_node_that_finishes_without_evidence_is_invalid():
    execute_action = AdaptiveActionExecutor()
    node = ExperimentNode(
        id="node_0001",
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis="Already known",
        rationale="Check whether more evidence is needed",
        experiment_goal="resolve the claim",
    )

    result = Experimenter(
        FinishingModel(),
        max_actions_per_node=2,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    ).execute(
        node,
        CorralExecutor(
            execute_action=execute_action,
            tools=TOOLS,
            max_tool_calls=2,
        ),
        task_prompt="Resolve the claim.",
        formulation=TaskFormulation(objective="resolve", required_answer="claim"),
        tools=TOOLS,
        journal_context="{}",
    )

    assert result.status == NodeStatus.INVALID
    assert result.worker_conclusion == "No further measurement is useful."
    assert execute_action.calls == []


class RecoveringModel:
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt, response_model, *, model=None, purpose=None):
        self.call_count += 1
        if self.call_count == 1:
            value = "invalid"
        elif self.call_count == 2:
            assert "Invalid tool arguments" in prompt
            value = 3
        else:
            return ExperimentDecision(
                decision="finish",
                rationale="the corrected measurement succeeded",
                conclusion="A valid measurement was obtained.",
            )
        return ExperimentDecision(
            decision="act",
            rationale="correct the action from the latest observation",
            purpose="measure",
            tool_name="measure",
            arguments={"value": value},
            expected_information="a valid measurement",
        )


def test_experiment_worker_can_recover_from_a_failed_action_within_the_node():
    execute_action = AdaptiveActionExecutor()
    node = ExperimentNode(
        id="node_0001",
        stage=ResearchStage.RESEARCH,
        node_type=NodeType.RESEARCH,
        hypothesis="A valid measurement is obtainable",
        rationale="Correct malformed parameters from tool feedback",
        experiment_goal="obtain one valid measurement",
    )
    executor = CorralExecutor(
        execute_action=execute_action,
        tools=TOOLS,
        max_tool_calls=2,
        stop_on_error=False,
    )

    result = Experimenter(
        RecoveringModel(),
        max_actions_per_node=2,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    ).execute(
        node,
        executor,
        task_prompt="Measure once.",
        formulation=TaskFormulation(objective="measure", required_answer="value"),
        tools=TOOLS,
        journal_context="{}",
    )

    assert result.status == NodeStatus.SUCCESSFUL
    assert [observation.success for observation in result.observations] == [
        False,
        True,
    ]
    assert execute_action.calls == [{"value": 3}]
