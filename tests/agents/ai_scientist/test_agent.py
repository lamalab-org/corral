import re
import threading
import time
from dataclasses import dataclass

from corral.agents import AIScientistAgent, AIScientistConfig
from corral.agents.ai_scientist.search.nodes import (
    NodeEvaluation,
    NodeProposal,
    NodeType,
    PlannedAction,
    PlanningBatch,
    Recommendation,
    ResearchStage,
)
from corral.agents.ai_scientist.state import Hypothesis, TaskFormulation
from corral.agents.ai_scientist.workers.synthesizer import FinalAnswer


@dataclass
class ToolResponse:
    success: bool
    result: str | None = None
    error: str | None = None


class FakeRouter:
    def __init__(self):
        self.tool_calls = []

    def get_task_prompt(self, task_id):
        return "Determine the measured value and return one integer."

    def get_mcp_tool_schema(self, task_id):
        return {
            "tools": [
                {
                    "name": "measure",
                    "description": "Return a measurement",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                }
            ],
            "mcp_schema_sha256": "digest",
        }

    def execute_tool(self, task_id, tool_name, arguments):
        self.tool_calls.append((task_id, tool_name, arguments))
        return ToolResponse(success=True, result="42")


class ScriptedModel:
    def __init__(
        self,
        actions_per_plan=1,
        *,
        validity=0.95,
        evidence_strength=0.9,
        recommendation=Recommendation.FINALIZE,
    ):
        self.call_count = 0
        self.token_count = 0
        self.purposes = []
        self.records = []
        self.actions_per_plan = actions_per_plan
        self.validity = validity
        self.evidence_strength = evidence_strength
        self.recommendation = recommendation

    def generate(
        self, prompt, response_model, *, model=None, purpose="scientific_worker"
    ):
        self.call_count += 1
        self.token_count += 10
        self.purposes.append(purpose)
        self.records.append((purpose, prompt))
        if response_model is TaskFormulation:
            return TaskFormulation(
                objective="measure the value",
                required_answer="one integer",
                candidate_hypotheses=[
                    Hypothesis(statement="The value is 42", rationale="candidate")
                ],
                observable_quantities=["measurement"],
                possible_experiments=["call measure"],
                success_criteria=["replicated value"],
                tunable_parameters=[],
            )
        if response_model is PlanningBatch:
            node_type = next(
                (
                    item
                    for item in NodeType
                    if item.value in purpose.removeprefix("plan_")
                ),
                NodeType.RESEARCH,
            )
            match = re.search(r"Design (\d+)", prompt)
            count = int(match.group(1)) if match else 1
            plan = []
            if node_type != NodeType.AGGREGATION:
                plan = [
                    PlannedAction(
                        purpose="measure",
                        tool_name="measure",
                        arguments={"value": index},
                        expected_information="the value",
                    )
                    for index in range(self.actions_per_plan)
                ]
            return PlanningBatch(
                proposals=[
                    NodeProposal(
                        node_type=node_type,
                        hypothesis=f"proposal {index}",
                        rationale="discriminating experiment",
                        plan=plan,
                    )
                    for index in range(count)
                ]
            )
        if response_model is NodeEvaluation:
            return NodeEvaluation(
                validity=self.validity,
                task_progress=0.95,
                evidence_strength=self.evidence_strength,
                information_gain=0.8,
                consistency=0.95,
                recommendation=self.recommendation,
                reason="the tool repeatedly returned 42",
                conclusions=["The value is 42"],
            )
        if response_model is FinalAnswer:
            return FinalAnswer(final_answer="42")
        raise AssertionError(response_model)


class ConcurrentProbeModel(ScriptedModel):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(
        self, prompt, response_model, *, model=None, purpose="scientific_worker"
    ):
        is_parallel_worker = response_model in {PlanningBatch, NodeEvaluation}
        if is_parallel_worker:
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            time.sleep(0.03)
        try:
            return super().generate(
                prompt, response_model, model=model, purpose=purpose
            )
        finally:
            if is_parallel_worker:
                with self._lock:
                    self._active -= 1


def test_agent_runs_all_scientific_stages_without_using_scorer(tmp_path):
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=2,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=2,
        verification_node_budget=4,
        verification_min_nodes=4,
        max_nodes=10,
        max_tool_calls=10,
        max_llm_calls=20,
        trace_path=tmp_path,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "scientific-task")

    assert result.answer == "42"
    assert agent.requires_answer_extraction is False
    assert agent.last_state.current_stage == ResearchStage.COMPLETE
    assert len(agent.last_state.tree.by_stage(ResearchStage.PRELIMINARY)) == 2
    assert len(agent.last_state.tree.by_stage(ResearchStage.RESEARCH)) == 1
    verification = agent.last_state.tree.by_stage(ResearchStage.VERIFICATION)
    assert [node.node_type for node in verification] == [
        NodeType.ABLATION,
        NodeType.REPLICATION,
        NodeType.COUNTERFACTUAL,
        NodeType.AGGREGATION,
    ]
    assert len(router.tool_calls) == 6
    assert model.call_count == 16
    assert (tmp_path / "scientific-task.jsonl").is_file()

    preliminary = agent.last_state.tree.by_stage(ResearchStage.PRELIMINARY)
    assert [node.branch_workspace for node in preliminary] == [
        "ai_scientist/node_0001/",
        "ai_scientist/node_0002/",
    ]
    planning_prompts = [
        prompt for purpose, prompt in model.records if purpose.startswith("plan_")
    ]
    assert any("ai_scientist/node_0001/" in prompt for prompt in planning_prompts)
    assert any("ai_scientist/node_0002/" in prompt for prompt in planning_prompts)


def test_agent_stops_new_nodes_at_tool_budget_and_keeps_partial_plan_valid():
    model = ScriptedModel(actions_per_plan=2)
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=3,
        max_tool_calls=1,
        max_llm_calls=10,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "budgeted-task")

    assert result.answer == "42"
    assert len(router.tool_calls) == 1
    assert len(agent.last_state.tree.nodes) == 1
    assert agent.last_state.tree.nodes[0].status.value == "successful"
    assert len(agent.last_state.tree.nodes[0].plan) == 1
    assert model.call_count == 4


def test_reusable_injected_model_gets_a_fresh_per_run_llm_budget():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=3,
        max_tool_calls=3,
        max_llm_calls=8,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    first = agent.run_agent(FakeRouter(), "first-task")
    first_run_calls = model.call_count
    second = agent.run_agent(FakeRouter(), "second-task")

    assert first.answer == second.answer == "42"
    assert first_run_calls > 0
    assert model.call_count == first_run_calls * 2
    assert len(agent.last_state.tree.nodes) == 3


def test_agent_stops_expansion_after_total_llm_token_budget():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=3,
        max_tool_calls=3,
        max_llm_calls=10,
        max_llm_tokens=25,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "token-budgeted-task")

    assert result.answer == "42"
    assert len(router.tool_calls) == 1
    assert len(agent.last_state.tree.nodes) == 1
    # Formulation + planning + evaluation crossed the ceiling; only the
    # separately reserved final-synthesis call was allowed afterward.
    assert agent.last_state.llm_calls == 4
    assert agent.last_state.llm_tokens == 40


def test_low_validity_tool_success_is_marked_invalid_and_not_ranked_best():
    model = ScriptedModel(validity=0.1, recommendation=Recommendation.ABANDON)
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=3,
        max_tool_calls=1,
        max_llm_calls=8,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    result = agent.run_agent(FakeRouter(), "invalid-evidence-task")

    assert result.answer == "42"
    node = agent.last_state.tree.nodes[0]
    assert node.status.value == "invalid"
    assert agent.last_state.best_nodes == []
    assert agent.last_state.journal.failed_experiments[0].errors == [
        "the tool repeatedly returned 42"
    ]


def test_critic_debug_recommendation_drives_the_next_node_type():
    model = ScriptedModel(
        validity=0.9,
        evidence_strength=0.1,
        recommendation=Recommendation.DEBUG,
    )
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=2,
        max_llm_calls=8,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "debug-recommendation-task")

    assert [node.node_type for node in agent.last_state.tree.nodes] == [
        NodeType.DRAFT,
        NodeType.DEBUG,
    ]


def test_llm_planning_and_evaluation_overlap_for_independent_roots():
    model = ConcurrentProbeModel()
    config = AIScientistConfig(
        initial_drafts=2,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=2,
        max_llm_calls=10,
        parallel_llm_workers=2,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "parallel-llm-task")

    assert model.max_active >= 2


def test_parallel_root_planning_preserves_the_reserved_final_llm_call():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=3,
        preliminary_node_budget=3,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=3,
        max_tool_calls=3,
        max_llm_calls=4,
        parallel_llm_workers=3,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    result = agent.run_agent(FakeRouter(), "parallel-budget-task")

    assert result.answer == "42"
    assert len(agent.last_state.tree.nodes) == 1
    assert model.call_count == 4
