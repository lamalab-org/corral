from types import SimpleNamespace

import pytest

from corral.agents.ai_scientist.config import AIScientistConfig
from corral.agents.ai_scientist.journal import JSONLTraceWriter
from corral.agents.ai_scientist.manager import ExperimentManager
from corral.agents.ai_scientist.prompts import load_prompt
from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    NodeEvaluation,
    NodeProposal,
    NodeType,
    PlanningBatch,
    ResearchStage,
)
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.state import (
    ScientistState,
    StageProgress,
    TaskFormulation,
)
from corral.agents.ai_scientist.tools import BranchSessionHandle, ExecutionPool
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner
from corral.agents.ai_scientist.workers.synthesizer import FinalAnswer, FinalSynthesizer


@pytest.mark.parametrize(
    "name", ["preliminary", "tuning", "research", "ablation", "experiment_step"]
)
def test_scientist_prompts_do_not_override_the_task_path_contract(name):
    prompt = load_prompt(name)

    assert "paths relative" not in prompt
    assert "Execution-relative" not in prompt
    assert "CORRAL_WORKSPACE" not in prompt


def test_final_synthesis_projects_the_canonical_workspace(tmp_path):
    prompts = []

    def generate(prompt, response_model, **kwargs):
        prompts.append(prompt)
        return FinalAnswer(final_answer="/workspace/result.json")

    answer = FinalSynthesizer(
        SimpleNamespace(generate=generate), max_journal_chars=2_000
    ).generate(
        task_prompt="Report the result path.",
        journal_context="{}",
        best_nodes=[],
        canonical_workspace=str(tmp_path / "physical-scoring-workspace"),
    )

    assert str(tmp_path) not in prompts[0]
    assert "Canonical scored workspace:\n/workspace" in prompts[0]
    assert answer == "/workspace/result.json"


def test_scientist_projects_branch_paths_without_changing_isolated_executions(tmp_path):
    prompts = []
    proposal = NodeProposal(
        node_type=NodeType.RESEARCH,
        hypothesis="A measurement is reproducible",
        rationale="Compare independent measurements",
        experiment_goal="Measure once",
    )

    class Model:
        call_count = 0

        def generate(self, prompt, response_model, *, purpose=None, **kwargs):
            self.call_count += 1
            prompts.append(prompt)
            if response_model is PlanningBatch:
                return PlanningBatch(proposals=[proposal])
            if response_model is NodeEvaluation:
                return NodeEvaluation(
                    validity=1,
                    task_progress=1,
                    evidence_strength=1,
                    information_gain=1,
                    consistency=1,
                    recommendation="finalize",
                    reason="Measured successfully",
                )
            if purpose.endswith("_step_1"):
                return ExperimentDecision(
                    decision="act",
                    rationale="Measure once",
                    purpose="measurement",
                    tool_name="measure",
                    arguments={},
                    expected_information="A measurement",
                )
            return ExperimentDecision(decision="finish", rationale="Measured")

    class Sessions:
        def __init__(self):
            self.workspaces = []

        def create_branch(self):
            workspace = tmp_path / f"private-branch-{len(self.workspaces)}"
            workspace.mkdir()
            self.workspaces.append(str(workspace))
            return BranchSessionHandle(
                execution_id=workspace.name,
                workspace=str(workspace),
                execute=lambda action: SimpleNamespace(success=True, result="1"),
            )

        def close_branch(self, execution_id):
            pass

    tools = [{"name": "measure", "inputSchema": {"type": "object"}}]
    model = Model()
    planner = NodePlanner(
        model,
        max_actions_per_node=1,
        max_journal_chars=2_000,
        max_tool_schema_chars=2_000,
    )
    manager = ExperimentManager(
        config=AIScientistConfig(enable_visual_feedback=False),
        max_llm_calls=20,
        model=model,
        planner=planner,
        experimenter=Experimenter(
            model,
            max_actions_per_node=1,
            max_journal_chars=2_000,
            max_tool_schema_chars=2_000,
        ),
        critic=ScientificCritic(model, evaluator_model="test", max_journal_chars=2_000),
        selector=TreeSelector(),
        trace=JSONLTraceWriter(None, "workspace-test"),
    )
    sessions = Sessions()
    pool = ExecutionPool(sessions=sessions, tools=tools, max_tool_calls=4)
    state = ScientistState(
        task_prompt="Use the environment's public /workspace paths.",
        tools=tools,
        formulation=TaskFormulation(objective="Measure", required_answer="Value"),
    )
    stage = ResearchStage.RESEARCH
    state.current_stage = stage
    state.stages[stage] = StageProgress(stage=stage)
    candidates = manager._propose(
        state, stage=stage, node_type=NodeType.RESEARCH, count=1, parent=None
    )
    nodes = manager._execute_candidates(state, pool, candidates, parent=None)
    replications = manager._run_deterministic_replications(
        state, pool, stage, nodes[0], reserve_aggregation=0
    )
    manager._propose(
        state, stage=stage, node_type=NodeType.RESEARCH, count=1, parent=nodes[0]
    )

    assert len(nodes) == len(replications) == 1
    assert all(node.branch_workspace == "/workspace" for node in nodes + replications)
    assert len(set(sessions.workspaces)) == 2
    assert [branch.workspace for branch in pool.active.values()] == sessions.workspaces
    assert all(str(tmp_path) not in prompt for prompt in prompts)
    assert '"branch_workspace": "/workspace"' in prompts[1]
    assert '["/workspace"]' in prompts[0]
