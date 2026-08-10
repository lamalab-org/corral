import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from corral.agents import (
    AIScientistAgent,
    AIScientistConfig,
    SakanaAIScientistConfig,
)
from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    NodeEvaluation,
    NodeProposal,
    NodeType,
    PlanningBatch,
    Recommendation,
    ResearchStage,
    StageWinnerSelection,
    SubstageCompletion,
    SubstagePlan,
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
        self.trial_tool_calls = []
        self.created_trials = []
        self.closed_trials = []

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

    def create_trial(self, task_id):
        trial_runtime_id = f"trial-{len(self.created_trials) + 1}"
        self.created_trials.append(trial_runtime_id)
        return {
            "trial_runtime_id": trial_runtime_id,
            "workspace": f"/fake/{trial_runtime_id}",
        }

    def for_trial(self, trial_runtime_id, task_id, *, verbosity=None, workspace=None):
        return FakeTrialRouter(self, trial_runtime_id)

    def close_trial(self, trial_runtime_id):
        self.closed_trials.append(trial_runtime_id)

    def execute_trial_tool(self, trial_runtime_id, task_id, tool_name, arguments):
        call = (task_id, tool_name, arguments)
        self.tool_calls.append(call)
        self.trial_tool_calls.append((trial_runtime_id, *call))
        return ToolResponse(success=True, result="42")


class FakeTrialRouter:
    def __init__(self, parent, trial_runtime_id):
        self.parent = parent
        self.trial_runtime_id = trial_runtime_id

    def execute_tool(self, task_id, tool_name, arguments):
        return self.parent.execute_trial_tool(
            self.trial_runtime_id, task_id, tool_name, arguments
        )


class ConcurrentToolRouter(FakeRouter):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def execute_trial_tool(self, trial_runtime_id, task_id, tool_name, arguments):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(0.03)
        try:
            return super().execute_trial_tool(
                trial_runtime_id, task_id, tool_name, arguments
            )
        finally:
            with self._lock:
                self._active -= 1


class ArtifactRouter(FakeRouter):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.trial_runtime_id = "outer-trial"
        self.trial_workspace = str(root / "canonical")
        Path(self.trial_workspace).mkdir()
        self.workspaces = {self.trial_runtime_id: self.trial_workspace}

    def create_trial(self, task_id):
        descriptor = super().create_trial(task_id)
        workspace = self.root / descriptor["trial_runtime_id"]
        workspace.mkdir()
        descriptor["workspace"] = str(workspace)
        self.workspaces[descriptor["trial_runtime_id"]] = str(workspace)
        return descriptor

    def execute_trial_tool(self, trial_runtime_id, task_id, tool_name, arguments):
        artifact = Path(self.workspaces[trial_runtime_id]) / "results" / "model.json"
        artifact.parent.mkdir()
        artifact.write_text('{"value": 42}', encoding="utf-8")
        return super().execute_trial_tool(
            trial_runtime_id, task_id, tool_name, arguments
        )

    def promote_trial_artifacts(
        self, source_trial_runtime_id, destination_trial_runtime_id
    ):
        source = Path(self.workspaces[source_trial_runtime_id])
        destination = Path(self.workspaces[destination_trial_runtime_id])
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return {
            "files": [
                str(item.relative_to(source))
                for item in source.rglob("*")
                if item.is_file()
            ]
        }


class ScriptedModel:
    def __init__(
        self,
        actions_per_plan=1,
        *,
        validity=0.95,
        evidence_strength=0.9,
        recommendation=Recommendation.FINALIZE,
        substage_complete=True,
    ):
        self.call_count = 0
        self.token_count = 0
        self.purposes = []
        self.records = []
        self.actions_per_plan = actions_per_plan
        self.validity = validity
        self.evidence_strength = evidence_strength
        self.recommendation = recommendation
        self.substage_complete = substage_complete

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
            slot_match = re.search(
                r"Proposal slots?.*?\[\s*(\d+)", prompt, flags=re.DOTALL
            )
            proposal_offset = int(slot_match.group(1)) - 1 if slot_match else 0
            return PlanningBatch(
                proposals=[
                    NodeProposal(
                        node_type=node_type,
                        hypothesis=f"proposal {proposal_offset + index}",
                        rationale="discriminating experiment",
                        experiment_goal="measure the value",
                        success_criteria=["obtain a measurement"],
                    )
                    for index in range(count)
                ]
            )
        if response_model is ExperimentDecision:
            step = int(purpose.rsplit("_", 1)[-1])
            if step > self.actions_per_plan:
                return ExperimentDecision(
                    decision="finish",
                    rationale="the requested measurements are complete",
                    conclusion="The measured value is 42",
                )
            return ExperimentDecision(
                decision="act",
                rationale="take the next measurement",
                purpose="measure",
                tool_name="measure",
                arguments={"value": step - 1},
                expected_information="the value",
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
        if response_model is StageWinnerSelection:
            candidate_ids = re.findall(r'"id": "(node_\d+)"', prompt)
            return StageWinnerSelection(
                selected_node_id=candidate_ids[-1],
                reason="the last candidate is the strongest controlled follow-up",
                candidate_comparison=["Compared all valid candidates directly."],
            )
        if response_model is SubstagePlan:
            return SubstagePlan(
                goal="Resolve the largest remaining uncertainty.",
                rationale="The preceding substage did not meet its criteria.",
                objectives=["Run a materially different measurement."],
                completion_criteria=["Obtain valid discriminating evidence."],
            )
        if response_model is SubstageCompletion:
            return SubstageCompletion(
                complete=self.substage_complete,
                reason=(
                    "Valid observations satisfy the current agenda."
                    if self.substage_complete
                    else "The observations do not satisfy the agenda yet."
                ),
                satisfied_criteria=(
                    ["Obtain valid discriminating evidence."]
                    if self.substage_complete
                    else []
                ),
                unmet_criteria=(
                    []
                    if self.substage_complete
                    else ["Obtain valid discriminating evidence."]
                ),
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
        is_parallel_worker = response_model in {
            PlanningBatch,
            ExperimentDecision,
            NodeEvaluation,
        }
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


class SeedPreferringModel(ScriptedModel):
    def generate(
        self, prompt, response_model, *, model=None, purpose="scientific_worker"
    ):
        if response_model is StageWinnerSelection and purpose == "select_best_research":
            self.call_count += 1
            self.token_count += 10
            self.purposes.append(purpose)
            self.records.append((purpose, prompt))
            seed_match = re.search(r"Inherited seed node id:\s*(node_\d+)", prompt)
            assert seed_match is not None
            return StageWinnerSelection(
                selected_node_id=seed_match.group(1),
                reason="none of the research candidates improves the inherited seed",
                candidate_comparison=["The seed remains strongest."],
            )
        return super().generate(prompt, response_model, model=model, purpose=purpose)


class TunableScriptedModel(ScriptedModel):
    def generate(
        self, prompt, response_model, *, model=None, purpose="scientific_worker"
    ):
        result = super().generate(
            prompt,
            response_model,
            model=model,
            purpose=purpose,
        )
        if response_model is TaskFormulation:
            return result.model_copy(update={"tunable_parameters": ["temperature"]})
        return result


class VisualScriptedModel(ScriptedModel):
    def __init__(self):
        super().__init__()
        self.multimodal_paths = []

    def generate_multimodal(
        self,
        prompt,
        response_model,
        *,
        image_paths,
        model=None,
        purpose="multimodal_scientific_worker",
    ):
        self.multimodal_paths.append(list(image_paths))
        evaluation = self.generate(
            prompt,
            response_model,
            model=model,
            purpose=purpose,
        )
        return evaluation.model_copy(
            update={"visual_feedback": ["The plotted measurement converged to 42."]}
        )


class VisualArtifactRouter(ArtifactRouter):
    def execute_trial_tool(self, trial_runtime_id, task_id, tool_name, arguments):
        response = super().execute_trial_tool(
            trial_runtime_id, task_id, tool_name, arguments
        )
        plot = Path(self.workspaces[trial_runtime_id]) / "results" / "curve.png"
        plot.write_bytes(b"\x89PNG\r\n\x1a\nvisual-test")
        return response


def test_agent_runs_all_scientific_stages_without_using_scorer(tmp_path):
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=2,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=2,
        verification_node_budget=4,
        verification_min_nodes=4,
        adaptive_substages=False,
        stage_boundary_replications=1,
        max_nodes=20,
        max_tool_calls=20,
        max_llm_calls=80,
        max_actions_per_node=1,
        trace_path=tmp_path,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "scientific-task")

    assert result.answer == "42"
    assert agent.requires_answer_extraction is False
    assert agent.last_state.current_stage == ResearchStage.COMPLETE
    assert (
        len(
            agent.last_state.tree.by_stage(
                ResearchStage.PRELIMINARY, include_boundary=False
            )
        )
        == 2
    )
    assert (
        len(
            agent.last_state.tree.by_stage(
                ResearchStage.RESEARCH, include_boundary=False
            )
        )
        == 2
    )
    verification = agent.last_state.tree.by_stage(
        ResearchStage.VERIFICATION, include_boundary=False
    )
    assert [node.node_type for node in verification] == [
        NodeType.ABLATION,
        NodeType.ABLATION,
        NodeType.ABLATION,
        NodeType.COUNTERFACTUAL,
    ]
    assert len(router.tool_calls) == 11
    assert agent.last_state.scientific_tool_calls == 11
    assert agent.last_state.replay_tool_calls == 0
    assert agent.last_state.trial_runtimes_created == 11
    assert router.closed_trials == router.created_trials
    assert model.call_count == 49
    assert (tmp_path / "scientific-task.jsonl").is_file()

    preliminary = agent.last_state.tree.by_stage(
        ResearchStage.PRELIMINARY, include_boundary=False
    )
    assert [node.branch_workspace for node in preliminary] == [
        "/fake/trial-1",
        "/fake/trial-2",
    ]
    assert len({node.branch_id for node in preliminary}) == 2
    assert len({node.trial_runtime_id for node in preliminary}) == 2
    preliminary_progress = agent.last_state.stage_progress(ResearchStage.PRELIMINARY)
    research_progress = agent.last_state.stage_progress(ResearchStage.RESEARCH)
    verification_progress = agent.last_state.stage_progress(ResearchStage.VERIFICATION)
    assert preliminary_progress.best_node_id == "node_0002"
    assert research_progress.seed_node_id == preliminary_progress.best_node_id
    assert research_progress.best_node_id == "node_0006"
    assert verification_progress.seed_node_id == research_progress.best_node_id
    assert all(
        node.stage_seed_id == preliminary_progress.best_node_id
        for node in agent.last_state.tree.by_stage(
            ResearchStage.RESEARCH, include_boundary=False
        )
    )
    assert all(
        node.boundary_validation
        for stage in (
            ResearchStage.PRELIMINARY,
            ResearchStage.RESEARCH,
            ResearchStage.VERIFICATION,
        )
        for node in agent.last_state.tree.by_stage(stage)
        if node.node_type in {NodeType.REPLICATION, NodeType.AGGREGATION}
    )
    assert sum(purpose.startswith("select_best_") for purpose in model.purposes) == 3
    planning_prompts = [
        prompt for purpose, prompt in model.records if purpose.startswith("plan_")
    ]
    assert all("Trial-relative workspace" in prompt for prompt in planning_prompts)
    assert all("Existing child experiments" in prompt for prompt in planning_prompts)
    assert all("Proposal slot" in prompt for prompt in planning_prompts)


def test_boundary_validation_does_not_consume_the_search_node_cap():
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        stage_boundary_replications=1,
        max_validation_nodes=2,
        max_nodes=1,
        max_tool_calls=2,
        max_llm_calls=12,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=ScriptedModel())

    agent.run_agent(FakeRouter(), "separate-node-budgets-task")

    search_nodes = [
        node for node in agent.last_state.tree.nodes if not node.boundary_validation
    ]
    validation_nodes = [
        node for node in agent.last_state.tree.nodes if node.boundary_validation
    ]
    assert len(search_nodes) == 1
    assert [node.node_type for node in validation_nodes] == [
        NodeType.REPLICATION,
        NodeType.AGGREGATION,
    ]


def test_tuning_and_verification_use_fixed_stage_baselines():
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=4,
        research_node_budget=1,
        verification_node_budget=4,
        verification_min_nodes=4,
        adaptive_substages=False,
        stage_boundary_replications=0,
        max_search_nodes=10,
        max_tool_calls=10,
        max_llm_calls=64,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=TunableScriptedModel())

    agent.run_agent(FakeRouter(), "fixed-stage-baselines-task")

    tuning_progress = agent.last_state.stage_progress(ResearchStage.TUNING)
    tuning_nodes = agent.last_state.tree.by_stage(
        ResearchStage.TUNING,
        include_boundary=False,
    )
    assert len(tuning_nodes) == 4
    assert {node.parent_id for node in tuning_nodes} == {tuning_progress.seed_node_id}
    assert {node.node_type for node in tuning_nodes} == {NodeType.PARAMETER_SEARCH}

    verification_progress = agent.last_state.stage_progress(ResearchStage.VERIFICATION)
    verification_nodes = agent.last_state.tree.by_stage(
        ResearchStage.VERIFICATION,
        include_boundary=False,
    )
    assert len(verification_nodes) == 4
    assert {node.parent_id for node in verification_nodes} == {
        verification_progress.seed_node_id
    }


def test_sakana_profile_runs_full_stage_four_budget_with_only_ablations():
    config = SakanaAIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=4,
        verification_min_nodes=1,
        adaptive_substages=False,
        stage_boundary_replications=0,
        max_search_nodes=6,
        max_tool_calls=6,
        max_llm_calls=40,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=ScriptedModel())

    agent.run_agent(FakeRouter(), "sakana-verification-task")

    verification = agent.last_state.tree.by_stage(
        ResearchStage.VERIFICATION,
        include_boundary=False,
    )
    seed_id = agent.last_state.stage_progress(ResearchStage.VERIFICATION).seed_node_id
    assert len(verification) == 4
    assert {node.node_type for node in verification} == {NodeType.ABLATION}
    assert {node.parent_id for node in verification} == {seed_id}


def test_agent_marks_budget_truncated_node_partial_and_does_not_rank_it():
    model = ScriptedModel(actions_per_plan=2)
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        stage_boundary_replications=0,
        max_nodes=3,
        max_tool_calls=1,
        max_llm_calls=8,
        max_actions_per_node=3,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "budgeted-task")

    assert result.answer == "42"
    assert len(router.tool_calls) == 1
    assert len(agent.last_state.tree.nodes) == 1
    node = agent.last_state.tree.nodes[0]
    assert node.status.value == "partial"
    assert node.allocated_action_budget == 1
    assert node.termination_reason.value == "action_budget_exhausted"
    assert len(node.plan) == 1
    assert agent.last_state.best_nodes == []
    assert agent.last_state.journal.claims == []
    assert model.call_count == 6


def test_parallel_nodes_reserve_shared_tool_budget_before_execution():
    model = ScriptedModel(actions_per_plan=2)
    config = AIScientistConfig(
        initial_drafts=2,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=3,
        max_llm_calls=14,
        max_actions_per_node=2,
        parallel_experiment_workers=2,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "parallel-budget-reservation-task")

    assert [len(node.plan) for node in agent.last_state.tree.nodes] == [2, 1]
    assert [node.status.value for node in agent.last_state.tree.nodes] == [
        "successful",
        "partial",
    ]
    assert agent.last_state.scientific_tool_calls == 3
    assert agent.last_state.replay_tool_calls == 0


class FourStepModel(ScriptedModel):
    def __init__(self):
        super().__init__()
        self.actions_taken = 0

    def generate(
        self, prompt, response_model, *, model=None, purpose="scientific_worker"
    ):
        if response_model is ExperimentDecision:
            self.call_count += 1
            self.token_count += 10
            self.purposes.append(purpose)
            self.records.append((purpose, prompt))
            if self.actions_taken < 4:
                next_action = self.actions_taken + 1
                if "You have 0 tool action(s) left" not in prompt:
                    self.actions_taken = next_action
                return ExperimentDecision(
                    decision="act",
                    rationale="complete the next workflow step",
                    purpose=f"workflow step {next_action}",
                    tool_name="measure",
                    arguments={"value": next_action},
                    expected_information="the next intermediate result",
                )
            return ExperimentDecision(
                decision="finish",
                rationale="all four workflow steps are complete",
                conclusion="The four-step workflow completed.",
            )
        return super().generate(prompt, response_model, model=model, purpose=purpose)


def test_partial_node_is_continued_in_the_same_physical_branch():
    model = FourStepModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=4,
        max_llm_calls=14,
        max_actions_per_node=3,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "four-step-task")

    first, continuation = agent.last_state.tree.nodes
    assert first.status.value == "partial"
    assert continuation.node_type == NodeType.CONTINUE
    assert continuation.status.value == "successful"
    assert continuation.parent_id == first.id
    assert continuation.branch_id == first.branch_id
    assert continuation.hypothesis == first.hypothesis
    assert len(first.executed_actions) == 3
    assert len(continuation.executed_actions) == 1
    continuation_prompt = next(
        prompt
        for purpose, prompt in model.records
        if purpose == f"experiment_{continuation.id}_step_1"
    )
    assert "Prior partial checkpoint being continued" in continuation_prompt
    assert '"result": "42"' in continuation_prompt


def test_stage_four_does_not_substitute_aggregation_when_tools_are_exhausted():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=1,
        max_llm_calls=8,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "zero-tool-aggregation-task")

    assert [node.node_type for node in agent.last_state.tree.nodes] == [NodeType.DRAFT]
    assert (
        agent.last_state.stage_progress(ResearchStage.PRELIMINARY).replication_node_ids
        == []
    )
    assert agent.last_state.scientific_tool_calls == 1
    assert agent.last_state.replay_tool_calls == 0


def test_agent_promotes_winning_artifacts_into_the_scored_trial(tmp_path):
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=1,
        max_tool_calls=1,
        max_llm_calls=6,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=ScriptedModel())
    router = ArtifactRouter(tmp_path)

    agent.run_agent(router, "artifact-task")

    promoted = Path(router.trial_workspace) / "results" / "model.json"
    assert promoted.read_text(encoding="utf-8") == '{"value": 42}'
    assert agent.last_state.promoted_artifacts == ["results/model.json"]
    assert agent.last_state.artifact_destination_workspace == router.trial_workspace
    assert router.closed_trials == router.created_trials


def test_reusable_injected_model_gets_a_fresh_per_run_llm_budget():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        stage_boundary_replications=0,
        max_nodes=3,
        max_tool_calls=4,
        max_llm_calls=16,
        max_actions_per_node=1,
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
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = FakeRouter()

    result = agent.run_agent(router, "token-budgeted-task")

    assert result.answer == "42"
    assert len(router.tool_calls) == 1
    assert len(agent.last_state.tree.nodes) == 1
    # Formulation + planning + adaptive action + explicit completion +
    # evaluation crossed the token ceiling; final synthesis remained reserved.
    assert agent.last_state.llm_calls == 6
    assert agent.last_state.llm_tokens == 60


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
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    result = agent.run_agent(FakeRouter(), "invalid-evidence-task")

    assert result.answer == "42"
    node = agent.last_state.tree.nodes[0]
    assert node.status.value == "invalid"
    assert agent.last_state.best_nodes == []
    assert set(agent.last_state.stages) == {ResearchStage.PRELIMINARY}
    assert agent.last_state.experimental_search_terminated_reason is not None
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
        max_llm_calls=10,
        max_actions_per_node=1,
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
        max_actions_per_node=1,
        parallel_llm_workers=2,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "parallel-llm-task")

    assert model.max_active >= 2


def test_tool_execution_overlaps_for_isolated_root_trials():
    model = ScriptedModel()
    config = AIScientistConfig(
        initial_drafts=2,
        preliminary_node_budget=2,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        max_nodes=2,
        max_tool_calls=2,
        max_llm_calls=9,
        max_actions_per_node=1,
        parallel_llm_workers=1,
        parallel_experiment_workers=2,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)
    router = ConcurrentToolRouter()

    agent.run_agent(router, "parallel-tool-task")

    assert router.max_active == 2
    assert agent.last_state.peak_simultaneous_trials == 2


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
        max_llm_calls=6,
        max_actions_per_node=3,
        parallel_llm_workers=3,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    result = agent.run_agent(FakeRouter(), "parallel-budget-task")

    assert result.answer == "42"
    assert len(agent.last_state.tree.nodes) == 1
    assert len(agent.last_state.tree.nodes[0].plan) == 1
    assert model.call_count == 6


def test_research_keeps_searching_when_comparison_prefers_the_stage_seed():
    model = SeedPreferringModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=3,
        verification_node_budget=1,
        verification_min_nodes=1,
        candidates_per_expansion=1,
        adaptive_substages=False,
        stage_boundary_replications=0,
        max_nodes=4,
        max_tool_calls=4,
        max_llm_calls=24,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "seed-remains-best-task")

    research_nodes = agent.last_state.tree.by_stage(
        ResearchStage.RESEARCH, include_boundary=False
    )
    progress = agent.last_state.stage_progress(ResearchStage.RESEARCH)
    assert len(research_nodes) == 3
    assert progress.best_node_id == progress.seed_node_id
    assert progress.improved_over_seed is False
    assert progress.completion_criteria_met is False


def test_manager_creates_evidence_dependent_substages():
    model = ScriptedModel(
        evidence_strength=0.1,
        recommendation=Recommendation.CONTINUE,
    )
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=3,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        candidates_per_expansion=1,
        nodes_per_substage=1,
        max_substages_per_stage=3,
        stage_boundary_replications=0,
        max_nodes=3,
        max_tool_calls=3,
        max_llm_calls=24,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "adaptive-substage-task")

    progress = agent.last_state.stage_progress(ResearchStage.PRELIMINARY)
    assert [substage.id for substage in progress.substages] == [
        "preliminary.1",
        "preliminary.2",
        "preliminary.3",
    ]
    assert [len(substage.node_ids) for substage in progress.substages] == [1, 1, 1]
    assert [
        purpose
        for purpose in model.purposes
        if purpose.startswith("manage_preliminary_substage_")
    ] == ["manage_preliminary_substage_2", "manage_preliminary_substage_3"]


def test_node_count_alone_does_not_advance_an_incomplete_substage():
    model = ScriptedModel(
        evidence_strength=0.1,
        recommendation=Recommendation.CONTINUE,
        substage_complete=False,
    )
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=3,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        candidates_per_expansion=1,
        nodes_per_substage=1,
        max_substages_per_stage=3,
        stage_boundary_replications=0,
        max_nodes=3,
        max_tool_calls=3,
        max_llm_calls=24,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(FakeRouter(), "incomplete-substage-task")

    progress = agent.last_state.stage_progress(ResearchStage.PRELIMINARY)
    assert [substage.id for substage in progress.substages] == ["preliminary.1"]
    assert progress.current_substage.completion_criteria_met is False
    assert progress.current_substage.last_completion_check_node_count == 2
    assert model.purposes.count("evaluate_preliminary_substage") == 2


def test_plot_artifacts_are_sent_to_multimodal_critic_and_journal(tmp_path):
    model = VisualScriptedModel()
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=1,
        verification_node_budget=1,
        verification_min_nodes=1,
        stage_boundary_replications=0,
        max_nodes=1,
        max_tool_calls=1,
        max_llm_calls=6,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=model)

    agent.run_agent(VisualArtifactRouter(tmp_path), "visual-feedback-task")

    node = agent.last_state.tree.nodes[0]
    assert len(node.visual_artifacts) == 1
    assert node.visual_artifacts[0].endswith("results/curve.png")
    assert model.multimodal_paths == [node.visual_artifacts]
    assert agent.last_state.journal.visual_feedback == [
        {
            "node_id": node.id,
            "artifacts": node.visual_artifacts,
            "feedback": ["The plotted measurement converged to 42."],
        }
    ]


def test_default_expansion_uses_three_clean_parallel_siblings():
    config = AIScientistConfig(
        initial_drafts=1,
        preliminary_node_budget=1,
        tuning_node_budget=0,
        research_node_budget=3,
        verification_node_budget=1,
        verification_min_nodes=1,
        adaptive_substages=False,
        stage_boundary_replications=0,
        max_nodes=4,
        max_tool_calls=4,
        max_llm_calls=24,
        max_actions_per_node=1,
    )
    agent = AIScientistAgent(config=config, model_gateway=ScriptedModel())

    agent.run_agent(FakeRouter(), "parallel-clean-expansion-task")

    research_nodes = agent.last_state.tree.by_stage(
        ResearchStage.RESEARCH, include_boundary=False
    )
    seed_id = agent.last_state.stage_progress(ResearchStage.RESEARCH).seed_node_id
    assert len(research_nodes) == 3
    assert {node.parent_id for node in research_nodes} == {seed_id}
    assert all(not node.physical_state_inherited for node in research_nodes)
    assert len({node.branch_id for node in research_nodes}) == 3
    assert agent.last_state.replay_tool_calls == 0
