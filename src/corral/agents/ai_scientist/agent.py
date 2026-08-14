"""Public Corral BaseAgent implementation."""

import json
import re
from typing import Any

from corral.agents.ai_scientist.config import AIScientistConfig
from corral.agents.ai_scientist.journal import JSONLTraceWriter
from corral.agents.ai_scientist.manager import ExperimentManager
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.state import ScientistState
from corral.agents.ai_scientist.tools.trial_pool import ReplayEquivalence, TrialPool
from corral.agents.ai_scientist.workers.base import (
    LiteLLMStructuredModel,
    StructuredModel,
)
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner, TaskFormulator
from corral.agents.ai_scientist.workers.synthesizer import FinalSynthesizer
from corral.agents.base_agent import BaseAgent
from corral.agents.schema import AgentRunResult
from corral.router.routes import CorralRouter

_SCIENTIST_SYSTEM_PROMPT = """
You are the reasoning component of a scientific experiment manager. Interact
with the task environment only through experiments explicitly represented by
the supplied Corral tool schemas. Never invent observations, never use a
benchmark score as evidence, and distinguish measured facts from hypotheses.
Return the requested structured object without Markdown or surrounding prose.
""".strip()

_TRACE_NODE_ID = re.compile(r"node_\d+")
_TRACE_LABEL_HYPOTHESIS_CHARS = 96


class AIScientistAgent(BaseAgent):
    """Progressive tree search over scientific actions exposed by Corral.

    ``model_gateway`` is an injection seam for offline tests or custom providers;
    normal use leaves it unset and uses Corral's thread-safe LiteLLM utility with
    structured-output fallback. Custom gateways must support concurrent calls
    when ``config.parallel_llm_workers`` is greater than one; set it to one for a
    gateway that requires serial access.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        evaluator_model: str | None = None,
        config: AIScientistConfig | None = None,
        model_gateway: StructuredModel | None = None,
        replay_equivalence: ReplayEquivalence | None = None,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.config = config or AIScientistConfig()
        self.evaluator_model = evaluator_model or model
        self._injected_gateway = model_gateway
        self.replay_equivalence = replay_equivalence
        if user_prompt is None:
            # BaseAgent resolves this prompt even though this scaffold constructs
            # its worker prompts directly.
            user_prompt = "tool_calling/user_prompt"
        super().__init__(
            model=model,
            max_iterations=self.config.max_llm_calls,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )
        self._available_tools: list[dict[str, Any]] = []
        self.last_state: ScientistState | None = None
        self._branch_tool_statistics: dict[str, Any] | None = None

    @property
    def requires_answer_extraction(self) -> bool:
        """Return the synthesizer's submission-ready answer verbatim."""
        return False

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        **_kwargs: Any,
    ) -> str:
        # Clear prior-run state before any router/model operation can fail. A
        # serial caller may reuse an agent, and a partial second run must never
        # save the first run's experiment graph beside its messages.
        self.last_state = None
        self._available_tools = []
        self._branch_tool_statistics = None
        self.messages = list(self._initial_messages or [])
        prompt_value = task_prompt or interface.get_task_prompt(task_id)
        prompt = (
            prompt_value
            if isinstance(prompt_value, str)
            else json.dumps(prompt_value, ensure_ascii=False)
        )
        tool_payload = self._get_tool_payload(interface, task_id)
        tools = tool_payload.get("tools", [])
        self._available_tools = tools

        gateway = self._injected_gateway or LiteLLMStructuredModel(
            owner=self,
            default_model=self.model,
            evaluator_model=self.evaluator_model,
            system_prompt=f"{self.system_prompt}\n\n{_SCIENTIST_SYSTEM_PROMPT}",
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            max_calls=self.config.max_llm_calls,
            use_structured_output=self.config.use_structured_output,
            llm_kwargs=self.kwargs,
        )
        trace = JSONLTraceWriter(self.config.trace_path, task_id)
        formulator = TaskFormulator(
            gateway, max_tool_schema_chars=self.config.max_tool_schema_chars
        )
        initial_llm_calls = gateway.call_count
        initial_llm_tokens = getattr(gateway, "token_count", 0)
        formulation = formulator.formulate(
            task_prompt=prompt, tools=tools, examples=examples
        )
        state = ScientistState(task_prompt=prompt, tools=tools, formulation=formulation)
        self.last_state = state
        trace.write("formulation", formulation.model_dump(mode="json"))

        pool = TrialPool(
            interface=interface,
            task_id=task_id,
            tools=tool_payload,
            max_tool_calls=self.config.max_tool_calls,
            max_observation_chars=self.config.max_observation_chars,
            stop_on_error=self.config.stop_plan_on_tool_error,
            replay_equivalence=self.replay_equivalence,
        )
        planner = NodePlanner(
            gateway,
            max_actions_per_node=self.config.max_actions_per_node,
            max_journal_chars=self.config.max_journal_chars,
            max_tool_schema_chars=self.config.max_tool_schema_chars,
        )
        critic = ScientificCritic(
            gateway,
            evaluator_model=self.evaluator_model,
            max_journal_chars=self.config.max_journal_chars,
        )
        manager = ExperimentManager(
            config=self.config,
            model=gateway,
            planner=planner,
            experimenter=Experimenter(
                gateway,
                max_actions_per_node=self.config.max_actions_per_node,
                max_journal_chars=self.config.max_journal_chars,
                max_tool_schema_chars=self.config.max_tool_schema_chars,
            ),
            critic=critic,
            selector=TreeSelector(
                debug_probability=self.config.debug_probability,
                max_debug_depth=self.config.max_debug_depth,
                debug_leaf_only=self.config.debug_leaf_only,
                max_children_per_node=self.config.max_children_per_node,
                exploration_weight=self.config.tree_exploration_weight,
                random_seed=self.config.random_seed,
            ),
            trace=trace,
            initial_llm_calls=initial_llm_calls,
            initial_llm_tokens=initial_llm_tokens,
        )
        try:
            state = manager.run(state, pool)
            self.last_state = state

            promotion = pool.promote_artifacts(
                state.best_nodes,
                state.tree,
                destination_workspace=getattr(interface, "trial_workspace", None),
                destination_trial_runtime_id=getattr(
                    interface, "trial_runtime_id", None
                ),
            )
            if promotion is not None:
                state.artifact_source_workspace = promotion.source_workspace
                state.artifact_destination_workspace = promotion.destination_workspace
                state.promoted_artifacts = list(promotion.files)
                trace.write(
                    "artifacts_promoted",
                    {
                        "branch_id": promotion.branch_id,
                        "source_workspace": promotion.source_workspace,
                        "destination_workspace": promotion.destination_workspace,
                        "files": list(promotion.files),
                    },
                )

            answer = FinalSynthesizer(
                gateway, max_journal_chars=self.config.max_journal_chars
            ).generate(
                task_prompt=prompt,
                journal_context=state.journal.context(self.config.max_journal_chars),
                best_nodes=state.best_nodes,
                canonical_workspace=state.artifact_destination_workspace,
            )
            if promotion is not None:
                answer = answer.replace(
                    promotion.source_workspace,
                    promotion.destination_workspace,
                )
        finally:
            # JSONL traces get the same directly-renderable graph that verbose
            # agent logs store at top level. This is written even for a partial
            # run; the last_state tree contains every node recorded so far.
            trace.write("graph_snapshot", self._trace_metadata())
            pool.close_all()
            self._branch_tool_statistics = pool.tool_statistics()

        state.llm_calls = gateway.call_count - initial_llm_calls
        state.llm_tokens = getattr(gateway, "token_count", 0) - initial_llm_tokens
        trace.write(
            "final_answer",
            {
                "answer": answer,
                **pool.stats(),
                "llm_calls": state.llm_calls,
                "llm_tokens": state.llm_tokens,
            },
        )
        return answer

    def _agent_run_result(self, answer: str) -> AgentRunResult:
        """Attach isolated branch calls to the benchmark-facing run result."""
        result = super()._agent_run_result(answer)
        if self._branch_tool_statistics is not None:
            result.metadata["tool_statistics"] = self._branch_tool_statistics
        return result

    def _trace_metadata(self) -> dict[str, Any]:
        """Describe the experiment tree without modifying ``self.messages``.

        The returned object is stored as a sibling of ``messages`` in verbose
        agent logs. Consumers can render the tree directly from ``nodes`` and
        ``edges``; readable labels avoid having to inspect a full node payload.
        ``message_links`` correlates worker responses whose existing, legal
        ``name`` field contains a node id. No custom key is ever placed on an
        API-bound message.
        """
        state = self.last_state
        if state is None:
            return {
                "schema": "corral.ai_scientist.graph",
                "schema_version": 1,
                "current_stage": None,
                "nodes": [],
                "edges": [],
                "root_node_ids": [],
                "best_node_ids": [],
                "stage_winner_node_ids": {},
                "message_links": [],
            }

        nodes = []
        node_ids = {node.id for node in state.tree.nodes}
        for node in state.tree.nodes:
            hypothesis = " ".join(node.hypothesis.split())
            if len(hypothesis) > _TRACE_LABEL_HYPOTHESIS_CHARS:
                hypothesis = (
                    hypothesis[: _TRACE_LABEL_HYPOTHESIS_CHARS - 3].rstrip() + "..."
                )
            nodes.append(
                {
                    "id": node.id,
                    "label": (
                        f"{node.id} [{node.stage.value}/{node.node_type.value}; "
                        f"{node.status.value}] {hypothesis}"
                    ),
                    "parent_id": node.parent_id,
                    "stage": node.stage.value,
                    "node_type": node.node_type.value,
                    "status": node.status.value,
                    "depth": node.depth,
                    "boundary_validation": node.boundary_validation,
                    "substage_id": node.substage_id,
                    "stage_seed_id": node.stage_seed_id,
                    "branch_id": node.branch_id,
                    "hypothesis": node.hypothesis,
                    "experiment_goal": node.experiment_goal,
                    "related_node_ids": list(node.related_node_ids),
                }
            )

        message_links = []
        for index, message in enumerate(self.messages):
            if not isinstance(message, dict):
                continue
            name = message.get("name")
            if not isinstance(name, str):
                continue
            linked_ids = list(
                dict.fromkeys(
                    node_id
                    for node_id in _TRACE_NODE_ID.findall(name)
                    if node_id in node_ids
                )
            )
            if linked_ids:
                message_links.append(
                    {
                        "message_index": index,
                        "message_name": name,
                        "node_ids": linked_ids,
                    }
                )

        return {
            "schema": "corral.ai_scientist.graph",
            "schema_version": 1,
            "current_stage": state.current_stage.value,
            "nodes": nodes,
            "edges": [
                {
                    "source": node.parent_id,
                    "target": node.id,
                    "kind": "parent",
                }
                for node in state.tree.nodes
                if node.parent_id is not None
            ],
            "root_node_ids": [
                node.id for node in state.tree.nodes if node.parent_id is None
            ],
            "best_node_ids": [node.id for node in state.best_nodes],
            "stage_winner_node_ids": {
                stage.value: progress.best_node_id
                for stage, progress in state.stages.items()
                if progress.best_node_id is not None
            },
            "message_links": message_links,
        }

    @staticmethod
    def _get_tool_payload(interface: Any, task_id: str) -> dict[str, Any]:
        """Prefer the plan-specified MCP schema, tolerating older routers."""
        get_mcp_schema = getattr(interface, "get_mcp_tool_schema", None)
        if callable(get_mcp_schema):
            return get_mcp_schema(task_id)
        return interface.get_available_tools_for_task(task_id)
