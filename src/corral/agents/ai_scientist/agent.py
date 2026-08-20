"""AI Scientist search harness implemented as a first-class session agent."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from threading import RLock
from typing import Any
from uuid import uuid4

import anyio
from anyio.from_thread import BlockingPortal

from corral.agents.ai_scientist.config import AIScientistConfig
from corral.agents.ai_scientist.journal import JSONLTraceWriter
from corral.agents.ai_scientist.manager import ExperimentManager
from corral.agents.ai_scientist.search.selector import TreeSelector
from corral.agents.ai_scientist.state import ScientistState
from corral.agents.ai_scientist.tools.execution_pool import (
    BranchSessionHandle,
    ExecutionPool,
    ReplayEquivalence,
)
from corral.agents.ai_scientist.workers.base import (
    LiteLLMStructuredModel,
    LLMBudgetExceeded,
    StructuredModel,
)
from corral.agents.ai_scientist.workers.critic import ScientificCritic
from corral.agents.ai_scientist.workers.experimenter import Experimenter
from corral.agents.ai_scientist.workers.planner import NodePlanner, TaskFormulator
from corral.agents.ai_scientist.workers.synthesizer import FinalSynthesizer
from corral.agents.base_agent import BaseAgent, prompt_with_state_history
from corral.agents.schema import AgentOutcome, AgentUsage, BudgetExhaustedError
from corral.agents.session import AgentSession, AgentSessionCapabilities
from corral.agents.utils import llm_call
from corral.core.action import SUBMIT_ANSWER_TOOL_NAME, Action
from corral.core.errors import concise_error_message

_SCIENTIST_SYSTEM_PROMPT = """
You are the reasoning component of a scientific experiment manager. Interact
with the task environment only through experiments explicitly represented by
the supplied Corral tool schemas. Never invent observations, never use a
benchmark score as evidence, and distinguish measured facts from hypotheses.
Return the requested structured object without Markdown or surrounding prose.
""".strip()

_TRACE_NODE_ID = re.compile(r"node_\d+")
_TRACE_LABEL_HYPOTHESIS_CHARS = 96
_SCIENTIST_STATE_NAMESPACE = "ai_scientist"


@dataclass(slots=True)
class _GatewayOwner:
    """Per-run transcript/usage sink expected by LiteLLMStructuredModel."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_usages: list[dict[str, int]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    cumulative_token_usage: dict[str, int] = field(default_factory=dict)

    def _accumulate_token_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self.cumulative_token_usage[key] = self.cumulative_token_usage.get(
                key, 0
            ) + int(value or 0)

    def _record_turn_usage(self, usage: dict[str, int]) -> None:
        self.turn_usages.append(usage)


def _call_llm_from_harness(portal: BlockingPortal, **call_kwargs: Any) -> Any:
    """Run one native-async LiteLLM call from the blocking search harness."""
    return portal.call(partial(llm_call, **call_kwargs))


class _BranchSessionRegistry:
    """Create logical/physical AI Scientist branches from a commit projection."""

    def __init__(self, parent: AgentSession, portal: BlockingPortal) -> None:
        self.parent = parent
        self.portal = portal
        self.execution_id = parent.execution_id
        self.execution_workspace = parent.workspace
        self._sessions: dict[str, AgentSession] = {}
        self._lock = RLock()

    def _create(
        self,
        source: AgentSession,
    ) -> BranchSessionHandle:
        execution_id = f"{self.parent.execution_id}-scientist-{uuid4()}"
        session = self.portal.call(partial(source.fork_branch, branch_id=execution_id))
        with self._lock:
            self._sessions[execution_id] = session
        return BranchSessionHandle(
            execution_id=execution_id,
            workspace=session.workspace,
            execute=partial(self.portal.call, session.execute),
        )

    def create_branch(self) -> BranchSessionHandle:
        return self._create(self.parent)

    def clone_branch(self, source_execution_id: str) -> BranchSessionHandle:
        with self._lock:
            source = self._sessions[source_execution_id]
        return self._create(source)

    def close_branch(self, execution_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(execution_id, None)
        if session is not None:
            session.close()
            session.environment.shutdown_jobs()

    def session(self, execution_id: str) -> AgentSession:
        with self._lock:
            return self._sessions[execution_id]


class AIScientistAgent(BaseAgent):
    """Progressive tree search over scientific actions exposed by Corral.

    ``model_gateway`` is an injection seam for offline tests or custom providers;
    normal use leaves it unset and uses Corral's thread-safe LiteLLM utility with
    structured-output fallback. Custom gateways must support concurrent calls
    when ``config.parallel_llm_workers`` is greater than one; set it to one for a
    gateway that requires serial access.
    """

    session_capabilities = AgentSessionCapabilities(inspect_subagents=True)

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
        surrender_prompt: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.config = config or AIScientistConfig()
        self.evaluator_model = evaluator_model or model
        self._injected_gateway = model_gateway
        self.replay_equivalence = replay_equivalence
        super().__init__(
            model=model,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt or "tool_calling/user_prompt",
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

    @staticmethod
    def _search_snapshot(state: ScientistState) -> dict[str, Any]:
        """Serialize search progress and provenance as agent-state events."""
        return {
            "formulation": state.formulation.model_dump(mode="json"),
            "current_stage": state.current_stage.value,
            "tree": state.tree.model_dump(),
            "journal": state.journal.as_dict(),
            "stages": {
                stage.value: progress.model_dump(mode="json")
                for stage, progress in state.stages.items()
            },
            "usage": {
                "tool_calls": state.tool_calls,
                "scientific_tool_calls": state.scientific_tool_calls,
                "replay_tool_calls": state.replay_tool_calls,
                "executions_created": state.executions_created,
                "executions_cloned": state.executions_cloned,
                "peak_simultaneous_executions": state.peak_simultaneous_executions,
                "llm_calls": state.llm_calls,
                "llm_tokens": state.llm_tokens,
            },
            "replay_results": [
                {
                    "branch_id": replay.branch_id,
                    "exact": replay.exact,
                    "equivalent": replay.equivalent,
                    "original": replay.original.model_dump(mode="json"),
                    "replayed": replay.replayed.model_dump(mode="json"),
                }
                for replay in state.replay_results
            ],
            "artifacts": {
                "source_workspace": state.artifact_source_workspace,
                "destination_workspace": state.artifact_destination_workspace,
                "promoted": list(state.promoted_artifacts),
            },
            "experimental_search_terminated_reason": (
                state.experimental_search_terminated_reason
            ),
        }

    @classmethod
    def _state_payload(
        cls,
        state: ScientistState | None,
        messages: Sequence[Mapping[str, Any]],
        *,
        status: str,
        tool_statistics: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "search": cls._search_snapshot(state) if state is not None else None,
            "graph": cls._trace_metadata(state, messages),
            "tool_statistics": dict(tool_statistics or {}),
            "error": error,
        }

    def _execute_session(
        self,
        session: AgentSession,
        owner: _GatewayOwner,
        portal: BlockingPortal,
    ) -> tuple[str, AgentUsage, dict[str, Any]]:
        # Search shape remains agent configuration, but the task-bound session
        # is the sole authority for how many model/SDK turns this run may use.
        config = self.config
        max_llm_calls = session.iteration_limit
        prompt_value = session.prompt
        prompt = (
            prompt_value
            if isinstance(prompt_value, str)
            else json.dumps(prompt_value, ensure_ascii=False)
        )
        prompt = prompt_with_state_history(prompt, session.messages)
        tools = [dict(tool) for tool in session.tools]
        tool_payload = {"tools": tools}
        tools = tool_payload.get("tools", [])

        gateway = self._injected_gateway or LiteLLMStructuredModel(
            owner=owner,
            default_model=self.model,
            evaluator_model=self.evaluator_model,
            system_prompt=f"{self.system_prompt}\n\n{_SCIENTIST_SYSTEM_PROMPT}",
            temperature=self.temperature,
            api_endpoint=self.api_endpoint,
            max_calls=max_llm_calls,
            use_structured_output=config.use_structured_output,
            completion_runner=partial(_call_llm_from_harness, portal),
            llm_kwargs=self.kwargs,
        )
        task_id = str(getattr(session, "task_id", session.execution_id))
        examples = session.examples
        trace = JSONLTraceWriter(config.trace_path, task_id)
        formulator = TaskFormulator(
            gateway, max_tool_schema_chars=config.max_tool_schema_chars
        )
        initial_llm_calls = gateway.call_count
        initial_llm_tokens = getattr(gateway, "token_count", 0)
        formulation = formulator.formulate(
            task_prompt=prompt, tools=tools, examples=examples
        )
        state = ScientistState(task_prompt=prompt, tools=tools, formulation=formulation)
        trace.write("formulation", formulation.model_dump(mode="json"))

        branch_sessions = _BranchSessionRegistry(session, portal)
        pool = ExecutionPool(
            sessions=branch_sessions,
            tools=tool_payload,
            max_tool_calls=config.max_tool_calls,
            max_observation_chars=config.max_observation_chars,
            stop_on_error=config.stop_plan_on_tool_error,
            replay_equivalence=self.replay_equivalence,
        )
        planner = NodePlanner(
            gateway,
            max_actions_per_node=config.max_actions_per_node,
            max_journal_chars=config.max_journal_chars,
            max_tool_schema_chars=config.max_tool_schema_chars,
        )
        critic = ScientificCritic(
            gateway,
            evaluator_model=self.evaluator_model,
            max_journal_chars=config.max_journal_chars,
        )
        manager = ExperimentManager(
            config=config,
            max_llm_calls=max_llm_calls,
            model=gateway,
            planner=planner,
            experimenter=Experimenter(
                gateway,
                max_actions_per_node=config.max_actions_per_node,
                max_journal_chars=config.max_journal_chars,
                max_tool_schema_chars=config.max_tool_schema_chars,
            ),
            critic=critic,
            selector=TreeSelector(
                debug_probability=config.debug_probability,
                max_debug_depth=config.max_debug_depth,
                debug_leaf_only=config.debug_leaf_only,
                max_children_per_node=config.max_children_per_node,
                exploration_weight=config.tree_exploration_weight,
                random_seed=config.random_seed,
            ),
            trace=trace,
            initial_llm_calls=initial_llm_calls,
            initial_llm_tokens=initial_llm_tokens,
        )
        try:
            state = manager.execute_search(state, pool)

            selected_branch = None
            for candidate in state.best_nodes:
                node = candidate
                while True:
                    if node.branch_id in pool.active:
                        selected_branch = pool.active[node.branch_id]
                        break
                    if node.parent_id is None:
                        break
                    node = state.tree.get(node.parent_id)
                if selected_branch is not None:
                    break
            promotion = pool.promote_artifacts(
                state.best_nodes,
                state.tree,
                destination_workspace=session.workspace,
                destination_execution_id=session.execution_id,
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
                gateway, max_journal_chars=config.max_journal_chars
            ).generate(
                task_prompt=prompt,
                journal_context=state.journal.context(config.max_journal_chars),
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
            # run; the local search state contains every node recorded so far.
            graph = self._trace_metadata(state, owner.messages)
            trace.write("graph_snapshot", graph)
            pool.close_all()
            branch_tool_statistics = pool.tool_statistics()
            portal.call(
                session.set_agent_state,
                _SCIENTIST_STATE_NAMESPACE,
                self._state_payload(
                    state,
                    owner.messages,
                    status="partial",
                    tool_statistics=branch_tool_statistics,
                ),
            )

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
        usage = self._usage(
            owner.cumulative_token_usage,
            llm_calls=state.llm_calls,
        )
        metadata = {
            "tool_statistics": branch_tool_statistics,
            "graph": self._trace_metadata(state, owner.messages),
        }
        portal.call(
            session.set_agent_state,
            _SCIENTIST_STATE_NAMESPACE,
            self._state_payload(
                state,
                owner.messages,
                status="completed",
                tool_statistics=branch_tool_statistics,
            ),
        )
        return answer, usage, metadata

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run search, experimentation, and synthesis through session branches."""
        owner = _GatewayOwner()
        await session.set_agent_state(
            _SCIENTIST_STATE_NAMESPACE,
            self._state_payload(None, (), status="running"),
        )
        try:
            async with BlockingPortal() as portal:
                answer, usage, metadata = await anyio.to_thread.run_sync(
                    lambda: self._execute_session(session, owner, portal)
                )
        except LLMBudgetExceeded as exc:
            error = concise_error_message(exc)
            await session.set_agent_state(
                _SCIENTIST_STATE_NAMESPACE,
                {
                    **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                    "status": "iteration_limit",
                    "error": error,
                },
            )
            return AgentOutcome(status="iteration_limit", error=error)
        except BudgetExhaustedError as exc:
            error = concise_error_message(exc)
            await session.set_agent_state(
                _SCIENTIST_STATE_NAMESPACE,
                {
                    **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                    "status": "budget_exhausted",
                    "error": error,
                },
            )
            return AgentOutcome(status="budget_exhausted", error=error)
        except Exception as exc:
            error = concise_error_message(exc)
            await session.set_agent_state(
                _SCIENTIST_STATE_NAMESPACE,
                {
                    **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                    "status": "failed",
                    "error": error,
                },
            )
            return AgentOutcome(status="agent_failure", error=error)

        if owner.turn_usages:
            for index, turn_usage in enumerate(owner.turn_usages):
                start = index * 3
                await session.record_messages(
                    owner.messages[start : start + 3],
                    usage=turn_usage,
                )
        else:
            for message in owner.messages:
                await session.record_message(message)
        answer = answer.strip()
        if not answer:
            await session.set_agent_state(
                _SCIENTIST_STATE_NAMESPACE,
                {
                    **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                    "status": "protocol_failure",
                    "error": "AI Scientist produced an empty final answer",
                },
            )
            return AgentOutcome(
                status="protocol_failure",
                error="AI Scientist produced an empty final answer",
                usage=usage,
                metadata=metadata,
            )
        await session.set_agent_state(
            _SCIENTIST_STATE_NAMESPACE,
            {
                **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                "status": "completed",
                "error": None,
            },
        )
        submission = await session.execute(
            Action(
                name=SUBMIT_ANSWER_TOOL_NAME,
                arguments={"answer": answer},
                content=answer,
                metadata={"agent": type(self).__name__},
            )
        )
        if not submission.success:
            await session.set_agent_state(
                _SCIENTIST_STATE_NAMESPACE,
                {
                    **dict(session.get_agent_state(_SCIENTIST_STATE_NAMESPACE) or {}),
                    "status": "protocol_failure",
                    "error": f"submit_answer failed: {submission.error}",
                },
            )
            return AgentOutcome(
                status="protocol_failure",
                error=f"submit_answer failed: {submission.error}",
                usage=usage,
                metadata=metadata,
            )
        return AgentOutcome(
            status="completed",
            answer=answer,
            usage=usage,
            metadata=metadata,
        )

    @staticmethod
    def _trace_metadata(
        state: ScientistState | None,
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Describe an experiment tree without mutating canonical messages.

        The returned object is stored as a sibling of ``messages`` in verbose
        agent logs. Consumers can render the tree directly from ``nodes`` and
        ``edges``; readable labels avoid having to inspect a full node payload.
        ``message_links`` correlates worker responses whose existing, legal
        ``name`` field contains a node id. No custom key is ever placed on an
        API-bound message.
        """
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
        for index, message in enumerate(messages):
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
