"""Run one bounded, adaptive scientific experiment inside each tree node."""

import json

from corral.agents.ai_scientist.prompts import render_prompt
from corral.agents.ai_scientist.search.nodes import (
    ExperimentDecision,
    ExperimentNode,
    ExperimentStep,
    ExperimentTermination,
    NodeStatus,
    NodeType,
)
from corral.agents.ai_scientist.state import TaskFormulation
from corral.agents.ai_scientist.tools.corral_executor import CorralExecutor
from corral.agents.ai_scientist.workers.base import StructuredModel


class Experimenter:
    """Observe, choose one action, execute it, and repeat within one node."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        max_actions_per_node: int,
        max_journal_chars: int,
        max_tool_schema_chars: int,
    ) -> None:
        self.model = model
        self.max_actions_per_node = max_actions_per_node
        self.max_journal_chars = max_journal_chars
        self.max_tool_schema_chars = max_tool_schema_chars

    def execute(
        self,
        node: ExperimentNode,
        executor: CorralExecutor | None,
        *,
        task_prompt: str,
        formulation: TaskFormulation,
        tools: list[dict],
        journal_context: str,
        action_limit: int | None = None,
        previous_node: ExperimentNode | None = None,
    ) -> ExperimentNode:
        node.status = NodeStatus.RUNNING
        # Aggregation is the one node type that reconciles existing evidence
        # without interacting with the environment.
        if node.node_type == NodeType.AGGREGATION:
            node.allocated_action_budget = 0
            node.termination_reason = ExperimentTermination.AGGREGATED
            node.status = NodeStatus.SUCCESSFUL
            return node
        if executor is None:
            raise ValueError("A physical experiment requires a Corral executor")

        limit = min(
            self.max_actions_per_node,
            self.max_actions_per_node if action_limit is None else action_limit,
        )
        node.allocated_action_budget = limit
        tool_text = json.dumps(tools, indent=2, ensure_ascii=False)
        if len(tool_text) > self.max_tool_schema_chars:
            tool_text = tool_text[: self.max_tool_schema_chars] + "\n... truncated ..."

        # ``limit`` bounds physical actions, not worker decisions. Once the
        # last permitted action has been observed, give the worker one final
        # zero-action decision in which it must explicitly finish. Without
        # that acknowledgement, a budget-truncated prefix must remain PARTIAL.
        action_count = 0
        step_index = 0
        while action_count <= limit:
            remaining_actions = limit - action_count
            if remaining_actions > 0 and executor.remaining_calls <= 0:
                node.termination_reason = ExperimentTermination.TOOL_BUDGET_EXHAUSTED
                break
            decision = self._decide(
                node=node,
                step_index=step_index,
                remaining_actions=remaining_actions,
                task_prompt=task_prompt,
                formulation=formulation,
                tools=tool_text,
                journal_context=journal_context,
                previous_node=previous_node,
            )
            action = decision.as_action()
            if action is None:
                node.trajectory.append(
                    ExperimentStep(step_index=step_index, decision=decision)
                )
                node.worker_conclusion = decision.conclusion
                node.termination_reason = ExperimentTermination.WORKER_FINISHED
                break

            # The worker saw the complete trajectory and was told that no
            # physical actions remained, but still requested another one. Keep
            # that rejected decision in the trajectory so truncation is
            # inspectable; never silently append it to the realized plan.
            if remaining_actions <= 0:
                node.trajectory.append(
                    ExperimentStep(step_index=step_index, decision=decision)
                )
                node.termination_reason = ExperimentTermination.ACTION_BUDGET_EXHAUSTED
                break

            node.plan.append(action)
            observations = executor.execute_plan(
                [action], start_index=len(node.observations)
            )
            observation = observations[0]
            node.observations.append(observation)
            node.trajectory.append(
                ExperimentStep(
                    step_index=step_index,
                    decision=decision,
                    observation=observation,
                )
            )
            action_count += 1
            step_index += 1
            if not observation.success and executor.stop_on_error:
                node.termination_reason = ExperimentTermination.ACTION_FAILED
                break

        # A failure followed by a successful adaptive correction does not poison
        # the whole experiment. Successful evidence is accepted only after the
        # worker explicitly declares the experiment complete; an exhausted
        # action/tool budget is a PARTIAL node, never a successful one.
        if node.termination_reason in {
            ExperimentTermination.ACTION_BUDGET_EXHAUSTED,
            ExperimentTermination.TOOL_BUDGET_EXHAUSTED,
        }:
            node.status = NodeStatus.PARTIAL
        elif node.observations and node.observations[-1].success:
            node.status = NodeStatus.SUCCESSFUL
        elif node.observations:
            node.status = NodeStatus.FAILED
        else:
            node.status = NodeStatus.INVALID
        return node

    def _decide(
        self,
        *,
        node: ExperimentNode,
        step_index: int,
        remaining_actions: int,
        task_prompt: str,
        formulation: TaskFormulation,
        tools: str,
        journal_context: str,
        previous_node: ExperimentNode | None,
    ) -> ExperimentDecision:
        prompt = render_prompt(
            "experiment_step",
            task_prompt=task_prompt,
            formulation=formulation.model_dump_json(indent=2),
            node=json.dumps(
                node.model_dump(
                    mode="json",
                    include={
                        "id",
                        "stage",
                        "node_type",
                        "hypothesis",
                        "rationale",
                        "experiment_goal",
                        "success_criteria",
                        "branch_workspace",
                    },
                ),
                indent=2,
                ensure_ascii=False,
            ),
            prior_checkpoint=(
                json.dumps(
                    previous_node.model_dump(
                        mode="json",
                        include={
                            "id",
                            "hypothesis",
                            "experiment_goal",
                            "success_criteria",
                            "trajectory",
                            "observations",
                            "worker_conclusion",
                            "termination_reason",
                        },
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                if previous_node is not None
                else "No prior partial checkpoint; this is a new experiment."
            ),
            trajectory=json.dumps(
                [step.model_dump(mode="json") for step in node.trajectory],
                indent=2,
                ensure_ascii=False,
            ),
            journal=journal_context[: self.max_journal_chars],
            tools=tools,
            remaining_actions=remaining_actions,
        )
        return self.model.generate(
            prompt,
            ExperimentDecision,
            purpose=f"experiment_{node.id}_step_{step_index + 1}",
        )
