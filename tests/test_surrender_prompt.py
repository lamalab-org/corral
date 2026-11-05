"""Default session prompts expose surrender only when it is enabled."""

import pytest

from corral.agents.prompt_utils import create_prompt
from corral.agents.react import ReActAgent
from corral.agents.tool_calling import ToolCallingAgent


@pytest.mark.parametrize("agent_type", [ReActAgent, ToolCallingAgent])
def test_default_prompt_describes_enabled_surrender(agent_type):
    agent = agent_type()

    def prompt(enabled: bool) -> str:
        messages = create_prompt(
            system_prompt=agent.system_prompt,
            user_prompt=agent.user_prompt,
            task_guide="Solve the task",
            examples=[],
            surrender_prompt=agent.surrender_prompt,
            enable_surrender=enabled,
        )
        return str(messages[-1]["content"])

    assert '"SURRENDER"' in prompt(True)
    assert '"SURRENDER"' not in prompt(False)
