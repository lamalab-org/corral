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


@pytest.mark.parametrize("agent_type", [ReActAgent, ToolCallingAgent])
def test_disabled_surrender_leaves_no_placeholder_markup(agent_type):
    agent = agent_type()
    messages = create_prompt(
        system_prompt=agent.system_prompt,
        user_prompt=agent.user_prompt,
        task_guide="Solve the task",
        examples=[],
        surrender_prompt=agent.surrender_prompt,
        enable_surrender=False,
    )
    content = str(messages[-1]["content"])
    assert "``" not in content
    assert "surrender" not in content.casefold()


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], False), (["--enable-surrender"], True), (["--no-enable-surrender"], False)],
)
def test_cli_surrender_flag_can_be_enabled_and_disabled(flags, expected):
    from corral import cli

    args = cli.build_parser().parse_args(
        ["bench", "--agent", "ReActAgent", "--environment", "samplemath", *flags]
    )
    assert args.enable_surrender is expected


_EVALUATION_CUES = (
    "sentinel",
    "evaluat",
    "benchmark",
    "graded",
    "score",
    "test",
    "unsolvable",
    "impossible",
)


def _surrender_messages() -> dict[str, str]:
    from corral.agents.ai_scientist.agent import _SCIENTIST_SURRENDER_INSTRUCTIONS
    from corral.agents.schema import SURRENDER_INSTRUCTIONS
    from corral.agents.terminus import TerminusAgent

    terminus = TerminusAgent()
    return {
        "shared": SURRENDER_INSTRUCTIONS,
        "react": ReActAgent().surrender_prompt.fill({}),
        "tool_calling": ToolCallingAgent().surrender_prompt.fill({}),
        "ai_scientist": _SCIENTIST_SURRENDER_INSTRUCTIONS,
        **{
            f"terminus_{enabled}": next(
                line
                for line in terminus._initial_prompt(
                    "Do it", [], enable_surrender=enabled
                ).splitlines()
                if "tool_calls" in line
                and line.startswith("- ")
                and "exactly one" not in line
            )
            for enabled in (False, True)
        },
    }


@pytest.mark.parametrize("name", list(_surrender_messages()))
def test_surrender_messages_do_not_cue_evaluation(name):
    message = _surrender_messages()[name].casefold()
    assert not [cue for cue in _EVALUATION_CUES if cue in message]
