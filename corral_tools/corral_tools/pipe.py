from __future__ import annotations

from typing import TYPE_CHECKING

from corral.base import Tool

if TYPE_CHECKING:
    from collections.abc import Sequence


# TODO pipe multiple tools together to create one tool, usefule for ocp workflows
class ToolPipe(Tool):
    """A pipeline of tools where output of one tool becomes input for the next."""

    def __init__(
        self,
        name: str,
        tools: Sequence[Tool],
        output_input_mappings: Sequence[dict[str, str]] | None = None,
    ):
        """
        Args:
            name: Name of the pipeline
            tools: List of tools to execute in sequence
            output_input_mappings: List of mappings defining how output of each tool
                                 maps to input of next tool. If None, attempts to map
                                 automatically based on argument names.
        """
        self.tools = tools
        self.mappings = output_input_mappings or []
        # Use first tool's arguments as pipeline arguments
        super().__init__(
            name=name,
            description=f"Pipeline of tools: {' -> '.join(t.name for t in tools)}",
            arguments=tools[0].arguments,
        )

    def execute(self, **kwargs) -> str:
        """Execute tools in sequence, passing outputs as inputs."""
        current_args = kwargs
        results = []

        for i, tool in enumerate(self.tools):
            result = tool.execute(**current_args)
            results.append(result)

            # If this isn't the last tool, prepare args for next tool
            if i < len(self.tools) - 1:
                if i < len(self.mappings):
                    # Use explicit mapping
                    current_args = {
                        target: result
                        if source == "output"
                        else current_args.get(source)
                        for target, source in self.mappings[i].items()
                    }
                else:
                    # Try automatic mapping
                    current_args = {
                        arg.name: result
                        if arg.name == "input"
                        else current_args.get(arg.name)
                        for arg in self.tools[i + 1].arguments
                    }

        return "\n".join(results)
