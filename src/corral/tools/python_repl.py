"""Agent-facing definition for Corral's checkpointed Python REPL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from corral.core.tool import Tool, ToolConcurrency, WorkspaceAccess
from corral.runtime.python_repl import (
    DEFAULT_MAX_CODE_CHARS,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
    CodeExecutor,
    NamespaceFactory,
    PythonREPLResult,
    PythonREPLSession,
    execute_python_repl,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Mapping, Sequence

DEFAULT_DESCRIPTION = (
    "A persistent Python REPL with no per-call execution timeout. "
    "Use print(...) to see results. Normal Python builtins and installed packages "
    "are available; file, process, and network access follow the runtime's permissions."
)


class PythonREPLTool(Tool):
    """Controller-managed schema plus reusable local/restricted executors.

    An environment owns the checkpoint in its projected state and calls
    `execute_repl`.  This object never evaluates model code in the
    controller. `controller_dispatch` routes it through the owning
    Environment without granting direct controller execution.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        argument_name: str,
        argument_description: str,
        namespace_factory: NamespaceFactory | None,
        code_executor: CodeExecutor | None,
        synchronized_names: Sequence[str],
        export_names: Sequence[str],
        export_result_names: Mapping[str, str] | None,
        max_code_chars: int,
        max_output_chars: int,
        address_space_bytes: int,
        workspace_access: WorkspaceAccess | str,
    ):
        super().__init__(
            name=name,
            description=description,
            params_json_schema={
                "type": "object",
                "properties": {
                    argument_name: {
                        "type": "string",
                        "description": argument_description,
                    }
                },
                "required": [argument_name],
            },
            # Model Python always runs in a worker. The controller only manages
            # the durable checkpoint and public-data dispatch.
            controller_dispatch=True,
            concurrency=ToolConcurrency.SERIAL,
            workspace_access=workspace_access,
        )
        self.argument_name = argument_name
        self.namespace_factory = namespace_factory
        self.code_executor = code_executor
        self.synchronized_names = tuple(synchronized_names)
        self.export_names = tuple(export_names)
        self.export_result_names = dict(export_result_names or {})
        self.max_code_chars = max_code_chars
        self.max_output_chars = max_output_chars
        self.address_space_bytes = address_space_bytes

    def execute(self, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "PythonREPLTool must be dispatched by an Environment that owns its checkpoint"
        )

    def create_session(
        self, initial_data: Mapping[str, Any] | None = None
    ) -> PythonREPLSession:
        """Create the isolated local-development implementation."""
        return PythonREPLSession(
            initial_data,
            namespace_factory=self.namespace_factory,
            code_executor=self.code_executor,
            export_names=self.export_names,
            max_code_chars=self.max_code_chars,
            max_output_chars=self.max_output_chars,
            address_space_bytes=self.address_space_bytes,
        )

    def execute_repl(
        self,
        *,
        code: str,
        workspace: str,
        initial_data: Mapping[str, Any] | None = None,
        checkpoint: str | None = None,
        namespace_updates: Mapping[str, Any] | None = None,
        cancel: threading.Event | None = None,
    ) -> PythonREPLResult:
        """Run one transactional call in a restricted Docker worker."""
        return execute_python_repl(
            code=code,
            initial_data=initial_data,
            checkpoint=checkpoint,
            workspace=workspace,
            namespace_factory=self.namespace_factory,
            code_executor=self.code_executor,
            namespace_updates=namespace_updates,
            synchronized_names=self.synchronized_names,
            export_names=self.export_names,
            export_result_names=self.export_result_names,
            cancel=cancel,
            max_code_chars=self.max_code_chars,
            max_output_chars=self.max_output_chars,
            address_space_bytes=self.address_space_bytes,
            workspace_access=self.workspace_access,
        )


def create_python_repl_tool(
    *,
    name: str = "PythonREPL",
    description: str = DEFAULT_DESCRIPTION,
    argument_name: str = "input_code",
    argument_description: str = "A valid Python command.",
    namespace_factory: NamespaceFactory | None = None,
    code_executor: CodeExecutor | None = None,
    synchronized_names: Sequence[str] = (),
    export_names: Sequence[str] = (),
    export_result_names: Mapping[str, str] | None = None,
    max_code_chars: int = DEFAULT_MAX_CODE_CHARS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    address_space_bytes: int = DEFAULT_WORKER_ADDRESS_SPACE_BYTES,
    workspace_access: WorkspaceAccess | str = WorkspaceAccess.NONE,
) -> PythonREPLTool:
    """Create a serial, checkpointed REPL definition for a stateful environment."""
    return PythonREPLTool(
        name=name,
        description=description,
        argument_name=argument_name,
        argument_description=argument_description,
        namespace_factory=namespace_factory,
        code_executor=code_executor,
        synchronized_names=synchronized_names,
        export_names=export_names,
        export_result_names=export_result_names,
        max_code_chars=max_code_chars,
        max_output_chars=max_output_chars,
        address_space_bytes=address_space_bytes,
        workspace_access=workspace_access,
    )


__all__ = ["PythonREPLTool", "create_python_repl_tool"]
