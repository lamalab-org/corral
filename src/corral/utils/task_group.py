from pathlib import Path

from loguru import logger

from corral.backend.env import Environment
from corral.backend.task import TaskGroup
from corral.backend.tool import Tool
from corral.utils.io_tools import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)


class TaskGroupEnvironment(Environment):
    """Environment that works with a task group - simple composition approach"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        subtask_specific_tools: dict[str, Tool],
        base_work_dir: str | Path,
        taskgroup_common_tools: dict[str, Tool] | None = None,
        hidden_args_keys: list[str] | None = None,
        fsmanager_app: str | None = None,
        extra_file_tools: dict[str, Tool] | None = None,
        extra_fsmanager_tool_classes: dict[str, type] | None = None,
    ):
        self.task_group = task_group
        self.subtask_specific_tools = subtask_specific_tools
        self.taskgroup_common_tools = taskgroup_common_tools or {}
        self._hidden_args_keys = hidden_args_keys or []
        self._fsmanager_app = fsmanager_app
        self._extra_file_tools = extra_file_tools or {}
        self._extra_fsmanager_tool_classes = extra_fsmanager_tool_classes or {}

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        # Initialize hidden_args before super().__init__ which calls reset_state
        self.hidden_args: dict[str, str] = {}

        super().__init__(f"{task_id}", base_work_dir=base_work_dir)

        self._update_hidden_args()

        # Add tools
        self._add_task_tools()
        self._setup_file_tools()

    def _update_hidden_args(self):
        """Update hidden_args with current workspace values."""
        if "work_dir" in self._hidden_args_keys:
            self.hidden_args["work_dir"] = self.get_current_work_dir()

    def _add_task_tools(self):
        """Add required tools for the task"""
        for tool_name in self.current_task.tools:
            if tool_name in self.subtask_specific_tools:
                self.add_tool(self.subtask_specific_tools[tool_name])
            else:
                logger.warning(
                    f"Tool {tool_name} required for task {self.task_id} not found"
                )

        for tool in self.taskgroup_common_tools.values():
            self.add_tool(tool)

    def _setup_file_tools(self):
        """Setup file tools scoped to the current trial workspace."""
        current_work_dir = self.get_current_work_dir()
        if current_work_dir:
            logger.info(f"Setting up FSManager with base_path: {current_work_dir}")
            # FSManager scoped to current_work_dir (per-trial) so relative paths
            # from hidden_args tools resolve correctly within the trial workspace
            fsmanager_kwargs = {
                "base_path": current_work_dir,
                "registry": self.workspace_registry,
                "workspace_id": self.current_workspace_id,
            }
            if self._fsmanager_app:
                fsmanager_kwargs["app"] = self._fsmanager_app
            fs_manager = FSManager("file", **fsmanager_kwargs)

            # Add/update file tools
            file_tools = {
                "list_files": ListFilesTool(fs_manager),
                "read_file": ReadFileTool(fs_manager),
                "write_file": WriteFileTool(fs_manager),
                "file_info": FileInfoTool(fs_manager),
                "cat_files": CatFilesTool(fs_manager),
                "copy_file": CopyFileTool(fs_manager),
            }
            # Add tools that need FSManager instance
            for name, cls in self._extra_fsmanager_tool_classes.items():
                file_tools[name] = cls(fs_manager)
            # Add static extra tools
            file_tools.update(self._extra_file_tools)
            self.tools.update(file_tools)
            logger.info(f"File tools setup complete for workspace: {current_work_dir}")
        else:
            logger.warning("No work_dir set, skipping file tools setup")

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
        self._update_hidden_args()
        # Recreate file tools for new workspace
        self._setup_file_tools()
        return trial_id

    def get_task_prompt(self) -> str:
        """Generate the task prompt for the current task"""
        _combined_input = self.task_group.get_task_input(self.task_id)

        prompt = f"""Task: {self.current_task.name}
Description: {self.current_task.description}

Required submission format:
{self.current_task.submission_format}

"""

        prompt += "\nAvailable input data:\n"

        # Display input data from dependencies
        for dep_task_id in self.current_task.input_from_tasks:
            if dep_task_id in self.task_group.results:
                dep_result = self.task_group.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    prompt += f"- Input from {dep_task_id}: {dep_result['answer']}\n"
                else:
                    prompt += f"- Input from {dep_task_id}: {dep_result}\n"

        # Display initial input data
        if self.current_task.initial_input:
            for key, value in self.current_task.initial_input.items():
                if key != "work_dir":
                    prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.current_work_dir:
            rel_workspace = Path(self.current_work_dir).name
            prompt += f"\nIMPORTANT: You have access to filesystem tools. Your workspace directory is: {rel_workspace}/\n"
            prompt += "Write your output files to this directory.\n"

        # Add note about dependencies
        if self.current_task.input_from_tasks:
            status = []
            for dep_id in self.current_task.input_from_tasks:
                status_text = (
                    "available"
                    if dep_id in self.task_group.results
                    else "not yet available"
                )
                status.append(f"{dep_id} ({status_text})")

            prompt += f"\n\nThis task uses output from tasks: {', '.join(status)}"

        return prompt

    def score(self) -> float:
        """Score the submitted answer.

        Resolves relative paths against the current workspace so agents
        don't need to submit absolute paths.  No searching or heuristics.
        """
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0

        try:
            answer = self.state.submitted_answer.strip()

            # Resolve relative file paths against the workspace.
            # Absolute paths and non-path data (JSON strings) pass through.
            if not answer.startswith(("{", "[")) and not Path(answer).is_absolute():
                work_dir = self.get_current_work_dir()
                if work_dir:
                    answer = str(Path(work_dir) / answer)

            score = self.current_task.scoring_fn(answer)
            self.task_group.store_result(self.task_id, {"answer": answer}, score)
            logger.info(f"Task {self.task_id} scored: {score}")
            return score

        except Exception as e:
            logger.error(f"Error scoring task {self.task_id}: {e!s}", exc_info=True)
            return 0.0
