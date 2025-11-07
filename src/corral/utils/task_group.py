import inspect

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
from corral.utils.tool_helpers import smart_resolve_path


class TaskGroupEnvironment(Environment):
    """Environment that works with a task group - simple composition approach"""

    def __init__(
        self,
        task_id: str,
        task_group: TaskGroup,
        subtask_specific_tools: dict[str, Tool],
        base_work_dir: str,
        taskgroup_common_tools: dict[str, Tool] | None = None,
    ):
        self.task_group = task_group
        self.subtask_specific_tools = subtask_specific_tools
        self.taskgroup_common_tools = taskgroup_common_tools or {}

        if task_id not in task_group.tasks:
            raise ValueError(f"Task {task_id} not found in task group")

        self.current_task = task_group.tasks[task_id]

        super().__init__(f"{task_id}", base_work_dir=base_work_dir)

        # Add tools
        self._add_task_tools()
        self._setup_file_tools()

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
        """Setup file tools for current workspace"""
        if self.current_work_dir:
            logger.info(
                f"DEBUG: Setting up FSManager with base_path: {self.current_work_dir}"
            )
            # Create new FSManager for current workspace
            fs_manager = FSManager("file", base_path=self.current_work_dir)

            # Add/update file tools
            self.tools.update(
                {
                    "list_files": ListFilesTool(fs_manager),
                    "read_file": ReadFileTool(fs_manager),
                    "write_file": WriteFileTool(fs_manager),
                    "file_info": FileInfoTool(fs_manager),
                    "cat_files": CatFilesTool(fs_manager),
                    "copy_file": CopyFileTool(fs_manager),
                }
            )
            logger.info(
                f"DEBUG: File tools setup complete for workspace: {self.current_work_dir}"
            )
        else:
            logger.warning("DEBUG: No current_work_dir set, skipping file tools setup")

    def reset_state(self) -> str:
        """Reset state and update file tools for new workspace"""
        trial_id = super().reset_state()
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
            prompt += "\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n"

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
        """Score the submitted answer
        
        This method supports scoring functions with two different signatures:
        1. Single parameter: scoring_fn(answer) - for scoring functions that don't need ground truth
        2. Dual parameters: scoring_fn(prediction, ground_truth) - for scoring functions that compare against ground truth
        
        The detection tries multiple strategies:
        - First checks for common dual-parameter names: 'ground_truth', 'target', 'expected', 'reference'
        - Falls back to parameter count: 2 or more parameters means dual-parameter mode
        """
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0

        try:
            # Get and log the raw submission
            answer_value = self.state.submitted_answer.strip()
            logger.info(f"Raw submission for {self.task_id}: {answer_value!r}")
            resolved_answer = smart_resolve_path(answer_value)
            logger.info(f"Resolved answer for {self.task_id}: {resolved_answer!r}")
            
            # Call the scoring function - support both single and dual parameter signatures
            # Check the scoring function signature to determine how to call it
            sig = inspect.signature(self.current_task.scoring_fn)
            params = sig.parameters
            param_names = list(params.keys())
            
            # Detect dual-parameter scoring functions by checking:
            # 1. Common ground truth parameter names
            # 2. Parameter count (2 or more required parameters indicates dual-parameter)
            common_ground_truth_params = {'ground_truth', 'target', 'expected', 'reference'}
            has_ground_truth_param = bool(common_ground_truth_params & set(param_names))
            
            # Only count required parameters (exclude those with defaults)
            required_params = [
                name for name, param in params.items()
                if param.default == inspect.Parameter.empty
            ]
            has_multiple_required_params = len(required_params) >= 2
            
            if has_ground_truth_param or has_multiple_required_params:
                # Scoring function expects both prediction and ground_truth
                # Use keyword arguments if the function has standard parameter names,
                # otherwise use positional arguments for flexibility
                logger.debug(
                    f"Using dual-parameter scoring (params: {param_names})"
                )
                if has_ground_truth_param:
                    # Use keyword arguments for clarity when standard names are present
                    score = self.current_task.scoring_fn(
                        prediction=resolved_answer, 
                        ground_truth=self.current_task.scoring_inputs
                    )
                else:
                    # Use positional arguments for non-standard parameter names
                    score = self.current_task.scoring_fn(
                        resolved_answer,
                        self.current_task.scoring_inputs
                    )
            else:
                # Scoring function expects only the answer (single parameter)
                logger.debug(
                    f"Using single-parameter scoring (params: {param_names})"
                )
                score = self.current_task.scoring_fn(resolved_answer)

            # Store result in task group
            self.task_group.store_result(
                self.task_id, {"answer": resolved_answer}, score
            )
            logger.info(f"Task {self.task_id} scored: {score}")

            return score

        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0
