import uvicorn
from loguru import logger
from tools import calculator

from corral.base import Environment, Tool
from corral.io import (
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
    MkdirTool,
    CatFilesTool
)

from corral.server import run_server


class MathEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        answer: float,
    ):
        self.question = question
        self.correct_answer = answer

        # The answer file path can be made configurable.
        # Here we use a virtual file path if using the local ("file") protocol,
        # self.answer_file = os.environ.get(
        #     "BASE_IO_PATH", f"./corral_tmp/{task_id}_answer.txt"
        # )
        # self.answer_file = "answer.txt"

        # Initialize environment with the given task id.
        super().__init__(task_id, base_work_dir="corral_tmp")

        # Add math-related tools
        self.add_tool(calculator)   
        self._setup_file_tools()
        # self.add_tool(number_converter)
        # self.add_tool(UnitConverterTool())

        # # Add I/O tools so that the agent can read/write answer files.
        # # (Agents will call these using submissions like "write_file" with appropriate arguments.)
        # self.add_tool(io_tools["read_file"])
        # self.add_tool(io_tools["write_file"])
        # self.add_tool(io_tools["list_files"])
        # self.add_tool(io_tools["file_info"])
        # self.add_tool(io_tools["copy_file"])
        # self.add_tool(io_tools["cat_files"])
        # self.add_tool(io_tools["mkdir"])

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

        prompt = f"Solve this math problem:\n{self.question}\n\n"
        # Add workspace info
        if self.current_work_dir:
            prompt += "\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n"
        
        logger.info(f"task prompt is : {prompt}")
        
        return prompt
        # return (
        #     f"Solve this math problem:\n{self.question}\n\n"
        #     f"Once you have solved the problem, write your numerical answer into the file:\n"
        #     f"  {self.answer_file}\n\n"
        #     "You can use the available I/O tools (e.g., write_file) to store your answer. "
        #     "For example, you might call write_file with parameters: path, and content (your answer)."
        # )

    def score(self) -> float:
        """Score the submitted answer"""
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            return 0.0

        try:
            # Get and log the raw submission
            answer_value = self.state.submitted_answer.strip()
            logger.info(f"Raw submission for {self.task_id}: {answer_value!r}")

            # Call the scoring function with the raw answer
            # score = self.current_task.scoring_fn(answer_value, **self.current_task.scoring_inputs)
            if float(answer_value) == float(self.correct_answer):
                score = 1.0
            else:
                score = 0.0
            # Store result in task group
            logger.info(f"Task {self.task_id} scored: {score}")

            return score

        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            return 0.0

if __name__ == "__main__":
    import os

    # fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
    # fs_kwargs = {}  # add more keyword options from config if needed
    # fs_manager = FSManager(protocol=fs_protocol, **fs_kwargs)
    # Create environments for different tasks
    environments = {
        "math_1": MathEnvironment(
            "math_1",
            "What is 23 + 45? Solve this and write the answer to the answer.txt file. Give the final numerical answer only.",
            68,
        ),
        # "math_2": MathEnvironment(
        #     "math_2",
        #     "What is 12 * 8? Solve this and write the answer to the designated file.",
        #     96,
        #     io_tools=io_tools,
        # ),
        # "math_3": MathEnvironment(
        #     "math_3",
        #     "What is 99 * 63 * 999 * 111?",
        #     691614693,
        #     io_tools=io_tools,
        # ),
        # "math_4": MathEnvironment(
        #     "math_4",
        #     (
        #         "What is twenty one thousand four hundred and seventy three * "
        #         "twenty one thousand four hundred and seventy three? "
        #         "Solve this and write the answer to the designated file."
        #     ),
        #     4666829,
        #     io_tools=io_tools,
        # ),
    }

    # with app.run():
    # app = create_benchmark_server(environments)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    host = os.environ.get("CORRAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CORRAL_PORT", "8000"))
    run_server(environments, host, port)