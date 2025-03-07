import uvicorn
from loguru import logger
from tools import UnitConverterTool, app, calculator, number_converter

from corral.base import Environment, Tool
from corral.io import (
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import create_benchmark_server


class MathEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        question: str,
        answer: float,
        io_tools: dict[str, Tool],
    ):
        self.question = question
        self.correct_answer = answer

        # The answer file path can be made configurable.
        # Here we use a virtual file path if using the local ("file") protocol,
        self.answer_file = os.environ.get(
            "BASE_IO_PATH", f"/corral_tmp/{task_id}_answer.txt"
        )

        # Initialize environment with the given task id.
        super().__init__(task_id)

        # Add math-related tools
        self.add_tool(calculator)
        self.add_tool(number_converter)
        self.add_tool(UnitConverterTool())

        # Add I/O tools so that the agent can read/write answer files.
        # (Agents will call these using submissions like "write_file" with appropriate arguments.)
        self.add_tool(io_tools["read_file"])
        self.add_tool(io_tools["write_file"])
        self.add_tool(io_tools["list_files"])
        self.add_tool(io_tools["file_info"])
        self.add_tool(io_tools["copy_file"])

    def get_task_prompt(self) -> str:
        return (
            f"Solve this math problem:\n{self.question}\n\n"
            f"Once you have solved the problem, write your numerical answer into the file:\n"
            f"  {self.answer_file}\n\n"
            "You can use the available I/O tools (e.g., write_file) to store your answer. "
            "For example, you might call write_file with parameters: path, and content (your answer)."
        )

    def score(self) -> float:
        """Score based on reading the answer from the designated answer file."""
        # Use the read_file tool (that was added during initialization) to load the answer.
        # The submitted answer is expected to be the file path where the answer was written.
        # (Often, the environment prompt instructs the agent to use the designated answer file.)
        read_result = self.tools.get("read_file")
        if read_result is None:
            logger.error("read_file tool not found in MathEnvironment.")
            return 0.0

        try:
            # Read the answer from the designated file.
            answer_content = read_result.execute(path=self.answer_file)

            # If no content is returned, score as zero.
            if not answer_content:
                logger.info("No content found in the answer file.")
                return 0.0

            # Clean the content and try parsing a float.
            submitted_result = float(answer_content.strip())
            logger.info(f"Submitted answer from file: {submitted_result}")
            return 1.0 if abs(submitted_result - self.correct_answer) < 0.001 else 0.0
        except Exception as e:
            logger.error(f"Error reading and scoring answer file: {e}")
            return 0.0


def create_tools(fs_manager: FSManager) -> dict[str, Tool]:
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
    }


if __name__ == "__main__":
    import os

    fs_protocol = os.environ.get("CORRAL_FS_PROTOCOL", "file")
    fs_kwargs = {}  # add more keyword options from config if needed
    fs_manager = FSManager(protocol=fs_protocol, **fs_kwargs)
    io_tools = create_tools(fs_manager)
    # Create environments for different tasks
    environments = {
        "math_1": MathEnvironment(
            "math_1",
            "What is 23 + 45? Solve this and write the answer to the designated file.",
            68,
            io_tools=io_tools,
        ),
        "math_2": MathEnvironment(
            "math_2",
            "What is 12 * 8? Solve this and write the answer to the designated file.",
            96,
            io_tools=io_tools,
        ),
        "math_3": MathEnvironment(
            "math_3",
            "What is 99 * 63 * 999 * 111?",
            691614693,
            io_tools=io_tools,
        ),
        "math_4": MathEnvironment(
            "math_4",
            (
                "What is twenty one thousand four hundred and seventy three * "
                "twenty one thousand four hundred and seventy three? "
                "Solve this and write the answer to the designated file."
            ),
            4666829,
            io_tools=io_tools,
        ),
    }

    with app.run():
        app = create_benchmark_server(environments)
        uvicorn.run(app, host="0.0.0.0", port=8000)
