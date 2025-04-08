import os
from typing import Any

import uvicorn
from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import Task
from loguru import logger
from promptstore import PromptStore
from tools import (
    app,
    crop_plot_with_labels,
    decimer_molecule_extraction,
    deplot_image_extractor,
    enhanced_brave_search,
    extract_table_text,
    llm_vision_expert,
    molscribe_molecule_extraction,
    rxnscribe_reaction_extraction,
    search_lab_safety,
    search_ms_guide,
    search_nmr_guide,
)

from corral.base import Environment
from corral.io import (
    CatFilesTool,
    CopyFileTool,
    FileInfoTool,
    FSManager,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from corral.server import create_benchmark_server
from corral.utils import (
    Model,
    chunk_text,
    create_vector_database,
)

store = PromptStore("./prompts")
BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")


def create_embedding_datasets():
    """Create embedding datasets for the tools"""
    lab_safety_guidelines = store.get("314a75ca-c96c-48d9-92e9-0ade5e1ce373")
    create_vector_database(
        chunk_text(lab_safety_guidelines.fill({})), "lab_safety_collection"
    )
    ms_guidelines = store.get("d8f0ce84-f4aa-4c77-a0da-dba2e054bfff").fill({})
    create_vector_database([ms_guidelines], "ms_guide_collection")
    nmr_guidelines = store.get("ee669c37-8a6a-400d-82de-f73cc3a7a175").fill({})
    create_vector_database([nmr_guidelines], "nmr_guide_collection")


_MACBENCH_TOOLS = [
    search_lab_safety,
    enhanced_brave_search,
    search_ms_guide,
    search_nmr_guide,
    llm_vision_expert,
    deplot_image_extractor,
    extract_table_text,
    decimer_molecule_extraction,
    molscribe_molecule_extraction,
    rxnscribe_reaction_extraction,
    crop_plot_with_labels,
]


class MaCBenchEnvironment(Environment):
    """MaCBench Environment based on the multimodal Q&A MaCBench dataset (arXiv:2411.16955).

    Args:
        task_id (str): The task ID.
        tasks (list[Task]): List of tasks.
        benchmark (ChemBenchmark): The benchmark object.
        prompter (PrompterBuilder): The prompter object.
        tools (dict[str, Any] | None): Dictionary of tools to be used apart from the default MaCBench tools, e.g., file system tools.
    """

    def __init__(
        self,
        task_id: str,
        tasks: list[Task],
        benchmark: ChemBenchmark,
        prompter: PrompterBuilder,
        tools: dict[str, Any] | None,
    ):
        """Initialize the MaCBench environment."""
        self.task_id = task_id
        self.tasks = tasks
        self.benchmark = benchmark
        self.prompter = prompter

        self.task_map = {}
        self.all_prompts = []
        self.all_score_maps = []

        super().__init__(task_id)
        # Add multiple tools
        for tool in _MACBENCH_TOOLS:
            self.add_tool(tool)
        for tool in tools.values():
            self.add_tool(tool)

    def get_task_prompt(self) -> list[dict]:
        """Get the task prompt for the environment.
        Each task will correspond to each question in MaCBench.
        """
        current_idx = 0
        for task_idx, task in enumerate(self.tasks):
            if self.prompter.is_mcq(task):
                prompts, score_maps = self.prompter._prompts_with_choices(
                    task._examples
                )
            else:
                prompts = self.prompter._prompts_general(task._examples)
                score_maps = [{"correct": 1, "incorrect": 0}] * len(task._examples)

            for i in range(len(prompts)):
                self.task_map[current_idx + i] = {
                    "task_idx": task_idx,
                    "example_idx": i,
                }
            self.all_prompts.extend(prompts)
            self.all_score_maps.extend(score_maps)
            current_idx += len(prompts)

        return self.all_prompts[0][0]["content"]

    def score(self) -> float:
        """Score based on submitted answer"""
        logger.info(self.state.submitted_answer)
        model_kwargs = {}
        if self.state.submitted_answer is None:
            return 0.0

        try:
            submitted_result = str(self.state.submitted_answer)
            completions = [submitted_result]

            all_results = []
            for i, completion_list in enumerate(completions):
                task_info = self.task_map[i]
                task = self.tasks[task_info["task_idx"]]

                result = self.prompter._process_single_result(
                    completion_list,
                    task,
                    task_info["example_idx"],
                    self.all_score_maps[i],
                    self.all_prompts[i],
                    **model_kwargs,
                )
                all_results.append(result)

            if len(all_results) != 1:
                raise ValueError("Only one result per task is supported")

            logger.info(all_results[0]["metrics"])
            return float(all_results[0]["metrics"]["all_correct"])

        except (ValueError, TypeError, AttributeError) as e:
            logger.info(f"Error in scoring: {e!s}")
            return 0.0


def get_all_tasks(benchmark: ChemBenchmark) -> list[Task]:
    """Get all tasks from the benchmark registry."""
    tasks = []
    topics = benchmark.registry.get_all_topics()
    for _i, topic in enumerate(topics, 1):
        questions = benchmark.registry.get_topic(topic)
        tasks.extend(questions.tasks)
    return tasks


def main():
    benchmark = ChemBenchmark.from_huggingface(
        "jablonkagroup/MaCBench", report_dir="../reports", verbose=True
    )
    prompter = PrompterBuilder.from_model_object(
        model=Model(),
        prompt_type="multimodal_instruction",
    )
    fs_manager = FSManager("file", base_path=BASE_WORK_DIR)

    fs_tools = {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
    }

    tasks = get_all_tasks(benchmark)

    create_embedding_datasets()
    environments = {}
    for task in tasks:
        environments[task._uuid] = MaCBenchEnvironment(
            task._uuid, [task], benchmark, prompter, tools=fs_tools
        )

    # Create and run server
    with app.run():
        server_app = create_benchmark_server(environments)
        uvicorn.run(server_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
