import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from loguru import logger
from tools import (
    enhanced_brave_search,
    get_c_nmr_spectra_pubchem,
    get_compound_isomers,
    get_element_info,
    get_formula_from_smiles,
    get_functional_groups,
    get_ghs_classification_pubchem,
    get_h_nmr_spectra_pubchem,
    get_ms_spectra_pubchem,
    get_number_of_isomers,
    get_pka_from_smiles,
    get_smiles_from_name,
    relevant_pubchem_sections,
    search_clinical_trials_by_drug,
    search_clinical_trials_by_query,
    search_materials_compatibility,
    simulate_spectra,
    smiles_to_name,
)

from chembench.baseline import Generation, Generations
from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import Task
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

load_dotenv("../.env", override=True)
BASE_WORK_DIR = os.environ.get("CORRAL_WORK_DIR", "../CORRAL_WORK_DIR/temp")

_CHEMBENCH_TOOLS = [
    enhanced_brave_search,
    relevant_pubchem_sections,
    smiles_to_name,
    get_smiles_from_name,
    get_pka_from_smiles,
    get_formula_from_smiles,
    get_element_info,
    get_number_of_isomers,
    get_compound_isomers,
    get_ghs_classification_pubchem,
    get_ms_spectra_pubchem,
    get_h_nmr_spectra_pubchem,
    get_c_nmr_spectra_pubchem,
    simulate_spectra,
    get_functional_groups,
    search_clinical_trials_by_query,
    search_clinical_trials_by_drug,
    search_materials_compatibility,
]


class Model:
    def __init__(self, name: str = "Dummy Model"):
        self.name = name

    def generate(self, prompts: list[str], **_kwargs):
        generations = []
        for _prompt in prompts:
            generation = None
            generations.append([Generation(text=generation)])

        return Generations(generations=generations)


class ChemBenchEnvironment(Environment):
    """
    Environment based in Q&A tasks from ChemBench (	arXiv:2404.01475)

    Args:
        task_id (str): Unique identifier for the task.
        tasks (list[Task]): List of tasks to be included in the environment.
        benchmark (ChemBenchmark): ChemBenchmark instance for evaluation.
        prompter (PrompterBuilder): PrompterBuilder instance from ChemBench for generating prompts.
        tools (dict[str, Any] | None): Dictionary of tools to be used in the environment
            in addition to the default ones from the environment (_CHEMBENCH_TOOLS).
            Normally I/O tools. Default is None.
    """

    def __init__(
        self,
        task_id: str,
        tasks: list[Task],
        benchmark: ChemBenchmark,
        prompter: PrompterBuilder,
        tools: dict[str, Any] | None = None,
    ):
        self.task_id = task_id
        self.tasks = tasks
        self.benchmark = benchmark
        self.prompter = prompter

        self.task_map = {}
        self.all_prompts = []
        self.all_score_maps = []

        super().__init__(task_id)
        # Add multiple tools
        for tool in _CHEMBENCH_TOOLS:
            self.add_tool(tool)
        for tool in tools.values():
            self.add_tool(tool)

    def get_task_prompt(self) -> str:
        """Get the task prompt for the environment.
        The task prompt is the default prompt from ChemBench containing the question and formatting instructions."""
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

        return f"\n\nSolve this problem: {prompts[0][0]['content']}"

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
    """Get all tasks from the ChemBench registry."""
    tasks = []
    topics = benchmark.registry.get_all_topics()
    for _i, topic in enumerate(topics, 1):
        questions = benchmark.registry.get_topic(topic)
        tasks.extend(questions.tasks)
    return tasks


def main():
    benchmark = ChemBenchmark.from_huggingface(report_dir="../reports", verbose=True)

    prompter = PrompterBuilder.from_model_object(
        model=Model(),
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

    environments = {}
    for task in tasks:
        environments[task._uuid] = ChemBenchEnvironment(
            task._uuid, [task], benchmark, prompter, tools=fs_tools
        )

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
