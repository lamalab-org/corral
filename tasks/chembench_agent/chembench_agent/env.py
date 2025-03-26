import uvicorn
from chembench.baseline import Generation, Generations
from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import Task
from dotenv import load_dotenv
from loguru import logger
from tools import brave_search, smiles_to_iupac_name, wikipedia_search, wolfram_alpha

from corral.base import Environment
from corral.server import create_benchmark_server

load_dotenv("../.env", override=True)


_CHEMBENCH_TOOLS = [wikipedia_search, brave_search, wolfram_alpha, smiles_to_iupac_name]


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
    def __init__(
        self,
        task_id: str,
        tasks: list[Task],
        benchmark: ChemBenchmark,
        prompter: PrompterBuilder,
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

    def get_task_prompt(self) -> str:
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

        if len(self.all_prompts) != 1:
            raise ValueError("Only one prompt per task is supported")

        return f"\n\nSolve this problem: {prompts[0][0]["content"]}"

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
    tasks = []
    for topic in benchmark.registry.get_all_topics():
        if topic == "chemical_preference":
            continue
        questions = benchmark.registry.get_topic(topic)
        tasks.extend(questions.tasks)
    return tasks


def main():
    benchmark = ChemBenchmark.from_huggingface(report_dir="../reports", verbose=True)
    prompter = PrompterBuilder.from_model_object(
        model=Model(),
    )
    tasks = get_all_tasks(benchmark)

    environments = {}
    for task in tasks:
        environments[task._uuid] = ChemBenchEnvironment(
            task._uuid, [task], benchmark, prompter
        )

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
