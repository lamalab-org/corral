import math

import uvicorn
from chembench.baseline import Generation, Generations
from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import TopicQuestions, TopicRegistry
from dotenv import load_dotenv
from loguru import logger
from tools import brave_search, smiles_to_iupac_name, wikipedia_search, wolfram_alpha

from corral.base import Environment
from corral.server import create_benchmark_server

load_dotenv("../.env", override=True)
_CHEMBENCH_TOOLS = [wikipedia_search, brave_search, wolfram_alpha, smiles_to_iupac_name]
benchmark = ChemBenchmark(
    report_dir="../reports", verbose=True, state_file="../benchmark_state.pkl"
)
registry = TopicRegistry.from_huggingface("n0w0f/ChemBench-dev")

dummy_prompter = PrompterBuilder.from_model_object(
    model=None,
    prompt_type="instruction",
    post_process_ce=None,
    post_process_math=None,
    post_process_pu=None,
    post_process_smiles=None,
    post_process_rxnsmiles=None,
    other=None,
)


def generate(
    prompt,
):
    return prompt


class Model:
    def __init__(self, answer):
        self.answer = answer

    def generate(self) -> Generations:
        generations = []
        for prompt_ in [self.answer]:
            generation = generate(prompt_)
            generation = [Generation(text=generation)]
            generations.append(generation)

        return Generations(generations=generations)


class ChemBenchEnvironment(Environment):
    def __init__(
        self,
        task_id: str,
        topic: str = "test_2",
    ):
        self.topic = topic
        self.task_id = int(task_id)
        self.registry = registry
        self.benchmark = benchmark
        self.dummy_prompter = dummy_prompter
        super().__init__(task_id)
        # Add multiple tools
        for tool in _CHEMBENCH_TOOLS:
            self.add_tool(tool)

    def get_prompter(self, answer):
        model = Model(answer)
        return PrompterBuilder.from_model_object(
            model=model,
            prompt_type="instruction",
            post_process_ce=None,
            post_process_math=None,
            post_process_pu=None,
            post_process_smiles=None,
            post_process_rxnsmiles=None,
            other=None,
        )

    def get_task_prompt(self) -> str:
        example = self.registry.topics[self.topic].tasks[self.task_id]._examples
        # assuming all tasks are mcq, if not conditionally prompt
        prompts, _ = self.dummy_prompter._prompts_with_choices(example)
        return f"\n \n Solve this problem: {prompts[0]}"

    def score(self) -> float:
        """Score based on submitted answer"""
        logger.info(self.state.submitted_answer)
        if self.state.submitted_answer is None:
            return 0.0

        try:
            # Get the specific task we want to benchmark
            task = self.registry.topics[self.topic].tasks[self.task_id]

            # Create a new TopicRegistry with just this task
            single_task_registry = TopicRegistry()
            single_task_registry.topics[self.topic] = TopicQuestions(
                topic=self.topic, tasks=[task]
            )

            submitted_result = str(self.state.submitted_answer)
            prompter = self.get_prompter(submitted_result)

            # Run benchmark with the single-task registry
            results = self.benchmark.bench(
                single_task_registry, prompter, topics=[self.topic]
            )

            if not results:
                raise ValueError("No results returned from benchmark")

            logger.info("RESULTS")
            logger.info(results[0]["results"][0]["metrics"])
            score = results[0]["results"][0]["metrics"].get("score", 0.0)

            if isinstance(score, bool):
                score = 1.0 if score else 0.0

            if isinstance(score, float) and math.isnan(score):
                return 0.0

            return float(score)  # Ensure we return a valid float

        except (ValueError, TypeError, AttributeError) as e:
            logger.info(f"Error in scoring: {e!s}")
            return 0.0


if __name__ == "__main__":
    environments = {
        "chembench_1": ChemBenchEnvironment("1"),
        "chembench_2": ChemBenchEnvironment("2"),
        "chembench_3": ChemBenchEnvironment("3"),
    }

    # Create and run server
    app = create_benchmark_server(environments)
    uvicorn.run(app, host="0.0.0.0", port=8000)
