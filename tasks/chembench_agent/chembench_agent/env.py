from __future__ import annotations

from typing import List

import uvicorn
from chembench.baseline import Generation, Generations
from chembench.evaluate import ChemBenchmark
from chembench.prompter import PrompterBuilder
from chembench.task import TopicRegistry
from dotenv import load_dotenv
from tools import brave_search, smiles_to_iupac_name, wikipedia_search, wolfram_alpha


from corral.server import create_benchmark_server

load_dotenv("../.env", override=True)
_CHEMBENCH_TOOLS = [wikipedia_search, brave_search, wolfram_alpha, smiles_to_iupac_name]
benchmark = ChemBenchmark(
    report_dir="reports", verbose=True, state_file="benchmark_state.pkl"
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
    def __init__(self, task_id: str):
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
        example = self.registry.topics["test_2"].tasks[self.task_id]._examples
        prompts, _ = self.dummy_prompter._prompts_with_choices(example)
        return f"\n \n Solve this problem: {prompts[0]}"

    def score(self) -> float:
        """Score based on submitted answer"""
        task_id = self.task_id

        task_registry = self.registry.topics["test_2"]
        task_registry = task_registry.tasks[task_id]
        if self.state.submitted_answer is None:
            return 0.0
        try:
            submitted_result = str(self.state.submitted_answer)
            _prompter = self.get_prompter(submitted_result)
            results = benchmark.bench(task_registry, _prompter, topics=["test_2"])
            print(results)
            return results[0]
        except ValueError:
            return 0.0


if __name__ == "__main__":
    # Create environments for different tasks
    number_task = 4

    environments = {
        "1": ChemBenchEnvironment("1"),
        "2": ChemBenchEnvironment("2"),
        "3": ChemBenchEnvironment("3"),
    }

    # Create and run server
    app = create_benchmark_server(environments)

    uvicorn.run(app, host="0.0.0.0", port=8000)
