from dotenv import load_dotenv
from litellm import completion
from manege import ManegeBenchmark, ManegeModel, PrompterBuilder
from manege.evaluate import (
    save_topic_reports,
)
from manege.utils import enable_caching, enable_logging

load_dotenv("../../../.env", override=True)


class Claude(ManegeModel):
    """Claude model wrapper for use with manege framework."""

    def __init__(self, name: str = "anthropic/claude-3-5-haiku-20241022"):
        self.name = name

    def generate(self, prompt: list[list[dict[str, str]]]) -> list[str]:
        generations = []
        for prompt_ in prompt:
            generation = completion(
                model=self.name,
                temperature=0.0,
                messages=prompt_,
            )
            generations.append(generation.choices[0].message.content)
        return generations


if __name__ == "__main__":
    enable_logging()
    enable_caching()

    llm = Claude()

    prompter = PrompterBuilder.from_model_object(
        model=llm,
    )

    benchmark = ManegeBenchmark.from_directory(data_dir="tasks_json", verbose=True)

    results = benchmark.bench(prompter)

    save_topic_reports(benchmark, results)
