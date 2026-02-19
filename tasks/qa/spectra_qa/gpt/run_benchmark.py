from dotenv import load_dotenv
from litellm import completion
from manege import ManegeBenchmark, ManegeModel, PrompterBuilder, PrompterPipeline
from manege.evaluate import (
    save_topic_reports,
)
from manege.utils import enable_caching, enable_logging

load_dotenv("../../../../.env", override=True)


class GPT4o(ManegeModel):
    """GPT-4o model wrapper for use with manege framework."""

    def __init__(self, name: str = "openai/gpt-4o-2024-08-06"):
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

    llm = GPT4o()
    pipeline = PrompterPipeline()
    pipeline.add_arg("llm_extractor", True)

    prompter = PrompterBuilder.from_model_object(
        model=llm,
        pipeline=pipeline,
    )

    benchmark = ManegeBenchmark.from_directory(data_dir="../tasks_json", verbose=True)

    results = benchmark.bench(prompter)

    save_topic_reports(benchmark, results)
