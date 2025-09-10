from dotenv import load_dotenv
from litellm import completion
from stadium import PrompterBuilder, StadiumBenchmark, StadiumModel
from stadium.evaluate import save_topic_reports
from stadium.utils import enable_logging

# import litellm
# litellm.set_verbose = True

# # Optional: JSON-formatted logs
# litellm.json_logs = True

load_dotenv("../../../../.env", override=True)


class Claude(StadiumModel):
    """Claude model wrapper for use with ChemBench framework."""

    def __init__(self, name: str = "anthropic/claude-sonnet-4-20250514"):
        self.name = name

    def generate(self, prompt: list[list[dict[str, str]]]) -> list[str]:
        generations = []
        for prompt_ in prompt:
            generation = completion(
                model=self.name,
                temperature=0.0,
                messages=prompt_,
                max_tokens=64000,
            )
            generations.append(generation.choices[0].message.content)
        return generations


def main():
    """Run all example experiments."""
    enable_logging()
    model = Claude()
    benchmark = StadiumBenchmark.from_directory("../tasks_json", verbose=True)
    prompter = PrompterBuilder.from_model_object(model=model)
    results = benchmark.bench(prompter=prompter)
    save_topic_reports(benchmark, results, "claude_sonnet4")


if __name__ == "__main__":
    main()
