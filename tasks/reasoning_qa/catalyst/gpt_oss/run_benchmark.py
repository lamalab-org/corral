import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import litellm
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

    def __init__(
        self,
        name: str = "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
    ):
        self.name = name
        self.logprobs_data = []  # Store logprobs for all generations

    def generate(self, prompt: list[list[dict[str, str]]]) -> list[str]:
        generations = []
        for prompt_ in prompt:
            generation = completion(
                model=self.name,
                temperature=0.0,
                messages=prompt_,
                api_base="https://api.helmholtz-blablador.fz-juelich.de/v1",
                logprobs=True,
                top_logprobs=5,  # Get top 5 logprobs per token
            )
            generations.append(
                generation.choices[0].message.content
            ) if generation.choices[0].message.content else generation.append(
                generation.choices[0].message.reasoning_content
            )

            # Extract logprobs data
            logprobs_info = self._extract_logprobs(generation, prompt_)
            self.logprobs_data.append(logprobs_info)

        return generations

    def _extract_logprobs(self, response, prompt) -> dict:
        """Extract logprobs from the response and calculate perplexity."""
        logprobs_content = None
        token_logprobs = []
        tokens = []
        top_logprobs_list = []

        if (
            hasattr(response.choices[0], "logprobs")
            and response.choices[0].logprobs is not None
        ):
            logprobs_obj = response.choices[0].logprobs
            if hasattr(logprobs_obj, "content") and logprobs_obj.content is not None:
                logprobs_content = logprobs_obj.content
                for token_info in logprobs_content:
                    tokens.append(token_info.token)
                    token_logprobs.append(token_info.logprob)
                    # Extract top logprobs for this token
                    if hasattr(token_info, "top_logprobs") and token_info.top_logprobs:
                        top_lps = [
                            {"token": tlp.token, "logprob": tlp.logprob}
                            for tlp in token_info.top_logprobs
                        ]
                        top_logprobs_list.append(top_lps)
                    else:
                        top_logprobs_list.append([])
        # Calculate perplexity per token: Perplexity = exp(-logprob)
        perplexity_per_token = [math.exp(-lp) for lp in token_logprobs]

        # Average perplexity (geometric mean via arithmetic mean of log perplexities)
        if token_logprobs:
            avg_logprob = sum(token_logprobs) / len(token_logprobs)
            avg_perplexity = math.exp(-avg_logprob)
            # Also compute sequence-level perplexity: exp(-1/N * sum(logprobs))
            sequence_perplexity = math.exp(-sum(token_logprobs) / len(token_logprobs))
        else:
            avg_perplexity = None
            sequence_perplexity = None

        return {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs_list,
            "perplexity_per_token": perplexity_per_token,
            "avg_perplexity": avg_perplexity,
            "sequence_perplexity": sequence_perplexity,
            "num_tokens": len(tokens),
        }

    def save_logprobs(self, filepath: str = "logprobs_data.json"):
        """Save all logprobs data to a JSON file."""
        output = {
            "model": self.name,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_generations": len(self.logprobs_data),
            "generations": self.logprobs_data,
            "summary": self._compute_summary(),
        }
        with Path(filepath).open("w") as f:
            json.dump(output, f, indent=2, default=str)

    def _compute_summary(self) -> dict:
        """Compute summary statistics across all generations."""
        all_perplexities = [
            gen["avg_perplexity"]
            for gen in self.logprobs_data
            if gen["avg_perplexity"] is not None
        ]
        all_token_counts = [gen["num_tokens"] for gen in self.logprobs_data]

        if all_perplexities:
            return {
                "total_generations": len(self.logprobs_data),
                "total_tokens": sum(all_token_counts),
                "avg_tokens_per_generation": sum(all_token_counts)
                / len(all_token_counts),
                "mean_perplexity": sum(all_perplexities) / len(all_perplexities),
                "min_perplexity": min(all_perplexities),
                "max_perplexity": max(all_perplexities),
            }
        return {
            "total_generations": len(self.logprobs_data),
            "note": "No logprobs available",
        }


if __name__ == "__main__":
    enable_logging()
    enable_caching()
    litellm.set_verbose = True

    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")
    os.environ["LITELLM_LOG"] = "DEBUG"
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

    # Save logprobs data to a separate JSON file
    llm.save_logprobs("logprobs_perplexity_data.json")
