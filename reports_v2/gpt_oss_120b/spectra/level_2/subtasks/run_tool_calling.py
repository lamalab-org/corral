import json
import os
from datetime import datetime, timezone
from pathlib import Path

import litellm
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from corral import CorralRouter, CorralRunner
from corral.agents import ToolCallingAgent
from corral.agents.hooks import AgentHooks, HookPoint
from corral.agents.hooks.core import HookContext
from corral.report import CorralWandbLogger


def logprobs_hook(context: HookContext) -> None:
    """Extract, convert to dict, and save logprobs."""
    llm_response = context.iteration_data.get("llm_response")
    if not llm_response:
        return

    log_probs_obj = getattr(llm_response, "logprobs", None)
    if not log_probs_obj:
        return

    # Convert Pydantic object to a standard Python Dict once
    try:
        if hasattr(log_probs_obj, "model_dump"):
            lp_dict = log_probs_obj.model_dump()
        elif hasattr(log_probs_obj, "dict"):
            lp_dict = log_probs_obj.dict()
        else:
            lp_dict = log_probs_obj

        # Store in metadata so other hooks can use the clean DICT
        context.metadata["current_logprobs_dict"] = lp_dict

    except Exception as e:
        logger.error(f"Failed to convert logprobs to dict: {e}")
        return

    # Save to disk as an artifact (optional, but good for records)
    lp_data = {
        "id": llm_response.id,
        "task_id": context.task_id,
        "content": llm_response.content,
        "iteration": context.iteration,
        "logprobs": lp_dict,
    }

    out_dir = Path("logprobs")
    out_dir.mkdir(exist_ok=True, parents=True)
    log_prob_hook_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    context.metadata["logprob_hook_timestamp"] = log_prob_hook_timestamp
    file_path = out_dir / f"{llm_response.id}_{log_prob_hook_timestamp}.json"

    with file_path.open("w") as f:
        json.dump(lp_data, f, indent=2)
    logger.info(f"Saved logprobs to {file_path}")


def compute_logprob_metrics(
    logprobs: dict, log_prob_hook_timestamp
) -> dict[str, float]:
    if not logprobs or "content" not in logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    token_logprobs = [tok["logprob"] for tok in logprobs["content"] if "logprob" in tok]
    if not token_logprobs:
        return {"avg_logprob": 0, "entropy": 0, "variance": 0, "avg_top_entropy": 0}

    avg_logprob = np.mean(token_logprobs)
    probs = np.exp(token_logprobs)
    entropy = -np.sum(probs * np.log(probs + 1e-10)) / len(probs)
    perplexity = np.exp(-avg_logprob)
    variance = np.var(token_logprobs)

    # Optional: Avg entropy and perplexity over top_logprobs per token (for alternative sharpness)
    top_entropies = []
    top_perplexities = []
    # Entropy with tail probability (accounts for probability mass outside top-k)
    eps = 1e-12
    full_entropies = []
    for tok in logprobs["content"]:
        if "top_logprobs" in tok:
            top_lps = [tl["logprob"] for tl in tok["top_logprobs"]]
            if top_lps:
                top_probs = np.exp(top_lps) / np.sum(np.exp(top_lps))  # Normalize
                top_ent = -np.sum(top_probs * np.log(top_probs + 1e-10))
                top_entropies.append(top_ent)
                top_perplexities.append(np.exp(top_ent))  # Perplexity = exp(entropy)

                # Full entropy including tail probability
                lps = np.array(top_lps)
                ps = np.exp(lps)
                ps = np.clip(ps, 0, 1)
                p_other = max(0.0, 1.0 - ps.sum())  # leftover mass outside top-k
                H = -(ps * np.log(ps + eps)).sum() - p_other * np.log(p_other + eps)
                full_entropies.append(H)
    avg_top_entropy = np.mean(top_entropies) if top_entropies else 0
    avg_top_perplexity = np.mean(top_perplexities) if top_perplexities else 0
    avg_full_entropy = np.mean(full_entropies) if full_entropies else 0

    return {
        "avg_logprob": avg_logprob,
        "entropy": entropy,
        "perplexity": perplexity,
        "variance": variance,
        "avg_top_entropy": avg_top_entropy,
        "avg_top_perplexity": avg_top_perplexity,
        "avg_full_entropy": avg_full_entropy,
        "path_length": len(token_logprobs),
        "log_prob_hook_timestamp": log_prob_hook_timestamp,
        # Per-token metrics
        "top_entropies": top_entropies,
        "top_perplexities": top_perplexities,
        "full_entropies": full_entropies,
    }


def save_metrics_to_file(
    task_id: str,
    iteration: int,
    metrics: dict[str, float],
    log_prob_hook_timestamp=None,
) -> None:
    """
    Save computed metrics to a JSON file.

    Args:
        task_id: The current task identifier
        iteration: The iteration number
        metrics: Dictionary of metrics (e.g., {'avg_logprob': -0.5, 'entropy': 1.2, ...})
    """
    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True, parents=True)

    file_path = (
        out_dir / f"{task_id}_iteration_{iteration}_{log_prob_hook_timestamp}.json"
    )
    try:
        with file_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")


def metrics_hook(context: HookContext) -> None:
    """Compute and save metrics using the dict already in memory."""
    # 1. Try to get the pre-converted dict from the previous hook
    lp_dict = context.metadata.get("current_logprobs_dict")

    # 2. Fallback: If logprobs_hook didn't run, try to get it from the response object
    if not lp_dict:
        llm_response = context.iteration_data.get("llm_response")
        if not llm_response or not getattr(llm_response, "logprobs", None):
            return

        # (Conversion logic if needed)
        obj = llm_response.logprobs
        lp_dict = obj.model_dump() if hasattr(obj, "model_dump") else obj

    # 3. Compute metrics using the in-memory dictionary
    log_prob_hook_timestamp = context.metadata["logprob_hook_timestamp"]
    metrics = compute_logprob_metrics(lp_dict, log_prob_hook_timestamp)

    # Store metrics in context for logging or final report
    context.metadata[f"iteration_{context.iteration}_metrics"] = metrics

    # Save metrics to file
    save_metrics_to_file(
        context.task_id, context.iteration, metrics, log_prob_hook_timestamp
    )


def setup_litellm():
    """Setup LiteLLM with appropriate configuration"""
    litellm.set_verbose = True


def run_benchmark(
    model: str = "claude-3-5-sonnet-20241022",
    _task_ids: list | None = None,
    temperature: float = 0,
    run_name: str = "corral_benchmark_run",
    verbose: str = "brief",
    hooks=None,
):
    """Run the benchmark with specified model and tasks"""

    interface = CorralRouter(base_url="http://localhost:1978")
    wandblogger = CorralWandbLogger(
        project="corral_resistor_oss",
        group="gpt_oss",
        name=run_name,
    )
    agent = ToolCallingAgent(
        model=model,
        logprobs=True,
        top_logprobs=20,
        max_iterations=20,
        temperature=temperature,
        api_endpoint="https://api.helmholtz-blablador.fz-juelich.de/v1",
    )
    runner = CorralRunner(
        interface,
        agent,
        logger=wandblogger,
    )

    # Run benchmark
    logger.info(f"Starting benchmark with model: {model}")
    result = runner.bench(
        trials_per_task=5,
        k_values=[1, 2, 3, 4, 5],
        verbose=True,
        tool_verbosity=verbose,
        hooks=hooks,
    )
    result.generate_report(f"{run_name}_try.json")
    logger.info("Benchmark completed")


def rename_output_dirs(verbosity: str, agent: str):
    for dir_name in ["logprobs", "metrics"]:
        src_dir = Path(dir_name)
        if src_dir.exists() and src_dir.is_dir():
            dest_dir = Path(f"{dir_name}_{verbosity}_{agent}")
            src_dir.rename(dest_dir)
            logger.info(f"Renamed {src_dir} to {dest_dir}")


if __name__ == "__main__":
    load_dotenv()
    setup_litellm()
    os.environ["OPENAI_API_KEY"] = os.getenv("BLABLADOR_API_KEY_TEST", "")
    verbosities = ["brief", "workflow", "comprehensive"]
    for verbose in verbosities:
        hooks = AgentHooks()
        hooks.register(HookPoint.AFTER_ITERATION, logprobs_hook)
        hooks.register(HookPoint.AFTER_ITERATION, metrics_hook)

        run_name = f"gpt_oss_120b-tool_calling-spectra_lvl2_env-{verbose}_verbosity"
        model = (
            "openai/1 - GPT-OSS-120b - an open model released by OpenAI in August 2025"
        )
        run_benchmark(model=model, run_name=run_name, verbose=verbose, hooks=hooks)
        # utility function that would rename directory logprobs and metrics to include verbosity level
        rename_output_dirs(verbose, "toolcalling")
