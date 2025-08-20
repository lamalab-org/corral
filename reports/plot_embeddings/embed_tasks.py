import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

load_dotenv("../../.env", override=True)
client = OpenAI()

# Model configuration
SIMILARITY_MODEL = "gpt-4o-2024-08-06"  # Model used for similarity analysis
EMBEDDING_MODEL = "text-embedding-3-large"  # Model used for embeddings

# Task paths
SPECTRA_PATH = Path(
    "../../tasks/spectra_elucidation/spectra_elucidation/tasks_json/task_1.json"
)
ML_PATH = Path("../../tasks/ml/config/chained/chained_oxide.json")
CATALYST_PATH = Path("../../tasks/catalyst/config/chained/chained_si.json")
MELTING_PATH = Path("../../tasks/corral_md/environments/melting/subtasks/al.json")
QUENCHING_PATH = Path("../../tasks/corral_md/environments/quenching/subtasks/al.json")
SE_PATH = Path("../../tasks/corral_md/environments/surface_energy/subtasks/al.json")
SPECTRA_FULL_PATH = Path(
    "../../tasks/spectra_elucidation/spectra_elucidation/tasks_json/task_1.json"
)
ML_FULL_PATH = Path("../../tasks/ml/config/single/single.json")
CATALYST_FULL_PATH = Path("../../tasks/catalyst/config/single/single.json")
MELTING_FULL_PATH = Path("../../tasks/corral_md/environments/melting/tasks/al.json")
QUENCHING_FULL_PATH = Path("../../tasks/corral_md/environments/quenching/tasks/al.json")
SE_FULL_PATH = Path("../../tasks/corral_md/environments/surface_energy/tasks/al.json")


# Overarching task descriptions (to be filled by user)
OVERARCHING_SPECTRA_TASK = "Analyze the provided organic compound sample in a lab environment and output the SMILES string, while minimizing resource consumption due to the costly nature of the process."
OVERARCHING_ML_TASK = "Generate a comprehensive dataset of oxide polymorphs from Materials Project and train an XGBoost model to predict formation energies. At the end evaluate the trained XGBoost model using test set metrics, cross-validation. Assess model performance and save results as JSON file with keys `test_set_evaluation` and `cross_validation_results`"
OVERARCHING_CATALYST_TASK = "Create a CO2 adsorbed structure on a Silicon slab. Submit the path to the final combined structure CIF file."
OVERARCHING_MD_TASK = "Simulate the melting of Aluminum by first equilibrating the system under NVT conditions, then relaxing it under NPT conditions, followed by a heating stage under NPT where the temperature is gradually increased to induce melting. Use the Embedded Atom Method (EAM) potential and the given simulation parameters. The simulation should replicate the unit cell 5 times in each direction. Report the final density of the system in g/cm\u00b3"


class SimilarityRating(BaseModel):
    """Structured output for task similarity rating."""

    similarity_score: int  # Rating from 1-10
    reasoning: str  # Explanation for the rating


def embed_text(text: str) -> list[float]:
    """
    Embed text using OpenAI's embedding model.
    """
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def rate_consecutive_similarity(task1_text: str, task2_text: str) -> SimilarityRating:
    """
    Rate the similarity between two consecutive tasks using a structured output.

    Args:
        task1_text: Text description of the first task
        task2_text: Text description of the second task

    Returns:
        SimilarityRating: Structured rating with score (1-10) and reasoning
    """
    prompt = f"""
    Please rate how similar these two consecutive tasks are to each other on a scale of 1-10, where:
    - 1 = Completely different tasks with no overlap
    - 5 = Some shared concepts or methodologies
    - 10 = Nearly identical tasks or very closely related

    Task 1:
    {task1_text}

    Task 2:
    {task2_text}

    Consider factors like:
    - Subject matter overlap
    - Methodological similarity
    - Required expertise/knowledge
    - Problem-solving approaches
    - Expected outputs or goals
    """

    response = client.beta.chat.completions.parse(
        model=SIMILARITY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SimilarityRating,
    )

    return response.choices[0].message.parsed


def rate_subtask_to_overarching_similarity(
    subtask_text: str, overarching_text: str
) -> SimilarityRating:
    """
    Rate how well a subtask aligns with its overarching task using structured output.

    Args:
        subtask_text: Text description of the subtask
        overarching_text: Text description of the overarching task

    Returns:
        SimilarityRating: Structured rating with score (1-10) and reasoning
    """
    prompt = f"""
    Please rate how well this subtask aligns with its overarching task on a scale of 1-10, where:
    - 1 = Subtask is unrelated to the overarching task
    - 5 = Subtask contributes to the overarching goal but with moderate relevance
    - 10 = Subtask is directly and critically aligned with the overarching task

    Overarching Task:
    {overarching_text}

    Subtask:
    {subtask_text}

    Consider factors like:
    - How directly the subtask contributes to the overarching goal
    - Relevance of the subtask's methodology to the main objective
    - Whether the subtask output would be useful for the overarching task
    - Logical progression from subtask to main task
    """

    response = client.beta.chat.completions.parse(
        model=SIMILARITY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SimilarityRating,
    )

    return response.choices[0].message.parsed


def embed_ml_task(path_path=ML_PATH):
    # Extract task name with special handling for corral_md tasks
    path_parts = str(path_path).split("/")
    if "corral_md" in path_parts:
        # For corral_md tasks, just use the environment type
        # e.g., "melting", "quenching", "surface_energy"
        env_index = path_parts.index("environments") + 1
        env_type = path_parts[env_index]
        task = env_type
    else:
        # For other tasks, use the original logic
        task = path_parts[3]

    # Check if this is a FULL path by looking for specific FULL path patterns
    is_full_path = path_path in (
        ML_FULL_PATH,
        CATALYST_FULL_PATH,
        MELTING_FULL_PATH,
        QUENCHING_FULL_PATH,
        SE_FULL_PATH,
    )

    with path_path.open("r") as f:
        ml_tasks = json.load(f)

    embeddings = []
    task_names = []
    for k, v in ml_tasks.items():
        prompt = v["description"]
        task_name = k
        if "aluminum_" in task_name:
            task_name = task_name.replace("aluminum_", "")
        embed_input = prompt
        try:
            embedding = embed_text(embed_input)
            embeddings.append(embedding)
            task_names.append(task_name)
            logger.info(f"    ✓ Successfully embedded {task_name}")
        except Exception as e:
            logger.error(f"Failed to embed task {task_name}: {e}")
            raise ValueError(f"Failed to embed task {task_name}: {e}") from e

    if embeddings:
        embeddings_array = np.array(embeddings)

        # Use different naming convention for FULL paths
        if is_full_path:
            embeddings_file = f"embeddings/{task}_tasks_embeddings_full.npy"
            task_names_file = f"embeddings/{task}_task_names_full.npy"
        else:
            embeddings_file = f"embeddings/{task}_tasks_embeddings.npy"
            task_names_file = f"embeddings/{task}_task_names.npy"

        np.save(embeddings_file, embeddings_array)
        np.save(task_names_file, np.array(task_names))
        logger.info("Embeddings saved successfully.")
    else:
        logger.error("No embeddings were created. Check the input data.")
        raise ValueError("No embeddings were created. Check the input data.")


def embed_spectra():
    with SPECTRA_PATH.open("r") as f:
        spectra_tasks = json.load(f)

    embeddings = []
    task_names = []
    for task in spectra_tasks:
        prompt = task["input"]["prompt"]
        task_name = task["id"].split("_")[-2:]
        task_name = "_".join(task_name)
        embed_input = prompt
        try:
            embedding = embed_text(embed_input)
            embeddings.append(embedding)
            task_names.append(task_name)
            logger.info(f"    ✓ Successfully embedded {task_name}")
        except Exception as e:
            logger.error(f"Failed to embed task {task_name}: {e}")
            raise ValueError(f"Failed to embed task {task_name}: {e}") from e

    if embeddings:
        embeddings_array = np.array(embeddings)

        embeddings_file = "embeddings/spectra_elucidation_tasks_embeddings_full.npy"
        np.save(embeddings_file, embeddings_array)

        task_names_file = "embeddings/spectra_elucidation_task_names_full.npy"
        np.save(task_names_file, np.array(task_names))
        logger.info("Embeddings saved successfully.")

    else:
        logger.error("No embeddings were created. Check the input data.")
        raise ValueError("No embeddings were created. Check the input data.")


def analyze_consecutive_similarity(tasks_data: dict, task_type: str) -> dict:
    """
    Analyze similarity between consecutive tasks using LLM ratings.

    Args:
        tasks_data: Dictionary of tasks with descriptions
        task_type: Type of task (for logging/file naming)

    Returns:
        Dictionary with similarity ratings between consecutive tasks
    """
    task_keys = list(tasks_data.keys())
    similarity_results = {}

    logger.info(f"Analyzing consecutive similarity for {task_type} tasks...")

    for i in range(len(task_keys) - 1):
        task1_key = task_keys[i]
        task2_key = task_keys[i + 1]

        task1_text = tasks_data[task1_key]["description"]
        task2_text = tasks_data[task2_key]["description"]

        try:
            rating = rate_consecutive_similarity(task1_text, task2_text)
            pair_key = f"{task1_key}_to_{task2_key}"
            similarity_results[pair_key] = {
                "similarity_score": rating.similarity_score,
                "reasoning": rating.reasoning,
                "task1": task1_key,
                "task2": task2_key,
            }
            logger.info(
                f"    ✓ Rated similarity between {task1_key} and {task2_key}: {rating.similarity_score}/10"
            )
        except Exception as e:
            logger.error(
                f"Failed to rate similarity between {task1_key} and {task2_key}: {e}"
            )
            continue

    # Save results
    results_file = f"embeddings/{task_type}_consecutive_similarity.json"
    with Path(results_file).open("w") as f:
        json.dump(similarity_results, f, indent=2)

    logger.info(f"Consecutive similarity results saved to {results_file}")
    return similarity_results


def analyze_subtask_overarching_similarity(
    tasks_data: dict, overarching_text: str, task_type: str
) -> dict:
    """
    Analyze similarity between subtasks and their overarching task.

    Args:
        tasks_data: Dictionary of subtasks with descriptions
        overarching_text: Description of the overarching task
        task_type: Type of task (for logging/file naming)

    Returns:
        Dictionary with similarity ratings for each subtask to overarching task
    """
    if not overarching_text.strip():
        logger.warning(
            f"No overarching task text provided for {task_type}. Skipping analysis."
        )
        return {}

    similarity_results = {}

    logger.info(f"Analyzing subtask-to-overarching similarity for {task_type} tasks...")

    for task_key, task_data in tasks_data.items():
        subtask_text = task_data["description"]

        try:
            rating = rate_subtask_to_overarching_similarity(
                subtask_text, overarching_text
            )
            similarity_results[task_key] = {
                "similarity_score": rating.similarity_score,
                "reasoning": rating.reasoning,
                "subtask": task_key,
            }
            logger.info(
                f"    ✓ Rated alignment of {task_key} with overarching task: {rating.similarity_score}/10"
            )
        except Exception as e:
            logger.error(f"Failed to rate alignment of {task_key}: {e}")
            continue

    # Save results
    results_file = f"embeddings/{task_type}_subtask_overarching_similarity.json"
    with Path(results_file).open("w") as f:
        json.dump(similarity_results, f, indent=2)

    logger.info(f"Subtask-overarching similarity results saved to {results_file}")
    return similarity_results


def run_similarity_analysis():
    """
    Run similarity analysis for all task types.
    """
    logger.info("Starting similarity analysis...")

    # ML tasks
    with ML_PATH.open("r") as f:
        ml_tasks = json.load(f)
    analyze_consecutive_similarity(ml_tasks, "ml")
    analyze_subtask_overarching_similarity(ml_tasks, OVERARCHING_ML_TASK, "ml")

    # Catalyst tasks
    with CATALYST_PATH.open("r") as f:
        catalyst_tasks = json.load(f)
    analyze_consecutive_similarity(catalyst_tasks, "catalyst")
    analyze_subtask_overarching_similarity(
        catalyst_tasks, OVERARCHING_CATALYST_TASK, "catalyst"
    )

    # MD tasks
    with MELTING_PATH.open("r") as f:
        md_tasks = json.load(f)
    analyze_consecutive_similarity(md_tasks, "md")
    analyze_subtask_overarching_similarity(md_tasks, OVERARCHING_MD_TASK, "md")

    # Spectra tasks need special handling due to different format
    with SPECTRA_PATH.open("r") as f:
        spectra_data = json.load(f)

    # Convert spectra format to match others
    spectra_tasks = {}
    for task in spectra_data:
        task_id = task["id"].split("_")[-2:]
        task_name = "_".join(task_id)
        spectra_tasks[task_name] = {"description": task["input"]["prompt"]}

    analyze_consecutive_similarity(spectra_tasks, "spectra_elucidation")
    analyze_subtask_overarching_similarity(
        spectra_tasks, OVERARCHING_SPECTRA_TASK, "spectra_elucidation"
    )

    logger.info("Similarity analysis completed!")


if __name__ == "__main__":
    embed_spectra()
    embed_ml_task(ML_PATH)
    embed_ml_task(CATALYST_PATH)
    embed_ml_task(MELTING_PATH)
    embed_ml_task(QUENCHING_PATH)
    embed_ml_task(SE_PATH)

    # Process FULL paths
    embed_ml_task(ML_FULL_PATH)
    embed_ml_task(CATALYST_FULL_PATH)
    embed_ml_task(MELTING_FULL_PATH)
    embed_ml_task(QUENCHING_FULL_PATH)
    embed_ml_task(SE_FULL_PATH)

    # run_similarity_analysis()
