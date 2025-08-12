import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv("../../.env", override=True)
client = OpenAI()

SPECTRA_PATH = Path(
    "../../tasks/spectra_elucidation/spectra_elucidation/tasks_json/task_1.json"
)
ML_PATH = Path("../../tasks/ml/config/single/single.json")
CATALYST_PATH = Path("../../tasks/catalyst/config/single/single.json")
MD_PATH = Path("../../tasks/corral_md/environments/melting/tasks/al.json")


def embed_text(text: str) -> list[float]:
    """
    Embed text using OpenAI's text-embedding-3-small model.
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding


def embed_ml_task(path_path=ML_PATH):
    task = str(path_path).split("/")[3]
    with path_path.open("r") as f:
        ml_tasks = json.load(f)

    embeddings = []
    task_names = []
    for k, v in ml_tasks.items():
        prompt = v["description"]
        task_name = k
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

        embeddings_file = f"embeddings/{task}_tasks_embeddings_full.npy"
        np.save(embeddings_file, embeddings_array)

        task_names_file = f"embeddings/{task}_task_names_full.npy"
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


if __name__ == "__main__":
    embed_spectra()
    embed_ml_task(ML_PATH)
    embed_ml_task(CATALYST_PATH)
    embed_ml_task(MD_PATH)
