import json
from pathlib import Path

import pandas as pd

model_name_map = {"gpt": "gpt-4o", "claude": "Claude-4.5", "gpt_oss": "oss"}


def generate_qa_csv(base_dir, envs, output_csv):
    """
    Generates a CSV file containing QA results from the specified directory structure.

    Args:
        base_dir (str): The base directory containing environment subdirectories.
        envs (list): List of environment directory names to scan.
        output_csv (str): Path for the output CSV file.
    """

    data = []

    for env_name in envs:
        env_path = Path(base_dir) / env_name
        if not env_path.is_dir():
            continue

        # Normalize env_id: remove _qa suffix, then _new suffix
        # so afm_qa -> afm, afm_new_qa -> afm, md_new_qa -> md
        env_id = env_name.removesuffix("_qa").removesuffix("_new")

        for model_name in ["claude", "gpt", "gpt_oss"]:
            model_path = env_path / model_name
            if not model_path.is_dir():
                continue

            reports_base_path = model_path / "reports"
            if not reports_base_path.is_dir():
                continue

            for timestamp_dir in reports_base_path.iterdir():
                if timestamp_dir.name == "topic_reports":
                    continue
                if not timestamp_dir.is_dir():
                    continue

                for file_path in timestamp_dir.iterdir():
                    if file_path.suffix == ".json":
                        try:
                            with file_path.open() as f:
                                json_data = json.load(f)

                            item_id = json_data.get("uuid")
                            if not item_id:
                                item_id = file_path.stem

                            if (
                                json_data.get("results")
                                and len(json_data["results"]) > 0
                            ):
                                metrics = json_data["results"][0].get("metrics", {})
                                correct = metrics.get("all_correct")

                                data.append(
                                    {
                                        "item_id": item_id,
                                        "model_id": model_name_map[model_name],
                                        "env_id": env_id,
                                        "correct": correct,
                                    }
                                )
                        except json.JSONDecodeError:
                            pass
                        except Exception:
                            pass

    qa_dataframe = pd.DataFrame(data)
    qa_dataframe.to_csv(output_csv, index=False)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    tasks_dir = script_dir / ".." / "tasks"

    reasoning_qa_envs = [
        "afm",
        "catalyst",
        "corral_md",
        "ml",
        "resistor",
        "retro",
        "spectra",
        "wetlab",
    ]
    generate_qa_csv(
        tasks_dir / "reasoning_qa",
        reasoning_qa_envs,
        script_dir / "reasoning_qa.csv",
    )

    knowledge_qa_envs = [
        "afm_qa",
        "afm_new_qa",
        "catalyst_qa",
        "md_qa",
        "md_new_qa",
        "ml_qa",
        "resistor_qa",
        "retro_qa",
        "spectra_qa",
        "wetlab",
    ]
    generate_qa_csv(
        tasks_dir / "qa",
        knowledge_qa_envs,
        script_dir / "knowledge_qa.csv",
    )
