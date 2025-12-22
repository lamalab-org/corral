import json
from pathlib import Path

from loguru import logger


def gather_results() -> None:
    qa_paths = [
        Path("../qa"),
        Path("../../tasks/qa"),
    ]

    results = {}
    for qa_path in qa_paths:
        if not qa_path.exists():
            raise FileNotFoundError(f"QA path not found: {qa_path}")

        for task in qa_path.iterdir():
            task_name = task.name
            for model in task.iterdir():
                model_name = model.name
                if "task" in model_name.lower():
                    continue  # Skip directories that are not models
                for json_file in model.glob("**/*.json"):
                    if "summary" not in json_file.name.lower():
                        continue  # Skip summary files
                    with json_file.open() as f:
                        data = json.load(f)

                    logger.info(
                        f"Processing {task_name} - {model_name} - {json_file.name}"
                    )
                    key = (task_name, model_name)
                    if key not in results:
                        results[key] = data["overall_score"]
    return results


def make_qa_table() -> None:
    results = gather_results()
    output_file = Path(__file__).parent / "tables/qa_results_table.md"
    with output_file.open("w") as f:
        f.write("| Task | Model | Overall Score |\n")
        f.write("|------|-------|---------------|\n")
        for (task, model), score in results.items():
            f.write(f"| {task} | {model} | {score:.2f} |\n")


if __name__ == "__main__":
    make_qa_table()
