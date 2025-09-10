import json
from pathlib import Path

from loguru import logger

MOLECULES = [
    "CCO",  # Ethanol
]
PRIZES = [100]


def main():
    tasks_path = Path(__file__).parent / "tasks_json"
    for i, molecule in enumerate(MOLECULES):
        task = {
            "id": f"make_{i}",
            "name": f"Synthesize)_{molecule}",
            "keywords": ["chemistry", "synthesis", "retrosynthesis"],
            "metrics": ["binary"],
            "input": {
                "prompt": f"Propose a retrosynthesis route to synthesize the molecule with SMILES {molecule}. The final precursors must be buyable and the total cost must not exceed ${PRIZES[i]}.",
                "input_from_task": False,
                "input_for_task": False,
            },
            "output": [
                {
                    "type": "integer",
                    "target": PRIZES[i],
                    "threshold": None,
                }
            ],
            "scoring_fn": "final_score",
            "submission_format": "Submit a JSON object representing the retrosynthesis route.",
        }
        task_file = tasks_path / f"make_{i}.json"
        with task_file.open("w") as f:
            json.dump(task, f, indent=4)
        logger.info(f"Generated task file: {task_file}")


if __name__ == "__main__":
    main()
