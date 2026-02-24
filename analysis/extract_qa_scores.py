"""
Extract overall_score from all QA and reasoning QA summary reports.

Walks tasks/reasoning_qa and tasks/qa, finds summary.json files inside
each model directory, and collects the overall_score along with metadata
(model, environment, category).

Output: analysis/qa_scores.json
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "tasks"

CATEGORY_DIRS = {
    "reasoning": TASKS_DIR / "reasoning_qa",
    "knowledge": TASKS_DIR / "qa",
}

# Directories that are NOT model directories (skip them)
SKIP_DIRS = {"tasks_json", "__pycache__"}
SKIP_FILES = {"create_task_json.py", "tasks.json", "questions.json"}


def collect_scores() -> list[dict]:
    results = []

    for category, base_dir in CATEGORY_DIRS.items():
        if not base_dir.exists():
            logger.warning("%s does not exist, skipping.", base_dir)
            continue

        for env_dir in sorted(base_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env_name = env_dir.name

            for model_dir in sorted(env_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name in SKIP_DIRS:
                    continue
                model_name = model_dir.name

                summary_path = model_dir / "reports" / "topic_reports" / "summary.json"
                if not summary_path.exists():
                    logger.info("No summary found: %s", summary_path.relative_to(ROOT))
                    continue

                with summary_path.open() as f:
                    data = json.load(f)

                overall_score = data.get("overall_score")
                if overall_score is None:
                    logger.warning(
                        "No overall_score in %s",
                        summary_path.relative_to(ROOT),
                    )
                    continue

                results.append(
                    {
                        "category": category,
                        "environment": env_name,
                        "model": model_name,
                        "overall_score": overall_score,
                    }
                )

    return results


def main():
    scores = collect_scores()

    out_path = ROOT / "analysis" / "qa_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(scores, f, indent=2)

    logger.info("Collected %d scores → %s", len(scores), out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
