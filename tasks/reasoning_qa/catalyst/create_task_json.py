import json
import uuid
from pathlib import Path


def convert_mcq_to_task_format(questions):
    """Convert MCQ questions to task format and save each as separate file."""
    for i, question_data in enumerate(questions):
        question_text = question_data["question"]
        options = question_data["options"]

        # Format the input text with question and options
        options_text = ""
        option_keys = list(options.keys())

        for j, option_text in enumerate(option_keys, 1):
            options_text += f"{j}. {option_text}\n"

        input_text = f"{question_text}"
        description = str(
            question_data.get(
                "description",
                "Multiple choice reasoning question about catalyst environment agent interactions QA",
            )
        )
        keywords = question_data.get("keywords", [])

        # Create target_scores dictionary mapping options to scores
        target_scores = dict(options.items())

        # Generate UUID for this specific question
        question_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"Catalyst Environment QA {i + 1}")
        )
        source_path = (question_data.get("source", ""),)
        # Create the task structure as a list with single task
        task_data = [
            {
                "name": f"Catalyst Environment Reasoning QA {i + 1}",
                "uuid": question_uuid,
                "description": description,
                "examples": [{"input": input_text, "target_scores": target_scores}],
                "keywords": keywords,
                "metrics": ["multiple_choice_grade"],
                "preferred_score": "multiple_choice_grade",
                "source": source_path,
            }
        ]

        # Save each question as separate JSON file
        filename = f"catalyst_reasoning_question_{i + 1:03d}.json"
        with Path(filename).open("w") as f:
            json.dump(task_data, f, indent=2)


def main():
    # Read the MCQ JSON file
    try:
        with Path("tasks.json").open() as f:
            mcq_data = json.load(f)
    except FileNotFoundError:
        return

    # Handle both list and dict formats
    if isinstance(mcq_data, list):
        questions = mcq_data
    else:
        # If it's a dict, get the first key's value
        questions = next(iter(mcq_data.values()))

    # Convert to task format and save separate files
    convert_mcq_to_task_format(questions)


if __name__ == "__main__":
    main()
