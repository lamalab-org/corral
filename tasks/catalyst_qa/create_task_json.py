import json
import random
import uuid
from pathlib import Path


def convert_mcq_to_task_format(questions):
    """Convert MCQ questions to task format."""
    tasks = []

    for i, question_data in enumerate(questions):
        question_text = question_data["question"]
        options = question_data["options"]

        # Format the input text with question and options
        options_text = ""
        for j, (option_text, _score) in enumerate(options.items(), 1):
            options_text += f"{j}. {option_text}\n"

        input_text = f"Question: {question_text}\n\nOptions:\n{options_text.strip()}"

        # Create target_scores dictionary mapping options to scores
        target_scores = {}
        option_list = list(options.keys())
        random.shuffle(option_list)

        for option_text in option_list:
            target_scores[option_text] = options[option_text]

        # Create the task structure
        task = {
            "name": f"Catalyst- Question {i+1}",
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"Catalyst- Question {i+1}")),
            "description": f"Multiple choice question about Catalyst adsorbate structures and configurations: {question_text[:100]}...",
            "examples": [
                {
                    "input": input_text,
                    "target": None,
                    "target_scores": str(target_scores),
                }
            ],
            "keywords": [
                "materials_science",
                "crystallography",
                "surface_science",
                "computational_chemistry",
                "crystal_structures",
                "adsorption",
                "dft",
                "mcq",
            ],
            "metrics": ["multiple_choice_grade"],
            "preferred_score": "multiple_choice_grade",
        }

        tasks.append(task)

    return tasks


def main():
    # Read the MCQ JSON file
    with Path("questions.json").open() as f:
        mcq_data = json.load(f)
        mcq_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(mcq_data, list):
        questions = mcq_data
    else:
        # If it's a dict, get the first key's value
        questions = next(iter(mcq_data.values()))

    # Convert to task format
    tasks = convert_mcq_to_task_format(questions)
    # Save all tasks in one file
    with Path("catalyst_task.json").open("w") as f:
        json.dump(tasks, f, indent=2)
        json.dump(tasks, f, indent=2)


if __name__ == "__main__":
    main()
