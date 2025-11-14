from pathlib import Path

from loguru import logger

LIMIT = 160


def take_questions(env) -> list[str]:
    root_path = Path(__file__).parent / "reports_v2"
    for model in root_path.iterdir():
        if "claude" not in model.name and "gpt" not in model.name:
            continue
        if model.is_dir():
            env_path = model / env
            if not env_path.exists():
                return [], []
            all_questions = []
            accepted_tasks = []
            for level in env_path.iterdir():
                if level.is_dir():
                    task_path = level / "tasks"
                    if task_path.exists():
                        for logs_dir in task_path.iterdir():
                            if "workflow" not in logs_dir.name:
                                continue
                            tasks = []
                            if logs_dir.is_dir():
                                for json_file in logs_dir.glob("*.json"):
                                    task_id_parts = json_file.stem.split("_")[:-2]
                                    all_questions.append(json_file)
                                    task_id = "_".join(task_id_parts)
                                    if task_id not in tasks:
                                        tasks.append(json_file)
                                        accepted_tasks.append(json_file)

    return accepted_tasks, all_questions


def copy_questions():
    envs = [
        "retrosynthesis",
    ]
    for env in envs:
        accepted_tasks, all_questions = take_questions(env)
        root_path = Path(__file__).parent
        dest_path = root_path / "accepted_questions"
        if not dest_path.exists():
            dest_path.mkdir()
        env_path = dest_path / env
        if not env_path.exists():
            env_path.mkdir()
        for i, question in enumerate(accepted_tasks):
            if i >= LIMIT:
                break
            dest_file = env_path / question.name
            with question.open("r") as src, dest_file.open("w") as dst:
                dst.write(src.read())

        if len(accepted_tasks) < LIMIT:
            for j, question in enumerate(all_questions):
                if i + j >= LIMIT:
                    break
                dest_file = env_path / question.name
                with question.open("r") as src, dest_file.open("w") as dst:
                    dst.write(src.read())

    return accepted_tasks, all_questions


def main():
    accepted_tasks, _all_questions = copy_questions()
    if accepted_tasks:
        logger.info("Copied accepted tasks.")


if __name__ == "__main__":
    main()
