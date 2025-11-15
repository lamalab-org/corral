import shutil
from pathlib import Path

repo_path = Path("reports/corral_md_optimised")

root_path = Path(__file__).parent.parent
the_path = root_path / repo_path
current_path = Path(__file__).parent

for path_ in the_path.iterdir():
    if path_.is_dir():
        path_parts = path_.stem.split("-")
        model = path_parts[0]
        agent = path_parts[1]
        nothing = path_parts[2]
        subtask = path_parts[3]
        level = path_parts[4]
        verbosity = path_parts[5]
        if "claude" in model:
            model_dir = "claude_sonnet_45"
        elif "gpt" in model:
            model_dir = "gpt-4o"

        if "false" in subtask:
            subtask_dir = "tasks"
        elif "true" in subtask:
            subtask_dir = "subtasks"
        else:
            raise ValueError(f"Unexpected subtask value: {subtask}")

        if "1" in level:
            level_dir = "level_1"
        elif "2" in level:
            level_dir = "level_2"
        elif "3" in level:
            level_dir = "level_3"
        else:
            raise ValueError(f"Unexpected level value: {level}")

        final_path = current_path / model_dir / "md" / level_dir / subtask_dir

        # Create the destination directory if it doesn't exist
        final_path.mkdir(parents=True, exist_ok=True)

        # Copy all files and directories from path_ to final_path
        for item in path_.iterdir():
            if item.is_dir():
                shutil.copytree(item, final_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, final_path / item.name)
