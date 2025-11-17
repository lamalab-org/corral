from pathlib import Path

path_ = Path(__file__).parent / "accepted_questions"

unique_files = {}
for file in path_.rglob("*.json"):
    file_name = file.stem
    if file_name in unique_files:
        unique_files[file_name].append(str(file))
    else:
        unique_files[file_name] = []

for file_name, paths in unique_files.items():
    print(file_name)
    for path in paths:
        print(f"\t{path}")

    print("*" * 50)
    print()
