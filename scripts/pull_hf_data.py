import argparse
from pathlib import Path

from datasets import load_dataset
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", required=True, choices=["gpt_4o", "claude_sonnet_45", "gpt_oss_120b"]
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--category", required=True, choices=["tasks", "subtasks"])
    parser.add_argument(
        "--agent", required=True, choices=["ReActAgent", "ToolCallingAgent"]
    )
    parser.add_argument("--verbosity", required=True)
    parser.add_argument(
        "--type",
        required=True,
        choices=["traces", "overall_reports", "task_reports"],
    )
    parser.add_argument("--output_dir", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    subset_path = f"{args.model}-{args.env}-{args.level}-{args.category}-{args.agent}-{args.verbosity}-{args.type}"

    # dataset repo
    if args.type == "traces":
        dataset_repo = "jablonkagroup/corral-traces"
    else:
        dataset_repo = "jablonkagroup/corral-reports"

    logger.info(f"Dataset repo : {dataset_repo}")
    logger.info(f"Subset path  : {subset_path}")

    output_path = Path(args.output_dir) / subset_path
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_repo,
        subset_path,
    )

    dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()
