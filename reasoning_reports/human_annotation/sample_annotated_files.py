import logging
import random
import shutil
from pathlib import Path

HUMAN_ANNOTATION_DIR = Path(__file__).resolve().parent
REASONING_REPORTS_DIR = HUMAN_ANNOTATION_DIR.parent
MODELS = ("claude_sonnet_45", "gpt_4o")
DEFAULT_PER_MODEL = 25
DEFAULT_OUTPUT_DIR = HUMAN_ANNOTATION_DIR / "files2annotate"


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def discover_annotated_files(model_dir: Path) -> list[Path]:
    """Return annotated JSON files found under recursive annotated folders."""
    return sorted(model_dir.glob("**/annotated/*.annotated.json"))


def clear_previous_sample(output_dir: Path) -> None:
    """Remove a prior sample set under the dedicated output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_flat_filename(source_path: Path) -> str:
    """Build a flat, collision-safe output filename from the source path."""
    rel = source_path.relative_to(REASONING_REPORTS_DIR)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected annotated file layout: {source_path}")
    model, env, level = parts[0], parts[1], parts[2]
    return f"{model}__{env}__{level}__{source_path.name}"


def copy_sampled_files(sampled_files: list[Path], output_dir: Path) -> int:
    """Copy sampled files into the output directory as flat files."""
    copied = 0
    used_names: set[str] = set()
    for source_path in sampled_files:
        filename = make_flat_filename(source_path)
        if filename in used_names:
            raise ValueError(f"Duplicate destination filename generated: {filename}")
        used_names.add(filename)

        destination_path = output_dir / filename
        shutil.copy2(source_path, destination_path)
        copied += 1
    return copied


def sample_model_files(model: str, k: int, rng: random.Random) -> list[Path]:
    """Sample up to *k* annotated files for a given model."""
    model_dir = REASONING_REPORTS_DIR / model
    files = discover_annotated_files(model_dir)
    if len(files) < k:
        raise ValueError(
            f"Requested {k} files for {model}, but only found {len(files)} annotated files."
        )
    return rng.sample(files, k)


def sample_annotated_files(
    output_dir: str | None = None,
    per_model: int = DEFAULT_PER_MODEL,
    seed: int = 42,
) -> None:
    """Sample annotated files for human review and copy them into one folder.

    The script draws *per_model* files from each supported model and copies
    them directly into `reasoning_reports/human_annotation/files2annotate/`
    as a flat set of 50 JSON files.
    """
    rng = random.Random(seed)
    destination_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    clear_previous_sample(destination_root)

    sampled: list[Path] = []
    summary: dict[str, int] = {}
    for model in MODELS:
        chosen = sample_model_files(model, per_model, rng)
        sampled.extend(chosen)
        summary[model] = len(chosen)

    copied = copy_sampled_files(sampled, destination_root)

    LOGGER.info(
        f"Copied {copied} annotated files into {destination_root} "
        f"({', '.join(f'{model}={count}' for model, count in summary.items())})."
    )


if __name__ == "__main__":
    sample_annotated_files()
