import argparse
import json
import logging
import random
import shutil
from pathlib import Path

HUMAN_ANNOTATION_DIR = Path(__file__).resolve().parent
REASONING_REPORTS_DIR = HUMAN_ANNOTATION_DIR.parent
REASONING_REPORTS_OLD_DIR = HUMAN_ANNOTATION_DIR.parent.parent / "reasoning_reports_old"
MODELS = ("claude_sonnet_45", "gpt_4o")
DEFAULT_TOTAL = 25
DEFAULT_OUTPUT_DIR = HUMAN_ANNOTATION_DIR / "files2annotate"
DEFAULT_OLD_OUTPUT_DIR = HUMAN_ANNOTATION_DIR / "oldfiles2annotate"

# 50 files used for code-improvement iterations - always excluded from sampling.
EXCLUDED_FILES: set[str] = {
    "claude_sonnet_45__afm__level_2__afm_experiment_level_2-8.annotated.json",
    "claude_sonnet_45__afm__level_3__afm_experiment_level_3-13.annotated.json",
    "claude_sonnet_45__afm__level_4__afm_experiment_level_4-16.annotated.json",
    "claude_sonnet_45__catalyst__level_1__tio2_workflow-12.annotated.json",
    "claude_sonnet_45__catalyst__level_1__tio2_workflow-3.annotated.json",
    "claude_sonnet_45__md__level_1__aluminum_surface_energy_1-3.annotated.json",
    "claude_sonnet_45__md__level_1__aluminum_surface_energy_1-9.annotated.json",
    "claude_sonnet_45__md__level_2__aluminum_surface_energy_2-11.annotated.json",
    "claude_sonnet_45__ml__level_1__ml_sulphides-10.annotated.json",
    "claude_sonnet_45__ml__level_1__ml_sulphides-14.annotated.json",
    "claude_sonnet_45__ml__level_1__ml_sulphides-8.annotated.json",
    "claude_sonnet_45__resistor__level_1__task_0-29.annotated.json",
    "claude_sonnet_45__resistor__level_1__task_2-12.annotated.json",
    "claude_sonnet_45__retrosynthesis__level_2__make_3_lvl2-19.annotated.json",
    "claude_sonnet_45__retrosynthesis__level_3__make_5_lvl3-43.annotated.json",
    "claude_sonnet_45__spectra__level_1__10_15227_orgsyn_084_0077-3.annotated.json",
    "claude_sonnet_45__spectra__level_1__10_15227_orgsyn_096_0036-19.annotated.json",
    "claude_sonnet_45__spectra__level_1__22_22222_orgsyn_222_2222-18.annotated.json",
    "claude_sonnet_45__spectra__level_1__66_66666_orgsyn_666_6666-12.annotated.json",
    "claude_sonnet_45__spectra__level_2__10_15227_orgsyn_084_0215-31.annotated.json",
    "claude_sonnet_45__spectra__level_2__33_33333_orgsyn_333_3333-36.annotated.json",
    "claude_sonnet_45__wetlab__level_1__qualysis_lvl1_08-13.annotated.json",
    "claude_sonnet_45__wetlab__level_1__qualysis_lvl1_08-6.annotated.json",
    "claude_sonnet_45__wetlab__level_3__qualysis_lvl3_08-31.annotated.json",
    "claude_sonnet_45__wetlab__level_3__qualysis_lvl3_09-35.annotated.json",
    "gpt_4o__afm__level_1__afm_experiment_level_1-21.annotated.json",
    "gpt_4o__catalyst__level_1__cu2o_workflow-23.annotated.json",
    "gpt_4o__catalyst__level_1__si_workflow-19.annotated.json",
    "gpt_4o__md__level_1__na2sio3_quenching_1-29.annotated.json",
    "gpt_4o__md__level_1__silicon_melting_1-21.annotated.json",
    "gpt_4o__ml__level_1__ml_nitrides-1.annotated.json",
    "gpt_4o__ml__level_1__ml_oxides-13.annotated.json",
    "gpt_4o__ml__level_1__ml_oxides-14.annotated.json",
    "gpt_4o__resistor__level_1__task_1-43.annotated.json",
    "gpt_4o__resistor__level_1__task_4-42.annotated.json",
    "gpt_4o__resistor__level_1__task_4-45.annotated.json",
    "gpt_4o__retrosynthesis__level_1__make_8_lvl1-50.annotated.json",
    "gpt_4o__retrosynthesis__level_1__make_8_lvl1-55.annotated.json",
    "gpt_4o__retrosynthesis__level_2__make_4_lvl2-63.annotated.json",
    "gpt_4o__spectra__level_1__10_15227_orgsyn_084_0011_sub1-53.annotated.json",
    "gpt_4o__spectra__level_1__10_15227_orgsyn_102_0086-67.annotated.json",
    "gpt_4o__spectra__level_2__10_15227_orgsyn_084_0215-77.annotated.json",
    "gpt_4o__spectra__level_2__66_66666_orgsyn_666_6666-76.annotated.json",
    "gpt_4o__spectra__level_2__77_77777_orgsyn_777_7777-80.annotated.json",
    "gpt_4o__wetlab__level_1__qualysis_lvl1_02-51.annotated.json",
    "gpt_4o__wetlab__level_1__qualysis_lvl1_09-55.annotated.json",
    "gpt_4o__wetlab__level_1__qualysis_lvl1_09-56.annotated.json",
    "gpt_4o__wetlab__level_2__qualysis_lvl2_08-62.annotated.json",
    "gpt_4o__wetlab__level_2__qualysis_lvl2_08-70.annotated.json",
    "gpt_4o__wetlab__level_3__qualysis_lvl3_05-76.annotated.json",
}


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def discover_annotated_files(model_dir: Path) -> list[Path]:
    """Return annotated JSON files found under recursive annotated folders."""
    return sorted(model_dir.glob("**/annotated/*.annotated.json"))


def rel_key(path: Path, base_dir: Path) -> str:
    """Return the path relative to base_dir as a string key for matching."""
    return str(path.relative_to(base_dir))


def clear_previous_sample(output_dir: Path) -> None:
    """Remove a prior sample set under the dedicated output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_flat_filename(source_path: Path, base_dir: Path) -> str:
    """Build a flat, collision-safe output filename from the source path."""
    rel = source_path.relative_to(base_dir)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected annotated file layout: {source_path}")
    model, env, level = parts[0], parts[1], parts[2]
    return f"{model}__{env}__{level}__{source_path.name}"


def copy_sampled_files(
    sampled_files: list[Path], output_dir: Path, base_dir: Path
) -> int:
    """Copy sampled files into the output directory as flat files."""
    copied = 0
    used_names: set[str] = set()
    for source_path in sampled_files:
        filename = make_flat_filename(source_path, base_dir)
        if filename in used_names:
            raise ValueError(f"Duplicate destination filename generated: {filename}")
        used_names.add(filename)

        destination_path = output_dir / filename
        shutil.copy2(source_path, destination_path)
        copied += 1
    return copied


def sample_paired_files(
    total: int = DEFAULT_TOTAL,
    seed: int = 42,
    output_dir: str | None = None,
    old_output_dir: str | None = None,
) -> None:
    """Sample annotated files present in BOTH iterations for paired comparison.

    For each sampled file (by relative path), the new-iteration version is
    copied to `files2annotate` and the old-iteration version to
    `oldfiles2annotate`.  The 50 files used for code improvement (listed
    in `EXCLUDED_FILES`) are always excluded.
    """
    rng = random.Random(seed)
    new_dir = REASONING_REPORTS_DIR
    old_dir = REASONING_REPORTS_OLD_DIR
    dest_new = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dest_old = Path(old_output_dir) if old_output_dir else DEFAULT_OLD_OUTPUT_DIR

    LOGGER.info(f"Excluding {len(EXCLUDED_FILES)} hardcoded improvement files")

    # Discover files in both iterations, keyed by relative path
    new_by_rel: dict[str, dict[str, Path]] = {}
    old_by_rel: dict[str, dict[str, Path]] = {}
    for model in MODELS:
        new_files = discover_annotated_files(new_dir / model)
        old_files = discover_annotated_files(old_dir / model)
        new_by_rel[model] = {rel_key(f, new_dir): f for f in new_files}
        old_by_rel[model] = {rel_key(f, old_dir): f for f in old_files}

    # Keep only files present in both iterations and not excluded
    eligible: dict[str, list[str]] = {}
    for model in MODELS:
        common_keys = sorted(set(new_by_rel[model]) & set(old_by_rel[model]))
        common_keys = [
            k
            for k in common_keys
            if make_flat_filename(new_by_rel[model][k], new_dir) not in EXCLUDED_FILES
        ]
        eligible[model] = common_keys

    # Determine per-model counts
    per = total // len(MODELS)
    remainder = total % len(MODELS)
    counts = {
        model: per + (1 if i < remainder else 0) for i, model in enumerate(MODELS)
    }

    clear_previous_sample(dest_new)
    clear_previous_sample(dest_old)

    total_new = 0
    total_old = 0
    summary: dict[str, int] = {}
    max_nodes = 50
    for model in MODELS:
        k = counts[model]
        pool = eligible[model]
        if len(pool) < k:
            raise ValueError(
                f"Requested {k} files for {model}, but only {len(pool)} "
                f"eligible files exist in both iterations."
            )

        # Count nodes per file and prioritise shorter traces (≤ max_nodes).
        def _node_count(key: str, _model: str = model) -> int:
            try:
                data = json.loads(new_by_rel[_model][key].read_text())
                return len(data.get("nodes", []))
            except Exception:
                return float("inf")

        small_pool = [key for key in pool if _node_count(key) <= max_nodes]
        large_pool = [key for key in pool if key not in set(small_pool)]

        if len(small_pool) >= k:
            # Enough short traces - sample only from them, preferring shorter.
            small_pool.sort(key=_node_count)
            chosen_keys = rng.sample(small_pool, k)
            LOGGER.info(
                f"  {model}: sampled {k} files all with ≤{max_nodes} nodes "
                f"(pool of {len(small_pool)} short traces)"
            )
        else:
            # Take all short traces and fill the rest from larger ones.
            chosen_keys = list(small_pool)
            need = k - len(chosen_keys)
            large_pool.sort(key=_node_count)
            chosen_keys += rng.sample(large_pool, need)
            LOGGER.info(
                f"  {model}: sampled {len(small_pool)} short (≤{max_nodes}) + "
                f"{need} larger traces"
            )
        new_paths = [new_by_rel[model][key] for key in chosen_keys]
        old_paths = [old_by_rel[model][key] for key in chosen_keys]
        total_new += copy_sampled_files(new_paths, dest_new, new_dir)
        total_old += copy_sampled_files(old_paths, dest_old, old_dir)
        summary[model] = k

    model_info = ", ".join(f"{m}={c}" for m, c in summary.items())
    LOGGER.info(
        f"Copied {total_new} new-iteration files into {dest_new} ({model_info})."
    )
    LOGGER.info(
        f"Copied {total_old} old-iteration files into {dest_old} ({model_info})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample paired annotated files for human review"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=DEFAULT_TOTAL,
        help="Total number of files to sample across all models",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for new-iteration files (default: files2annotate)",
    )
    parser.add_argument(
        "--old-output-dir",
        type=str,
        default=None,
        help="Output dir for old-iteration files (default: oldfiles2annotate)",
    )
    args = parser.parse_args()
    sample_paired_files(
        total=args.total,
        seed=args.seed,
        output_dir=args.output_dir,
        old_output_dir=args.old_output_dir,
    )
