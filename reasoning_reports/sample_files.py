import json
import random
import re
from collections import defaultdict
from pathlib import Path

import fire
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ID = "jablonkagroup/corral-traces"
MODELS = ("claude_sonnet_45", "gpt_4o")
AGENT = "ReActAgent"
VERBOSITY = "brief"
COMPLEXITY = "tasks"

CONFIG_RE = re.compile(
    r"^(?P<model>[^-]+(?:_[^-]+)*)-(?P<env>[^-]+)-level_(?P<level>\d+)"
    r"-(?P<complexity>[^-]+)-(?P<agent>[^-]+)-(?P<verbosity>[^-]+)-traces$"
)

SAVE_COLUMNS = (
    "messages",
    "task_name",
    "model",
    "env",
    "level",
    "category",
    "agent",
    "verbosity",
)


def discover_configs() -> list[dict]:
    """Return parsed metadata for every relevant config in the HF dataset."""
    api = HfApi()
    ds_info = api.dataset_info(DATASET_ID)
    parquet_siblings = [
        s.rfilename
        for s in (ds_info.siblings or [])
        if s.rfilename.endswith(".parquet")
    ]
    configs: list[dict] = []
    seen = set()
    for path in parquet_siblings:
        config_name = path.split("/")[0]
        if config_name in seen:
            continue
        seen.add(config_name)
        m = CONFIG_RE.match(config_name)
        if not m:
            continue
        info = m.groupdict()
        if (
            info["model"] in MODELS
            and info["agent"] == AGENT
            and info["verbosity"] == VERBOSITY
            and info["complexity"] == COMPLEXITY
        ):
            info["config_name"] = config_name
            configs.append(info)
    return configs


def group_by_env(configs: list[dict]) -> dict[str, list[dict]]:
    """Group configs by env name."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        grouped[c["env"]].append(c)
    return dict(grouped)


def sample_diverse(rows: list[dict], k: int) -> list[dict]:
    """Sample *k* rows maximising task_name diversity.

    Takes one row per unique task_name first (shuffled), then fills remaining
    slots from leftover rows at random.
    """
    if k <= 0:
        return []
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_name"]].append(r)

    task_names = list(by_task.keys())
    random.shuffle(task_names)

    sampled: list[dict] = []
    used_indices: set[int] = set()

    for tn in task_names:
        if len(sampled) >= k:
            break
        choice = random.choice(by_task[tn])
        idx = id(choice)
        sampled.append(choice)
        used_indices.add(idx)

    if len(sampled) < k:
        remaining = [r for r in rows if id(r) not in used_indices]
        random.shuffle(remaining)
        sampled.extend(remaining[: k - len(sampled)])

    return sampled[:k]


def sample_env(
    env_configs: list[dict],
    env_name: str,
    n: int,
    output_dir: Path,
) -> int:
    """Sample *n* traces for a single env and save them as JSON files.

    If *n* exceeds the total number of available traces for this env, it is
    clamped to the available amount.  Returns the number of files saved.
    """
    combos = [(c["model"], c["level"], c["config_name"]) for c in env_configs]
    num_combos = len(combos)
    if num_combos == 0:
        logger.info(f"  [{env_name}] No matching configs - skipping.")
        return 0

    combo_rows: list[tuple[str, str, str, list[dict]]] = []
    total_available = 0
    for model, level, config_name in combos:
        logger.info(f"    Loading {config_name} …")
        ds = load_dataset(DATASET_ID, config_name, split="train")
        rows = [dict(row) for row in ds]
        combo_rows.append((model, level, config_name, rows))
        total_available += len(rows)

    effective_n = min(n, total_available)
    if effective_n < n:
        logger.info(
            f"  [{env_name}] Requested n={n} but only {total_available} traces "
            f"available - clamping to {effective_n}."
        )

    per_combo = effective_n // num_combos
    remainder = effective_n % num_combos
    if per_combo == 0:
        logger.info(
            f"  [{env_name}] n={effective_n} < combos={num_combos}; "
            f"sampling 1 from first {effective_n} combos."
        )

    logger.info(
        f"  [{env_name}] {num_combos} (model, level) combos → "
        f"{per_combo} per combo (+{remainder} extra distributed)."
    )

    saved = 0

    for idx, (model, level, config_name, rows) in enumerate(combo_rows):
        target = per_combo + (1 if idx < remainder else 0)
        if target == 0:
            continue

        chosen = sample_diverse(rows, target)
        shortfall = target - len(chosen)
        if shortfall > 0:
            logger.warning(
                f"    ⚠ Only {len(chosen)}/{target} available for {config_name}."
            )

        for row in chosen:
            record = {col: row[col] for col in SAVE_COLUMNS if col in row}
            task = record.get("task_name", "unknown")
            combo_dir = output_dir / model / env_name / f"level_{level}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{task}-{saved}.json"
            out_path = combo_dir / fname
            with out_path.open("w") as f:
                json.dump(record, f, indent=2, default=str)
            saved += 1

    logger.info(f"  [{env_name}] Saved {saved}/{effective_n} traces.")
    return saved


def sample_files(
    output_dir: str | None = None,
    n: int = 10,
    env: str | None = None,
    seed: int | None = None,
) -> None:
    """Sample traces from the HF dataset and save as individual JSON files."""
    if seed is not None:
        random.seed(seed)

    output_path = Path(output_dir) if output_dir else SCRIPT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering configs …")
    all_configs = discover_configs()
    by_env = group_by_env(all_configs)

    if env is not None:
        if env not in by_env:
            available = sorted(by_env.keys())
            raise ValueError(f"Env '{env}' not found. Available envs: {available}")
        by_env = {env: by_env[env]}

    logger.info(f"Envs to process: {sorted(by_env.keys())}")
    total_saved = 0
    for env_name in sorted(by_env.keys()):
        total_saved += sample_env(by_env[env_name], env_name, n, output_path)

    logger.info(f"\nDone - saved {total_saved} trace(s) under '{output_path}'.")


def main(
    output_dir: str | None = None,
    n: int = 90,
    env: str | None = None,
    seed: int | None = None,
) -> None:
    """Sample trace files from the Hugging Face dataset into JSON files.

    The command filters dataset configs to the repository's target model,
    agent, verbosity, and complexity settings, then samples traces per
    environment while favoring task-name diversity across levels and models.

    Args:
        output_dir: Destination directory for sampled JSON files. When not
            provided, files are written beside this script.
        n: Number of traces to sample per environment.
        env: Optional environment name to restrict sampling to a single
            environment, such as "catalyst".
        seed: Optional random seed used to make sampling reproducible.
    """
    sample_files(output_dir=output_dir, n=n, env=env, seed=seed)


if __name__ == "__main__":
    fire.Fire(main)
