"""
Fit a model by number (Binomial likelihood).

Usage:
    python fit.py --model 3
    python fit.py --model 3 --data results/prepared_data.csv --output results/
    python fit.py --help
"""

import sys
import time

import fire
import pandas as pd
from loguru import logger
from models import MODEL_NAMES, MODELS
from sampling import check_convergence, sample_model, save_model_results


def main(model, data="results/prepared_data.csv", output="results/", **sampling_kwargs):
    """
    Fit a Bayesian latent factor model (Binomial likelihood).

    Args:
        model: Model number (1-8)
        data: Path to prepared data CSV (aggregated, with k_success/n_trials)
        output: Output directory for results
        **sampling_kwargs: Override sampling config (e.g. draws=1000, chains=2)
    """
    model_num = int(model)
    if model_num not in MODELS:
        logger.error(f"Unknown model {model_num}. Available: {sorted(MODELS.keys())}")
        sys.exit(1)

    model_name = MODEL_NAMES[model_num]
    model_module = MODELS[model_num]

    logger.info(f"Fitting {model_name} (model {model_num}) [Binomial]")

    # Load data
    data_df = pd.read_csv(data)
    logger.info(f"Loaded {len(data_df):,} rows from {data}")

    # Build model
    pm_model = model_module.build_model(data_df)

    # Sample
    start_time = time.time()
    idata = sample_model(pm_model, **sampling_kwargs)
    elapsed = time.time() - start_time
    logger.info(f"Sampling time: {elapsed/60:.1f} minutes")

    # Check convergence
    check_convergence(idata, model_name=model_name)

    # Save
    save_model_results(idata, model_name=model_name, output_dir=output)

    logger.info(f"Done: {model_name}")


if __name__ == "__main__":
    fire.Fire(main)
