import arviz as az
import fire
import numpy as np
import pandas as pd
import pymc as pm
from loguru import logger

knowledge_qa = pd.read_csv("results/data/knowledge_qa.csv")
reasoning_qa = pd.read_csv("results/data/reasoning_qa.csv")

agent_data = pd.read_csv("results/data/overall_trace.csv")

# Normalization mappings - standardize QA data to canonical IDs
# QA data uses: "Claude-4.5", "gpt-4o", "oss" for models
# QA data uses: "corral_md", "retro", etc. for envs
# Canonical IDs match reports.jsonl: claude-4.5, gpt-4o, gpt-oss-120b
QA_MODEL_NORMALIZATION = {
    "Claude-4.5": "claude-4.5",  # Normalize capital C to lowercase
    "oss": "gpt-oss-120b",  # Expand oss to full name
}

QA_ENV_NORMALIZATION = {
    "corral_md": "md",  # Normalize variant
}


def fit_irt_model(qa_df):
    """
    qa_df needs columns: model_id, env_id, item_id, correct (0/1)
    """
    # Create categorical codes for proper integer indexing
    qa_df = qa_df.copy()

    # Create integer indices for models, envs, items
    qa_df["model_idx"] = pd.Categorical(qa_df["model_id"]).codes
    qa_df["env_idx"] = pd.Categorical(qa_df["env_id"]).codes
    qa_df["item_idx"] = pd.Categorical(qa_df["item_id"]).codes

    # Create theta index for each (model, env) pair
    qa_df["theta_idx"] = qa_df.groupby(["model_idx", "env_idx"]).ngroup()

    N_theta = qa_df["theta_idx"].nunique()
    N_items = qa_df["item_idx"].nunique()
    N_models = qa_df["model_idx"].nunique()
    N_envs = qa_df["env_idx"].nunique()

    logger.info(
        f"N_models: {N_models}, N_envs: {N_envs}, N_theta: {N_theta}, N_items: {N_items}"
    )

    # Get mappings - these MUST be integer arrays
    theta_to_model = (
        qa_df.groupby("theta_idx")["model_idx"].first().to_numpy().astype(int)
    )
    theta_to_env = qa_df.groupby("theta_idx")["env_idx"].first().to_numpy().astype(int)

    logger.info(f"theta_to_model: {theta_to_model}")
    logger.info(f"theta_to_env: {theta_to_env}")

    with pm.Model():
        # Only add effects if there's variation
        if N_models > 1:
            mu_raw = pm.Normal("mu_raw", 0, 1, shape=N_models - 1)
            mu = pm.Deterministic(
                "mu", pm.math.concatenate([mu_raw, [-pm.math.sum(mu_raw)]])
            )
        else:
            mu = pm.Deterministic("mu", pm.math.constant(np.array([0.0])))

        if N_envs > 1:
            nu_raw = pm.Normal("nu_raw", 0, 1, shape=N_envs - 1)
            nu = pm.Deterministic(
                "nu", pm.math.concatenate([nu_raw, [-pm.math.sum(nu_raw)]])
            )
        else:
            nu = pm.Deterministic("nu", pm.math.constant(np.array([0.0])))

        # Theta
        sigma_theta = pm.HalfNormal("sigma_theta", 0.5)
        theta_mean = mu[theta_to_model] + nu[theta_to_env]
        theta = pm.Normal("theta", mu=theta_mean, sigma=sigma_theta, shape=N_theta)

        # Item parameters
        log_a = pm.Normal("log_a", 0, 0.5, shape=N_items)
        a = pm.Deterministic("a", pm.math.exp(log_a))
        b = pm.Normal("b", 0, 2, shape=N_items)

        # Likelihood - ensure these are integer arrays too
        item_indices = qa_df["item_idx"].to_numpy().astype(int)
        theta_indices = qa_df["theta_idx"].to_numpy().astype(int)

        eta = a[item_indices] * theta[theta_indices] - b[item_indices]
        pm.Bernoulli("y", logit_p=eta, observed=qa_df["correct"].to_numpy())

        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, qa_df


def get_theta_estimates(trace, qa_df):
    """Pull out theta posteriors as a nice dataframe (with raw IDs)."""
    summary = az.summary(trace, var_names=["theta"], hdi_prob=0.9)

    theta_to_model = qa_df.groupby("theta_idx")["model_id"].first()
    theta_to_env = qa_df.groupby("theta_idx")["env_id"].first()

    results = [
        {
            "model": theta_to_model[idx],
            "environment": theta_to_env[idx],
            "theta_mean": summary.iloc[idx]["mean"],
            "theta_sd": summary.iloc[idx]["sd"],
        }
        for idx in range(len(summary))
    ]
    return pd.DataFrame(results)


def filter_extreme_groups(df, min_rate=0.05, max_rate=1):
    grouped = df.groupby(["environment", "scaffold", "level"])

    def is_valid_group(group):
        rate = group["success"].mean()
        return min_rate <= rate <= max_rate

    return grouped.filter(is_valid_group)


def fit_agent_model_no_task_effects(
    agent_df, knowledge_theta_df, reasoning_theta_df, include_category=False
):
    """
    Model without task random effects.
    Uses both knowledge and reasoning theta from IRT.

    logit P(success) = beta0 + lambda_d * knowledge_theta_z
                       + psi_d * reasoning_theta_z
                       + gamma_s + delta_l + xi_v + (kappa_c if include_category)
    """
    agent_df = agent_df.copy()

    # Merge in knowledge theta estimates
    agent_df = agent_df.merge(
        knowledge_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "knowledge_theta"}
        ),
        on=["model", "environment"],
        how="left",
    )

    # Merge in reasoning theta estimates
    agent_df = agent_df.merge(
        reasoning_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "reasoning_theta"}
        ),
        on=["model", "environment"],
        how="left",
    )

    # Check for missing values
    for col in ["knowledge_theta", "reasoning_theta"]:
        missing = agent_df[col].isna().sum()
        if missing > 0:
            logger.info(f"Warning: {missing} rows missing {col}")
            missing_pairs = agent_df[agent_df[col].isna()][
                ["model", "environment"]
            ].drop_duplicates()
            logger.info(f"Missing pairs:\n{missing_pairs}")
    agent_df = agent_df.dropna(subset=["knowledge_theta", "reasoning_theta"])
    logger.info(f"Rows after dropping missing thetas: {len(agent_df)}")

    # Create indices
    agent_df["env_idx"] = pd.Categorical(agent_df["environment"]).codes
    agent_df["scaffold_idx"] = pd.Categorical(agent_df["scaffold"]).codes
    agent_df["level"] = agent_df["level"].str.extract(r"(\d+)").astype(int)
    agent_df["level_idx"] = agent_df["level"] - 1
    agent_df["verbosity_idx"] = pd.Categorical(agent_df["verbosity"]).codes

    if include_category:
        agent_df["category_idx"] = pd.Categorical(agent_df["category"]).codes
        N_categories = agent_df["category_idx"].nunique()
    else:
        N_categories = 0

    N_envs = agent_df["env_idx"].nunique()
    N_scaffolds = agent_df["scaffold_idx"].nunique()
    N_levels = agent_df["level_idx"].nunique()
    N_verbosities = agent_df["verbosity_idx"].nunique()

    if include_category:
        logger.info(
            f"Model dimensions: N_envs={N_envs}, N_scaffolds={N_scaffolds}, "
            f"N_levels={N_levels}, N_verbosities={N_verbosities}, N_categories={N_categories}"
        )
    else:
        logger.info(
            f"Model dimensions: N_envs={N_envs}, N_scaffolds={N_scaffolds}, "
            f"N_levels={N_levels}, N_verbosities={N_verbosities}"
        )

    # Success rate
    success_rate = agent_df["success"].mean()
    init_intercept = np.log(success_rate / (1 - success_rate))
    logger.info(
        f"Success rate: {success_rate:.3f}, init intercept: {init_intercept:.3f}"
    )

    # Standardize thetas
    agent_df["knowledge_z"] = (
        agent_df["knowledge_theta"] - agent_df["knowledge_theta"].mean()
    ) / agent_df["knowledge_theta"].std()
    agent_df["reasoning_z"] = (
        agent_df["reasoning_theta"] - agent_df["reasoning_theta"].mean()
    ) / agent_df["reasoning_theta"].std()
    logger.info(
        f"Knowledge theta: mean={agent_df['knowledge_theta'].mean():.3f}, "
        f"std={agent_df['knowledge_theta'].std():.3f}"
    )
    logger.info(
        f"Reasoning theta: mean={agent_df['reasoning_theta'].mean():.3f}, "
        f"std={agent_df['reasoning_theta'].std():.3f}"
    )

    with pm.Model() as model:
        # Intercept
        beta0 = pm.Normal("beta0", init_intercept, 10)

        # Knowledge loading per environment
        lam = pm.Normal("lambda", 0, 1, shape=N_envs)

        # Reasoning loading per environment
        psi = pm.Normal("psi", 0, 1, shape=N_envs)

        # Scaffold effect
        gamma = pm.Normal("gamma", 0, 1, shape=N_scaffolds)

        # Level effect (harder levels have more negative effects)
        delta_inc = pm.HalfNormal("delta_inc", 0.5, shape=N_levels - 1)
        delta = pm.Deterministic(
            "delta", pm.math.concatenate([[0], -pm.math.cumsum(delta_inc)])
        )

        # Verbosity
        xi = pm.Normal("xi", 0, 0.5, shape=N_verbosities)

        # Category effect (task vs subtask)
        if include_category:
            kappa = pm.Normal("kappa", 0, 1, shape=N_categories)

        # Build linear predictor
        env_idx = agent_df["env_idx"].to_numpy().astype(int)
        scaffold_idx = agent_df["scaffold_idx"].to_numpy().astype(int)
        level_idx = agent_df["level_idx"].to_numpy().astype(int)
        verbosity_idx = agent_df["verbosity_idx"].to_numpy().astype(int)
        knowledge_vals = agent_df["knowledge_z"].to_numpy().astype(float)
        reasoning_vals = agent_df["reasoning_z"].to_numpy().astype(float)

        linpred = (
            beta0
            + lam[env_idx] * knowledge_vals
            + psi[env_idx] * reasoning_vals
            + gamma[scaffold_idx]
            + delta[level_idx]
            + xi[verbosity_idx]
        )

        if include_category:
            category_idx = agent_df["category_idx"].to_numpy().astype(int)
            linpred = linpred + kappa[category_idx]

        pm.Bernoulli("y", logit_p=linpred, observed=agent_df["success"].to_numpy())

        logger.info("\nTesting initial point...")
        try:
            test_point = model.initial_point()
            logp_dict = model.point_logps(test_point)
            total_logp = sum(logp_dict.values())
            logger.info(f"Initial log probability: {total_logp:.2f}")
            if np.isfinite(total_logp):
                logger.info("Initial point is valid!")
                logger.info(f"Component logps: {logp_dict}")
            else:
                logger.info("Warning: Initial point has non-finite logp")
                logger.info(f"Component logps: {logp_dict}")
        except Exception as e:
            logger.info(f"Error during initial point test: {e}")

        trace = pm.sample(
            1000,
            tune=1000,
            chains=4,
            target_accept=0.95,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, agent_df


def variance_decomposition(trace, agent_df, include_task=False, include_category=False):
    """
    Compute how much variance each component explains.
    """
    # Get posterior means
    lam = trace.posterior["lambda"].mean(dim=["chain", "draw"]).to_numpy()
    psi = trace.posterior["psi"].mean(dim=["chain", "draw"]).to_numpy()
    gamma = trace.posterior["gamma"].mean(dim=["chain", "draw"]).to_numpy()
    delta = trace.posterior["delta"].mean(dim=["chain", "draw"]).to_numpy()
    xi = trace.posterior["xi"].mean(dim=["chain", "draw"]).to_numpy()

    # Compute component values for each observation
    knowledge = lam[agent_df["env_idx"]] * agent_df["knowledge_z"]
    reasoning = psi[agent_df["env_idx"]] * agent_df["reasoning_z"]
    scaffold = gamma[agent_df["scaffold_idx"]]
    level = delta[agent_df["level_idx"]]
    verbosity = xi[agent_df["verbosity_idx"]]

    vars_dict = {
        "knowledge": np.var(knowledge),
        "reasoning": np.var(reasoning),
        "scaffold": np.var(scaffold),
        "level": np.var(level),
        "verbosity": np.var(verbosity),
    }

    # Include task random effects if present
    if include_task and "eta" in trace.posterior:
        eta = trace.posterior["eta"].mean(dim=["chain", "draw"]).to_numpy()
        task = eta[agent_df["task_idx"]]
        vars_dict["task"] = np.var(task)

    # Include category effects if present
    if include_category and "kappa" in trace.posterior:
        kappa = trace.posterior["kappa"].mean(dim=["chain", "draw"]).to_numpy()
        category = kappa[agent_df["category_idx"]]
        vars_dict["category"] = np.var(category)

    total = sum(vars_dict.values())

    return {k: 100 * v / total for k, v in vars_dict.items()}


def fit_agent_model_with_task_effects(
    agent_df, knowledge_theta_df, reasoning_theta_df, include_category=False
):
    """
    Full model WITH task-specific random intercepts (eta_t).
    Uses both knowledge and reasoning theta from IRT.

    logit P(success) = beta0 + lambda_d * knowledge_theta_z
                       + psi_d * reasoning_theta_z
                       + gamma_s + delta_l + eta_t + xi_v + (kappa_c if include_category)
    """
    agent_df = agent_df.copy()

    # Merge in knowledge theta estimates
    agent_df = agent_df.merge(
        knowledge_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "knowledge_theta"}
        ),
        on=["model", "environment"],
        how="left",
    )

    # Merge in reasoning theta estimates
    agent_df = agent_df.merge(
        reasoning_theta_df[["model", "environment", "theta_mean"]].rename(
            columns={"theta_mean": "reasoning_theta"}
        ),
        on=["model", "environment"],
        how="left",
    )

    # Check for missing values
    for col in ["knowledge_theta", "reasoning_theta"]:
        missing = agent_df[col].isna().sum()
        if missing > 0:
            logger.info(f"Warning: {missing} rows missing {col}")
    agent_df = agent_df.dropna(subset=["knowledge_theta", "reasoning_theta"])
    logger.info(f"Rows after dropping missing thetas: {len(agent_df)}")

    # Create indices
    agent_df["env_idx"] = pd.Categorical(agent_df["environment"]).codes
    agent_df["task_idx"] = pd.Categorical(agent_df["task"]).codes
    agent_df["scaffold_idx"] = pd.Categorical(agent_df["scaffold"]).codes
    agent_df["level"] = agent_df["level"].str.extract(r"(\d+)").astype(int)
    agent_df["level_idx"] = agent_df["level"] - 1
    agent_df["verbosity_idx"] = pd.Categorical(agent_df["verbosity"]).codes

    if include_category:
        agent_df["category_idx"] = pd.Categorical(agent_df["category"]).codes
        N_categories = agent_df["category_idx"].nunique()
    else:
        N_categories = 0

    N_envs = agent_df["env_idx"].nunique()
    N_tasks = agent_df["task_idx"].nunique()
    N_scaffolds = agent_df["scaffold_idx"].nunique()
    N_levels = agent_df["level_idx"].nunique()
    N_verbosities = agent_df["verbosity_idx"].nunique()

    if include_category:
        logger.info(
            f"Model dimensions: N_envs={N_envs}, N_tasks={N_tasks}, N_scaffolds={N_scaffolds}, "
            f"N_levels={N_levels}, N_verbosities={N_verbosities}, N_categories={N_categories}"
        )
    else:
        logger.info(
            f"Model dimensions: N_envs={N_envs}, N_tasks={N_tasks}, N_scaffolds={N_scaffolds}, "
            f"N_levels={N_levels}, N_verbosities={N_verbosities}"
        )

    # Success rate
    success_rate = agent_df["success"].mean()
    init_intercept = np.log(success_rate / (1 - success_rate))
    logger.info(
        f"Success rate: {success_rate:.3f}, init intercept: {init_intercept:.3f}"
    )

    # Standardize thetas
    agent_df["knowledge_z"] = (
        agent_df["knowledge_theta"] - agent_df["knowledge_theta"].mean()
    ) / agent_df["knowledge_theta"].std()
    agent_df["reasoning_z"] = (
        agent_df["reasoning_theta"] - agent_df["reasoning_theta"].mean()
    ) / agent_df["reasoning_theta"].std()
    logger.info(
        f"Knowledge theta: mean={agent_df['knowledge_theta'].mean():.3f}, "
        f"std={agent_df['knowledge_theta'].std():.3f}"
    )
    logger.info(
        f"Reasoning theta: mean={agent_df['reasoning_theta'].mean():.3f}, "
        f"std={agent_df['reasoning_theta'].std():.3f}"
    )

    with pm.Model() as model:
        # Intercept
        beta0 = pm.Normal("beta0", init_intercept, 10)

        # Knowledge loading per environment
        lam = pm.Normal("lambda", 1, 0.5, shape=N_envs)

        # Reasoning loading per environment
        psi = pm.Normal("psi", 0, 1, shape=N_envs)

        # Scaffold effect
        gamma = pm.Normal("gamma", 0, 1, shape=N_scaffolds)

        # Level effect (enforce ordering via cumulative negative increments)
        delta_inc = pm.HalfNormal("delta_inc", 0.5, shape=N_levels - 1)
        delta = pm.Deterministic(
            "delta", pm.math.concatenate([[0], -pm.math.cumsum(delta_inc)])
        )

        # Task random effects
        sigma_task = pm.HalfNormal("sigma_task", 1)
        eta = pm.Normal("eta", 0, sigma_task, shape=N_tasks)

        # Verbosity
        xi = pm.Normal("xi", 0, 0.5, shape=N_verbosities)

        # Category effect (task vs subtask)
        if include_category:
            kappa = pm.Normal("kappa", 0, 1, shape=N_categories)

        # Build linear predictor
        env_idx = agent_df["env_idx"].to_numpy().astype(int)
        task_idx = agent_df["task_idx"].to_numpy().astype(int)
        scaffold_idx = agent_df["scaffold_idx"].to_numpy().astype(int)
        level_idx = agent_df["level_idx"].to_numpy().astype(int)
        verbosity_idx = agent_df["verbosity_idx"].to_numpy().astype(int)
        knowledge_vals = agent_df["knowledge_z"].to_numpy().astype(float)
        reasoning_vals = agent_df["reasoning_z"].to_numpy().astype(float)

        linpred = (
            beta0
            + lam[env_idx] * knowledge_vals
            + psi[env_idx] * reasoning_vals
            + gamma[scaffold_idx]
            + delta[level_idx]
            + eta[task_idx]
            + xi[verbosity_idx]
        )

        if include_category:
            category_idx = agent_df["category_idx"].to_numpy().astype(int)
            linpred = linpred + kappa[category_idx]

        pm.Bernoulli("y", logit_p=linpred, observed=agent_df["success"].to_numpy())

        logger.info("\nTesting initial point...")
        try:
            test_point = model.initial_point()
            logp_dict = model.point_logps(test_point)
            total_logp = sum(logp_dict.values())
            logger.info(f"Initial log probability: {total_logp:.2f}")
            if np.isfinite(total_logp):
                logger.info("Initial point is valid!")
            else:
                logger.info("Warning: Initial point has non-finite logp")
                logger.info(f"Component logps: {logp_dict}")
        except Exception as e:
            logger.info(f"Error during initial point test: {e}")

        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, agent_df


def fit_agent_model3_abilities_env(
    agent_df, knowledge_theta_df, reasoning_theta_df, include_category=False
):
    """
    Model 3: Environment-specific slopes for abilities + Task random effects
    (From latent_factor_modeling project)

    logit P(success) = B0 + (l_base + t[e])*knowledge_z + (p_base + f[e])*reasoning_z
                       + g[scaffold] + d[level] + n[task] + x[verbosity]
                       + a[environment] + (k[category] if include_category)

    Key features:
    - Base ability slopes (λ_base, ψ_base)
    - Environment-specific slope adjustments (θ[e], φ[e]) with sum-to-zero constraint
    - Task random effects
    - This is the model that achieved r=0.913 LOO correlation
    """
    agent_df = agent_df.copy()

    # Merge in knowledge theta estimates (if not already present, e.g., from balanced_data)
    if "knowledge_theta" not in agent_df.columns:
        agent_df = agent_df.merge(
            knowledge_theta_df[["model", "environment", "theta_mean"]].rename(
                columns={"theta_mean": "knowledge_theta"}
            ),
            on=["model", "environment"],
            how="left",
        )

    # Merge in reasoning theta estimates (if not already present)
    if "reasoning_theta" not in agent_df.columns:
        agent_df = agent_df.merge(
            reasoning_theta_df[["model", "environment", "theta_mean"]].rename(
                columns={"theta_mean": "reasoning_theta"}
            ),
            on=["model", "environment"],
            how="left",
        )

    # Check for missing values
    for col in ["knowledge_theta", "reasoning_theta"]:
        missing = agent_df[col].isna().sum()
        if missing > 0:
            logger.info(f"Warning: {missing} rows missing {col}")
    agent_df = agent_df.dropna(subset=["knowledge_theta", "reasoning_theta"])
    logger.info(f"Rows after dropping missing thetas: {len(agent_df)}")

    # Create indices
    agent_df["env_idx"] = pd.Categorical(agent_df["environment"]).codes
    agent_df["task_idx"] = pd.Categorical(agent_df["task"]).codes
    agent_df["scaffold_idx"] = pd.Categorical(agent_df["scaffold"]).codes
    agent_df["level"] = agent_df["level"].str.extract(r"(\d+)").astype(int)
    agent_df["level_idx"] = agent_df["level"] - 1
    agent_df["verbosity_idx"] = pd.Categorical(agent_df["verbosity"]).codes

    if include_category:
        agent_df["category_idx"] = pd.Categorical(agent_df["category"]).codes
        N_categories = agent_df["category_idx"].nunique()
    else:
        N_categories = 0

    N_envs = agent_df["env_idx"].nunique()
    N_tasks = agent_df["task_idx"].nunique()
    N_scaffolds = agent_df["scaffold_idx"].nunique()
    N_levels = agent_df["level_idx"].nunique()
    N_verbosities = agent_df["verbosity_idx"].nunique()

    logger.info("=" * 60)
    logger.info("MODEL 3: Environment-Specific Ability Slopes + Tasks")
    logger.info("=" * 60)
    logger.info(f"Environments:    {N_envs}")
    logger.info(f"Tasks:           {N_tasks}")
    logger.info(f"Scaffolds:       {N_scaffolds}")
    logger.info(f"Levels:          {N_levels}")
    logger.info(f"Verbosities:     {N_verbosities}")
    if include_category:
        logger.info(f"Categories:      {N_categories}")

    # Success rate
    success_rate = agent_df["success"].mean()
    init_intercept = np.log(success_rate / (1 - success_rate))
    logger.info(f"Success rate:    {success_rate:.3f}")
    logger.info(f"Init intercept:  {init_intercept:.3f}")

    # Standardize thetas
    agent_df["knowledge_z"] = (
        agent_df["knowledge_theta"] - agent_df["knowledge_theta"].mean()
    ) / agent_df["knowledge_theta"].std()
    agent_df["reasoning_z"] = (
        agent_df["reasoning_theta"] - agent_df["reasoning_theta"].mean()
    ) / agent_df["reasoning_theta"].std()

    with pm.Model() as model:
        # Intercept
        beta0 = pm.Normal("beta0", init_intercept, 10)

        # Base ability slopes (global effects)
        lambda_base = pm.Normal("lambda_base", 0, 1)
        psi_base = pm.Normal("psi_base", 0, 1)

        # Environment-specific slope adjustments (sum-to-zero)
        theta_raw = pm.Normal("theta_raw", 0, 0.5, shape=N_envs)
        theta = pm.Deterministic("theta", theta_raw - theta_raw.mean())

        phi_raw = pm.Normal("phi_raw", 0, 0.5, shape=N_envs)
        phi = pm.Deterministic("phi", phi_raw - phi_raw.mean())

        # Configuration effects (sum-to-zero constraints)
        gamma_raw = pm.Normal("gamma_raw", 0, 1, shape=N_scaffolds)
        gamma = pm.Deterministic("gamma", gamma_raw - gamma_raw.mean())

        delta_raw = pm.Normal("delta_raw", 0, 1, shape=N_levels)
        delta = pm.Deterministic("delta", delta_raw - delta_raw.mean())

        xi_raw = pm.Normal("xi_raw", 0, 1, shape=N_verbosities)
        xi = pm.Deterministic("xi", xi_raw - xi_raw.mean())

        # Hierarchical environment effects
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        alpha_raw = pm.Normal("alpha_raw", 0, 1, shape=N_envs)
        alpha = pm.Deterministic("alpha", sigma_alpha * alpha_raw)

        # Hierarchical task effects
        sigma_task = pm.HalfNormal("sigma_task", 0.5)
        eta_raw = pm.Normal("eta_raw", 0, 1, shape=N_tasks)
        eta = pm.Deterministic("eta", sigma_task * eta_raw)

        # Category effect (task vs subtask)
        if include_category:
            kappa_raw = pm.Normal("kappa_raw", 0, 1, shape=N_categories)
            kappa = pm.Deterministic("kappa", kappa_raw - kappa_raw.mean())

        # Build linear predictor with environment-varying ability slopes
        env_idx = agent_df["env_idx"].to_numpy().astype(int)
        task_idx = agent_df["task_idx"].to_numpy().astype(int)
        scaffold_idx = agent_df["scaffold_idx"].to_numpy().astype(int)
        level_idx = agent_df["level_idx"].to_numpy().astype(int)
        verbosity_idx = agent_df["verbosity_idx"].to_numpy().astype(int)
        knowledge_vals = agent_df["knowledge_z"].to_numpy().astype(float)
        reasoning_vals = agent_df["reasoning_z"].to_numpy().astype(float)

        # Total ability slopes by environment
        lambda_total = lambda_base + theta[env_idx]
        psi_total = psi_base + phi[env_idx]

        linpred = (
            beta0
            + lambda_total * knowledge_vals
            + psi_total * reasoning_vals
            + gamma[scaffold_idx]
            + delta[level_idx]
            + eta[task_idx]
            + xi[verbosity_idx]
            + alpha[env_idx]
        )

        if include_category:
            category_idx = agent_df["category_idx"].to_numpy().astype(int)
            linpred = linpred + kappa[category_idx]

        pm.Bernoulli("y", logit_p=linpred, observed=agent_df["success"].to_numpy())

        logger.info("\nTesting initial point...")
        try:
            test_point = model.initial_point()
            logp_dict = model.point_logps(test_point)
            total_logp = sum(logp_dict.values())
            logger.info(f"Initial log probability: {total_logp:.2f}")
            if np.isfinite(total_logp):
                logger.info("Initial point is valid!")
            else:
                logger.info("Warning: Initial point has non-finite logp")
                logger.info(f"Component logps: {logp_dict}")
        except Exception as e:
            logger.info(f"Error during initial point test: {e}")

        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, agent_df


def main(
    with_task_effects: bool = False,
    with_category: bool = False,
    use_model3: bool = False,
    use_balanced_data: bool = False,
    output_dir: str = "./results",
):
    """Latent Factor Model for Agent Benchmarks.

    Args:
        with_task_effects: Include task random effects (η_t)
        with_category: Include category effects (κ_c for task vs subtask)
        use_model3: Use Model 3 (environment-specific ability slopes + tasks)
                    This overrides with_task_effects/with_category flags
        use_balanced_data: Use balanced_data.csv instead of overall_trace.csv
                           (Recommended for Model 3)
        output_dir: Output directory for saved results
    """
    import json
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Stage 1a: IRT Model on Knowledge Q&A Data
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1a: Fitting IRT Model on Knowledge Q&A Data")
    logger.info("=" * 60)

    # Normalize QA data to canonical IDs
    knowledge_qa["model_id"] = knowledge_qa["model_id"].replace(QA_MODEL_NORMALIZATION)
    knowledge_qa["env_id"] = knowledge_qa["env_id"].replace(QA_ENV_NORMALIZATION)

    knowledge_irt_trace, knowledge_qa_df = fit_irt_model(knowledge_qa)
    knowledge_theta_df = get_theta_estimates(
        knowledge_irt_trace,
        knowledge_qa_df,
    )

    logger.info("\nKnowledge theta estimates:")
    logger.info(knowledge_theta_df)
    knowledge_theta_df.to_csv(f"{output_dir}/knowledge_theta.csv", index=False)
    logger.info(f"Saved {output_dir}/knowledge_theta.csv")

    # ==========================================================================
    # Stage 1b: IRT Model on Reasoning Q&A Data
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1b: Fitting IRT Model on Reasoning Q&A Data")
    logger.info("=" * 60)

    # Normalize QA data to canonical IDs
    reasoning_qa["model_id"] = reasoning_qa["model_id"].replace(QA_MODEL_NORMALIZATION)
    reasoning_qa["env_id"] = reasoning_qa["env_id"].replace(QA_ENV_NORMALIZATION)

    reasoning_irt_trace, reasoning_qa_df = fit_irt_model(reasoning_qa)
    reasoning_theta_df = get_theta_estimates(
        reasoning_irt_trace,
        reasoning_qa_df,
    )

    logger.info("\nReasoning theta estimates:")
    logger.info(reasoning_theta_df)
    reasoning_theta_df.to_csv(f"{output_dir}/reasoning_theta.csv", index=False)
    logger.info(f"Saved {output_dir}/reasoning_theta.csv")

    # ==========================================================================
    # Prepare Agent Data
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Preparing Agent Data")
    logger.info("=" * 60)

    # Load data (balanced or full)
    if use_balanced_data:
        logger.info("Using balanced dataset (recommended for Model 3)")
        agent_df_raw = pd.read_csv("results/balanced_data.csv")
        logger.info(f"Loaded balanced data: {len(agent_df_raw):,} rows")
    else:
        logger.info("Using full dataset")
        agent_df_raw = agent_data
        logger.info(f"Loaded full data: {len(agent_df_raw):,} rows")

    filtered_agent_data = agent_df_raw[agent_df_raw["environment"] != "wetlab"]

    # Agent data should already be normalized from prepare_agent_data.py

    logger.info("Models:", filtered_agent_data["model"].unique())
    logger.info("Environments:", filtered_agent_data["environment"].unique())

    # Filter extreme groups (skip if using balanced data, already balanced)
    if use_balanced_data:
        agent_data_filtered = filtered_agent_data
        logger.info(
            f"Using balanced data (no extreme group filtering): {len(agent_data_filtered)} rows"
        )
    else:
        agent_data_filtered = filter_extreme_groups(filtered_agent_data)
        logger.info(f"Filtered data: {len(agent_data_filtered)} rows")

    # ==========================================================================
    # Stage 2: Agent Model
    # ==========================================================================
    logger.info("\n" + "=" * 60)

    if use_model3:
        model_desc = (
            "STAGE 2: Fitting MODEL 3 (Environment-Specific Ability Slopes + Tasks)"
        )
        logger.info(model_desc)
        logger.info("=" * 60)

        agent_trace, agent_df = fit_agent_model3_abilities_env(
            agent_df=agent_data_filtered,
            knowledge_theta_df=knowledge_theta_df,
            reasoning_theta_df=reasoning_theta_df,
            include_category=with_category,
        )
        # Model 3 has different parameter structure (lambda_base, theta, etc.)
        # Skip variance decomposition for now - it would need to be rewritten for Model 3
        var_result = None
        logger.info(
            "Skipping variance decomposition for Model 3 (different parameter structure)"
        )
    else:
        model_desc = "STAGE 2: Fitting Agent Model"
        if with_task_effects and with_category:
            model_desc += " WITH Task Effects AND Category Effects"
        elif with_task_effects:
            model_desc += " WITH Task Effects"
        elif with_category:
            model_desc += " WITH Category Effects"
        else:
            model_desc += " (baseline: no task or category effects)"
        logger.info(model_desc)
        logger.info("=" * 60)

        if with_task_effects:
            agent_trace, agent_df = fit_agent_model_with_task_effects(
                agent_df=agent_data_filtered,
                knowledge_theta_df=knowledge_theta_df,
                reasoning_theta_df=reasoning_theta_df,
                include_category=with_category,
            )
            var_result = variance_decomposition(
                agent_trace, agent_df, include_task=True, include_category=with_category
            )
        else:
            agent_trace, agent_df = fit_agent_model_no_task_effects(
                agent_df=agent_data_filtered,
                knowledge_theta_df=knowledge_theta_df,
                reasoning_theta_df=reasoning_theta_df,
                include_category=with_category,
            )
            var_result = variance_decomposition(
                agent_trace,
                agent_df,
                include_task=False,
                include_category=with_category,
            )

    # ==========================================================================
    # Save all results
    # ==========================================================================
    if var_result is not None:
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS: Variance Decomposition")
        logger.info("=" * 60)
        for component, pct in sorted(var_result.items(), key=lambda x: -x[1]):
            logger.info(f"  {component:12s}: {pct:6.2f}%")

    # Save trace and agent_df for plotting
    agent_trace.to_netcdf(f"{output_dir}/agent_trace.nc")
    agent_df.to_csv(f"{output_dir}/agent_df.csv", index=False)

    results_path = f"{output_dir}/results.json"
    from pathlib import Path

    with Path(results_path).open("w") as f:
        json.dump(
            {
                "variance_decomposition": var_result if var_result is not None else {},
                "knowledge_theta": knowledge_theta_df.to_dict(orient="records"),
                "reasoning_theta": reasoning_theta_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("  - agent_trace.nc, agent_df.csv, results.json")
    logger.info("  - knowledge_theta.csv, reasoning_theta.csv")
    logger.info("Run plot_results.py to generate plots.")


if __name__ == "__main__":
    fire.Fire(main)
