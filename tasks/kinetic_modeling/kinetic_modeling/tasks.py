from kinetic_modeling.score import score_base_model, score_parameters

from corral.base import TaskDefinition


def return_tasks(subtask_level: bool, env_level: int, rag: bool):
    tasks = {}

    if env_level == 1:
        if subtask_level:
            pass
        else:
            # Level 1: Basic Parameter Fitting
            tasks.update(
                {
                    "kinetic_fit_first_order": TaskDefinition(
                        name="kinetic_fit_first_order",
                        description="Fit the rate constant for a simple first-order reaction A -> B. The reaction follows first-order kinetics with respect to A.",
                        tools=[
                            "setup_reaction_network",
                            "derive_rate_law",
                            "generate_ode_system",
                            "fit_kinetic_parameters",
                            "validate_model_fit",
                        ],
                        scoring_fn=score_parameters,
                        scoring_inputs={"k1": 0.025},
                        submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                        input_from_tasks=[],
                        initial_input={"experiment": "results_mrg_059_zn_12_2"},
                    ),
                    "kinetic_fit_second_order": TaskDefinition(
                        name="kinetic_fit_second_order",
                        description="Determine the rate constant for a second-order reaction A + B -> C. Both reactants have equal initial concentrations.",
                        tools=[
                            "setup_reaction_network",
                            "derive_rate_law",
                            "generate_ode_system",
                            "fit_kinetic_parameters",
                            "validate_model_fit",
                        ],
                        scoring_fn=score_parameters,
                        scoring_inputs={"k1": 0.15},
                        submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                        input_from_tasks=[],
                        initial_input={"experiment": "results_mrg_059_zn_12_2"},
                    ),
                }
            )

    elif env_level == 2:
        if subtask_level:
            pass
        else:
            # Level 2: Mechanism Selection
            tasks.update(
                {
                    "consecutive_reactions": TaskDefinition(
                        name="consecutive_reactions",
                        description="Analyze the kinetics of consecutive reactions A -> B -> C. Determine if the mechanism is simple consecutive steps or involves parallel pathways.",
                        tools=[
                            "setup_reaction_network",
                            "derive_rate_law",
                            "generate_ode_system",
                            "fit_kinetic_parameters",
                            "validate_model_fit",
                        ],
                        scoring_fn=score_parameters,
                        scoring_inputs={"k1": 0.05, "k2": 0.03},
                        submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                        input_from_tasks=[],
                        initial_input={"experiment": "results_mrg_059_zn_12_2"},
                    ),
                    "competitive_inhibition": TaskDefinition(
                        name="competitive_inhibition",
                        description="Model enzyme kinetics with competitive inhibition. Determine kinetic parameters for both substrate binding and inhibitor competition.",
                        tools=[
                            "setup_reaction_network",
                            "derive_rate_law",
                            "generate_ode_system",
                            "fit_kinetic_parameters",
                            "validate_model_fit",
                        ],
                        scoring_fn=score_parameters,
                        scoring_inputs={"Km": 2.5, "Vmax": 10.0, "Ki": 1.2},
                        submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                        input_from_tasks=[],
                        initial_input={"experiment": "results_mrg_059_zn_12_2"},
                    ),
                }
            )

    elif env_level == 3:
        if not rag:
            if subtask_level:
                pass
            else:
                # Level 3: Actual experimental Data
                tasks.update(
                    {
                        "jacob_o2_discovery": TaskDefinition(
                            name="jacob_o2_discovery",
                            description="Analyze oxygen concentration time-series from high-throughput photocatalytic experiments.\nThe data shows complex O2 depletion patterns that suggest multiple reaction pathways.\nYour task is to discover the underlying reaction network and kinetic parameters that explain these observations.",
                            tools=[
                                "setup_reaction_network",
                                "derive_rate_law",
                                "generate_ode_system",
                                "fit_kinetic_parameters",
                                "validate_model_fit",
                            ],
                            scoring_fn=score_base_model,
                            scoring_inputs="reference_models.py",
                            submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                            input_from_tasks=[],
                            initial_input={"experiment": "results_mrg_059_zn_12_2"},
                        )
                    }
                )
        else:
            if subtask_level:
                pass
            else:
                # Level 3: Experimental data + RAG tools
                tasks.update(
                    {
                        "jacob_o2_discovery_with_rag": TaskDefinition(
                            name="jacob_o2_discovery_with_rag",
                            description="Analyze oxygen concentration time-series from high-throughput photocatalytic experiments.\nThe data shows complex O2 depletion patterns that suggest multiple reaction pathways.\nYour task is to discover the underlying reaction network and kinetic parameters that explain these observations.",
                            tools=[
                                "setup_reaction_network",
                                "derive_rate_law",
                                "generate_ode_system",
                                "fit_kinetic_parameters",
                                "validate_model_fit",
                                "lookup_previous_works",
                            ],
                            scoring_fn=score_base_model,
                            scoring_inputs="reference_models.py",
                            submission_format="Return a dictionary with the fitted parameters, e.g., {'k1': 0.0}",
                            input_from_tasks=[],
                            initial_input={"experiment": "results_mrg_059_zn_12_2"},
                        )
                    }
                )

    else:
        raise ValueError("Invalid environment level")

    return tasks
