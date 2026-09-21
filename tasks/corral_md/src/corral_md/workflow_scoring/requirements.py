"""Trace scored checks to the task, never to the adapter's preferred workflow.

Anchors are literal excerpts of the shipped descriptions. Tests ensure they
remain present and that every scored production check has an explicit mapping.
"""

from __future__ import annotations

# Literal excerpts preserve the task's mathematical symbols and punctuation.
# ruff: noqa: RUF001
import json
from functools import lru_cache
from pathlib import Path

GROUPS = {
    1: [
        ("preparation", "Construct a periodic 216-atom system", "diamond_preparation"),
        (
            "thermal_protocol",
            "Under isotropic NPT at zero target pressure, heat continuously",
            "thermal_protocol_and_log",
        ),
        (
            "saved_measurements",
            "For the production trajectory, save the time",
            "production_data_integrity",
        ),
        (
            "continuity",
            "Preserve physical state and elapsed-time continuity",
            "state_and_thermo_continuity",
        ),
        (
            "diffusion",
            "Determine the average self-diffusion coefficient",
            "msd_reconstruction",
            "diffusion_estimator_and_units",
            "reported_diffusion",
        ),
        (
            "equilibrated_liquid",
            "Demonstrate that the analyzed production data represent equilibrated liquid silicon and support a diffusive regime.",
            "equilibrated_production",
            "liquid_and_diffusive_evidence",
            "diffusive_regime",
        ),
    ],
    2: [
        (
            "protocol",
            "Cool continuously from 4000 K to 300 K at 10 K/ps",
            "thermal_cycle",
        ),
        (
            "physics",
            "Use the supplied BKS-type Buckingham-Coulomb potential",
            "saved_physics_and_logs",
        ),
        (
            "continuity",
            "Carry the same system through all stages",
            "boundary_state_continuity",
        ),
        (
            "measurements",
            "Record elapsed time, target and measured temperature, pressure, and density",
            "thermal_observables",
            "boundary_density_consistency",
        ),
        (
            "transition",
            "Choose and document your estimator and the data it uses.",
            "transition_selections",
            "cooling_transition_reproduction",
            "reheating_transition_reproduction",
        ),
        (
            "coordinate",
            "using measured temperature as the temperature coordinate",
            "measured_temperature_coordinate",
        ),
        (
            "results",
            "Report both transition estimates and their signed difference",
            "signed_transition_difference",
            "reported_estimates_and_units",
        ),
        (
            "hold",
            "For the 300 K hold, report representative mean temperature and density and quantify density drift using justified analysis windows.",
            "hold_windows",
            "hold_means",
            "hold_density_drift",
        ),
    ],
    3: [
        (
            "dataset",
            "45 isolated, nonperiodic Ag2 structures",
            "dataset_geometry",
            "dataset_energy_force_labels",
        ),
        (
            "models",
            "Record which input checkpoints you used and where the trained model came from.",
            "reference_identities_and_units",
        ),
        (
            "fitting",
            "Quantify energy agreement per atom and force agreement with the teacher before and after training",
            "before_after_predictions",
            "aggregate_fitting_errors",
            "separation_resolved_errors",
        ),
        (
            "training",
            "Save a loadable trained checkpoint and training history",
            "training_history_and_dataset_identity",
            "checkpoint_container_and_digest",
        ),
        ("bulk_setup", "Build periodic FCC silver", "initial_bulk_geometry"),
        (
            "bulk_md",
            "Keep the cell fixed and use the fine-tuned model for a 5 ps simulation targeting 300 K.",
            "fixed_cell_and_five_ps_trajectory",
        ),
        (
            "initialization",
            "Initialize velocities at 300 K with a recorded random seed, remove center-of-mass momentum once",
            "recorded_md_initialization",
        ),
        (
            "md_evidence",
            "For the MD run, save the cell, positions, momenta, time, potential energy, and forces",
            "md_log_matches_trajectory",
            "md_log_covers_complete_run",
        ),
        (
            "stability",
            "Quantify temperature behavior and assess physical stability using energies, forces, and structural evolution.",
            "temperature_assessment",
            "energy_and_force_assessment",
            "structural_evolution_assessment",
        ),
        (
            "scope",
            "Clearly distinguish fitting diagnostics on training data from evidence of generalization.",
            "diagnostic_scope",
        ),
    ],
    4: [
        ("geometry", "Use a one-atom primitive FCC cell", "fcc_supercell_and_mapping"),
        (
            "settings",
            "Record the checkpoint identity and SHA-256 digest",
            "recorded_model_and_settings",
        ),
        (
            "force_evidence",
            "Keep the raw structures with their energies and full force arrays, along with the force constants.",
            "raw_energy_force_records",
            "derivative_reconstruction",
            "force_constant_transformations",
        ),
        ("bands", "Report the band path Γ–X–U–L–Γ–K", "band_path_and_signed_energies"),
        (
            "dos",
            "Normalize the total DOS to three modes per primitive cell, including imaginary modes.",
            "dos_from_signed_bz_samples",
        ),
        (
            "zpe",
            "For a dynamically stable spectrum, define ZPE",
            "zpe_and_imaginary_mode_accounting",
        ),
        (
            "diagnostics",
            "Report symmetry and acoustic diagnostics.",
            "symmetry_and_acoustic_diagnostics",
        ),
    ],
    5: [
        (
            "geometry",
            "Construct a periodic two-atom primitive diamond cell",
            "reference_primitive_structure",
        ),
        (
            "strain_grid",
            "Evaluate all 25 pairs",
            "complete_strain_grid_and_mapping",
            "cartesian_strains_without_relaxation",
        ),
        (
            "masses",
            "Use natural isotope-averaged silicon masses and record their values.",
            "natural_silicon_masses",
        ),
        (
            "conventions",
            "Document displacement sizes where applicable",
            "recorded_model_units_and_conventions",
        ),
        (
            "derivatives",
            "Keep the evaluated structures, raw forces or analytical derivative results, force constants, and all mode energies.",
            "force_constants_reconstructed_from_raw_evidence",
            "all_mode_energies_and_imaginary_modes",
        ),
        (
            "selected_mode",
            "determine the largest finite, real, strictly positive Gamma-point phonon energy in eV",
            "highest_positive_real_mode",
        ),
        (
            "regression",
            "Fit all 25 energies by unweighted ordinary least squares with an intercept",
            "unweighted_ols_strain_coefficients",
            "unweighted_ols_intercept",
            "all_25_fit_residuals",
        ),
    ],
    6: [
        (
            "geometry",
            "Build a periodic primitive FCC aluminum cell",
            "initial_fcc_geometry",
            "fixed_cell_and_saved_states",
        ),
        (
            "initialization",
            "Initialize Maxwell–Boltzmann velocities at 300 K",
            "initialization_and_recorded_settings",
        ),
        (
            "continuity",
            "continue in the microcanonical (NVE) ensemble from the same cell, positions, and momenta",
            "equilibration_to_nve_continuity",
        ),
        (
            "thermal",
            "Justify equilibration using thermal traces, report the measured production temperature",
            "kinetic_temperature_trace",
            "production_temperature_statistics",
            "production_temperature_sanity",
            "equilibration_thermal_diagnostics",
        ),
        (
            "energy",
            "assess total-energy conservation and any drift during NVE",
            "energy_trace_and_drift",
        ),
        (
            "spectrum",
            "Choose a defensible velocity-based spectral estimator",
            "velocity_spectrum_reproduction",
        ),
        (
            "normalization",
            "Provide a finite, nonnegative spectrum.",
            "finite_nonnegative_normalized_spectrum",
        ),
        (
            "sampling",
            "Report frequencies in THz, the sampling interval, usable frequency range, frequency resolution, and the VDOS normalization and units.",
            "sampling_frequency_range_and_nyquist",
            "spectral_resolution_and_conventions",
            "spectral_normalization_conventions",
        ),
    ],
    7: [
        (
            "preparation",
            "Start from the supplied periodic 108-atom FCC aluminum cell",
            "initial_fcc_geometry",
            "physical_settings_record",
        ),
        (
            "continuous_cycle",
            "Run one continuous isotropic NPT heating-then-cooling sequence",
            "continuous_isotropic_npt_cycle",
        ),
        (
            "stage_measurements",
            "Establish equilibration and assess temperature and pressure control using measured data.",
            "measured_npt_summaries_and_drift",
            "measured_temperature_and_pressure_targets",
        ),
        (
            "interval",
            "Define the main interval coefficient β_volume",
            "interval_expansion_fit",
        ),
        (
            "local_npt",
            "Separately estimate the local volumetric expansion coefficient at 400 K",
            "local_npt_expansion_fit",
        ),
        (
            "pressure_route",
            "Design fixed-cell NVT calculations near (400 K, V*)",
            "fixed_cell_nvt_evidence",
            "local_pressure_derivatives",
        ),
        (
            "pressure",
            "Use total hydrostatic pressure including the kinetic contribution",
            "raw_stress_and_kinetic_pressure",
        ),
        (
            "hysteresis",
            "Quantify heating–cooling volume differences at 300 and 400 K.",
            "heating_cooling_differences",
        ),
        (
            "comparison",
            "Report the signed difference between β_pressure and the local NPT estimate",
            "local_route_comparison",
        ),
        (
            "stability",
            "Treat positive B_T as a stability diagnostic",
            "bulk_modulus_stability_diagnostic",
        ),
    ],
    8: [
        (
            "datasets",
            "Generate two independent test sets",
            "training_and_id_geometry",
            "strained_geometry",
            "distortion_distributions_and_independence",
            "aligned_labels_recorded_inputs_and_rng",
        ),
        (
            "pipeline",
            "Save the final structure-to-energy pipeline",
            "portable_pipeline_and_dimensions",
            "ridge_coefficient_consistency",
        ),
        (
            "label_budget",
            "The training-label budget is these 100 structures",
            "training_partitions_and_label_budget",
        ),
        (
            "preprocessing",
            "Fit preprocessing separately on each validation fold's training portion.",
            "training_only_preprocessing",
        ),
        (
            "validation",
            "Record split indices, candidate settings, validation results, and selection criteria.",
            "validation_predictions_metrics_and_selection",
        ),
        (
            "test_metrics",
            "Report MAE, RMSE, and R²",
            "in_distribution_metrics",
            "strain_resolved_metrics",
            "pooled_strained_metrics",
        ),
        (
            "learning_curve",
            "Build a learning curve using multiple training sizes and reproducible subsets spanning the same distortion distribution",
            "learning_curve_sizes",
            "learning_curve_distortion_coverage",
            "learning_curve_training_only_models",
            "learning_curve_predictions_and_metrics",
            "reported_learning_curve",
        ),
    ],
    9: [
        (
            "trajectories",
            "Retain 200 chronological production frames per run",
            "fixed_cell_cu32_and_chronological_frames",
        ),
        (
            "initialization",
            "Start each run from the original supplied cell with a distinct recorded velocity seed",
            "distinct_initializations_and_recorded_md_settings",
        ),
        (
            "continuity",
            "continue directly from equilibration into production without replacing atoms or reinitializing velocities",
            "equilibration_to_production_boundaries",
        ),
        (
            "labels",
            "Obtain total potential-energy labels from the supplied teacher and compute one global SOAP descriptor per structure",
            "energy_descriptor_alignment_and_teacher_configuration",
        ),
        (
            "temperature",
            "Check equilibration and measured production temperatures for all four runs.",
            "kinetic_temperature_equilibration_and_logs",
            "production_temperature_targets",
        ),
        (
            "correlation",
            "Quantify temporal dependence of the main trajectory's energies using a justified estimator.",
            "temporal_dependence_curve_and_units",
            "characteristic_correlation_time_or_bound",
        ),
        (
            "validation",
            "Compare randomized-split validation with a justified time-aware validation strategy",
            "disjoint_randomized_and_time_aware_holdouts",
        ),
        (
            "preprocessing",
            "fit preprocessing exclusively on each training partition",
            "training_only_outer_preprocessing_and_ridge_consistency",
        ),
        (
            "selection",
            "protect evaluation holdouts, including the randomized holdout, from hyperparameter selection",
            "nested_tuning_protects_both_outer_holdouts",
        ),
        (
            "selection_criterion",
            "Document splits, tuning, and selection criteria.",
            "selection_follows_declared_criterion",
        ),
        (
            "metrics",
            "Report MAE and RMSE in eV and dimensionless R²",
            "reproduced_validation_predictions_and_metrics",
            "independent_800_mae_rmse_r2",
            "independent_500_mae_rmse_r2",
            "independent_1100_mae_rmse_r2",
        ),
        (
            "budgets",
            "Compare strategies at comparable training and evaluation sample budgets",
            "comparable_or_quantified_validation_budgets",
        ),
        (
            "final_pipeline",
            "Freeze a final structure-to-prediction pipeline trained on the main trajectory",
            "final_main_only_affine_pipeline",
            "independent_800_coefficient_predictions",
            "independent_500_coefficient_predictions",
            "independent_1100_coefficient_predictions",
        ),
        (
            "comparisons",
            "Interpret temporal dependence and temperature transfer and support conclusions with evidence",
            "quantified_temporal_and_temperature_transfer_comparisons",
        ),
    ],
    10: [
        (
            "preparation",
            "Build a periodic 108-atom crystal",
            "initial_fcc_and_temperature",
        ),
        (
            "settings",
            "Choose and document the thermostat, integration settings",
            "recorded_nvt_model_and_initialization",
        ),
        (
            "fixed_cell",
            "Keep the cell fixed throughout one continuous NVT trajectory",
            "fixed_cell_and_ordered_atoms",
        ),
        (
            "cycle",
            "preserve stage identifiers and actual cumulative elapsed time",
            "eight_stage_cycle_and_cumulative_time",
        ),
        (
            "continuity",
            "preserve positions and momenta between stages",
            "boundary_position_momentum_continuity",
        ),
        (
            "thermal",
            "Provide equilibration evidence and report overall and directional kinetic temperatures derived from momenta for every stage.",
            "thermal_trace_from_saved_momenta_and_energy",
            "total_and_directional_temperature_means",
            "production_target_temperature_sanity",
            "equilibration_thermal_diagnostics",
        ),
        (
            "energy",
            "Obtain total energy from potential energy plus kinetic energy calculated from momenta.",
            "total_energy_per_atom_means",
        ),
        (
            "fit",
            "fit mean total energy per atom against mean measured temperature using a declared fitting strategy",
            "measured_temperature_heat_capacity_fit",
        ),
        (
            "units",
            "report it in eV/(atom K) and k_B per atom",
            "heat_capacity_units_and_kB_conversion",
        ),
        (
            "residuals",
            "For 800 and 900 K, report energy residuals relative to that fit.",
            "high_temperature_energy_residuals",
        ),
        (
            "return_energy",
            "returning-minus-initial mean energy per atom",
            "return_minus_initial_energy",
        ),
        (
            "return_structure",
            "a defensible structural metric applied consistently",
            "consistent_structural_return_comparison",
        ),
    ],
}


@lru_cache(None)
def task_definition(number):
    path = (
        Path(__file__).resolve().parents[3]
        / "environments"
        / "level_2"
        / "tasks_json"
        / f"task_{number}.json"
    )
    return json.loads(path.read_text())[0]


def requirement_for(number, name):
    task = task_definition(number)
    shared = {
        "readable_manifest",
        "manifest_and_artifacts",
        "recorded_settings",
        "saved_scripts",
        "reported_results",
    }
    if name in shared:
        return {
            "id": f"submission.{name}",
            "source": "submission_format",
            "text": task["submission_format"],
        }
    if name == "remaining_evidence":
        return {
            "id": f"task_{number}.completion",
            "source": "description",
            "text": task["description"],
        }
    for identifier, anchor, *checks in GROUPS[number]:
        if name in checks:
            if anchor not in task["description"]:
                raise ValueError(
                    f"Requirement anchor is absent from task {number}: {anchor}"
                )
            return {
                "id": f"task_{number}.{identifier}",
                "source": "description",
                "text": anchor,
            }
    raise ValueError(f"Scored check lacks a task requirement: task {number}, {name}")


def attach_requirements(rubric):
    for check in rubric.checks:
        if check["points"]:
            check["requirement"] = requirement_for(rubric.task_number, check["name"])
