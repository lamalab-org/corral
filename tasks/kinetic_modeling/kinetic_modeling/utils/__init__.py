"""
Kinetic modeling utilities module.

This module provides utilities for kinetic modeling including:
- Reaction parsing and validation
- Rate law generation
- ODE system building and solving
- Parameter fitting and optimization
- Model validation and analysis
"""

# Reaction parsing and validation
# ODE generation and solving
from kinetic_modelling.utils.ode_generator import (
    build_ode_system,
    build_stoichiometry_matrix,
    extract_parameters,
    solve_ode_system,
)

# Parameter fitting and optimization
from kinetic_modelling.utils.parameter_fitting import (
    FittingModel,
    calculate_confidence_intervals,
    calculate_goodness_of_fit,
    fit_parameters,
    load_experimental_data,
    objective_function,
    residual_ode,
    square_loss_time_series,
)

# Rate law generation
from kinetic_modelling.utils.rate_laws import calculate_rate, generate_rate_law
from kinetic_modelling.utils.reaction_parser import (
    parse_reactions,
    reaction_string_to_matrix,
)

__all__ = [
    # Parameter fitting
    "FittingModel",
    # ODE generation
    "build_ode_system",
    "build_stoichiometry_matrix",
    "calculate_confidence_intervals",
    "calculate_goodness_of_fit",
    "calculate_rate",
    "extract_parameters",
    "fit_parameters",
    # Rate laws
    "generate_rate_law",
    "load_experimental_data",
    "objective_function",
    # Reaction parsing
    "parse_reactions",
    # ODE solving utilities
    "reaction_string_to_matrix",
    "residual_ode",
    "solve_ode_system",
    "square_loss_time_series",
]
