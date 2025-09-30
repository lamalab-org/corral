# Kinetic Fitting Environment

LLM agent environment for automated kinetic model optimization in photocatalysis research.

## Overview

This environment enables LLM agents to:
- Fit reaction networks to experimental oxygen evolution data
- Optimize both individual fit quality (RSS, R²) and phenomenological trends
- Iteratively modify reaction mechanisms to improve model accuracy
- Provide mechanistic insights into photocatalytic water oxidation

## System

- **Catalyst**: Ru(bpy)₃²⁺ photocatalyst
- **Oxidant**: Persulfate (S₂O₈²⁻)
- **Product**: O₂ evolution
- **Challenge**: Complex reaction network with competing pathways

## Goals

Find minimal reaction network that reproduces experimental trends:
- **[Ru] dependence**: rate peaks at 5-10 µM, then decreases (catalyst deactivation)
- **[S₂O₈²⁻] dependence**: monotonic increase with saturation
- **Irradiance dependence**: linear relationship

## Tools Available

- `describe_experimental_data`: List available datasets with metadata
- `get_current_network`: Show current reaction network
- `fit_single_experiment`: Fit network to one experiment
- `fit_all_experiments`: Fit network to all experiments
- `evaluate_phenomenological_trends`: Score trend reproduction
- `modify_reaction_network`: Add/remove/modify reactions
- `analyze_fit_with_vision`: Use vision model to analyze fit plots

## Target Score

Maximize phenomenological trend score (target: >0.75)