from typing import Dict, List
import numpy as np


def _detect_shape(x_values: np.ndarray, y_values: np.ndarray) -> str:
    """
    Detect the qualitative shape of a trend.

    Returns one of:
    - "monotonically increasing"
    - "monotonically decreasing"
    - "bell-shaped" (increases then decreases)
    - "U-shaped" (decreases then increases)
    - "saturating" (increases then plateaus)
    - "sub-linear" (increasing but with decreasing slope)
    - "super-linear" (increasing with increasing slope)
    - "flat" (no significant change)
    - "complex" (doesn't fit simple patterns)
    """
    if len(x_values) < 3:
        return "insufficient data"

    # Normalize y values for comparison
    y_range = np.max(y_values) - np.min(y_values)
    y_mean = np.mean(y_values)

    # Check if essentially flat (less than 10% variation relative to mean)
    if y_mean > 0 and y_range < 0.1 * y_mean:
        return "flat"
    if y_range < 1e-6:
        return "flat"

    # Calculate differences
    dy = np.diff(y_values)

    # Check monotonicity
    all_increasing = np.all(dy >= -0.05 * y_range)  # Allow small noise
    all_decreasing = np.all(dy <= 0.05 * y_range)
    mostly_increasing = np.sum(dy > 0) > 0.7 * len(dy)
    mostly_decreasing = np.sum(dy < 0) > 0.7 * len(dy)

    # Find the index of maximum value
    max_idx = np.argmax(y_values)
    min_idx = np.argmin(y_values)

    # Bell-shaped: max in the middle, increases then decreases
    if 0 < max_idx < len(y_values) - 1:
        left_increasing = np.mean(dy[:max_idx]) > 0 if max_idx > 0 else False
        right_decreasing = np.mean(dy[max_idx:]) < 0 if max_idx < len(dy) else False
        if left_increasing and right_decreasing:
            return "bell-shaped"

    # U-shaped: min in the middle, decreases then increases
    if 0 < min_idx < len(y_values) - 1:
        left_decreasing = np.mean(dy[:min_idx]) < 0 if min_idx > 0 else False
        right_increasing = np.mean(dy[min_idx:]) > 0 if min_idx < len(dy) else False
        if left_decreasing and right_increasing:
            return "U-shaped"

    # Check for saturation (increasing, but last portion is flat)
    if mostly_increasing:
        # Split into first half and second half
        mid = len(dy) // 2
        first_half_slope = np.mean(dy[:mid]) if mid > 0 else 0
        second_half_slope = np.mean(dy[mid:]) if mid < len(dy) else 0

        # Saturating: second half slope much smaller than first half
        if first_half_slope > 0 and second_half_slope >= 0:
            if second_half_slope < 0.3 * first_half_slope:
                return "saturating"
            # Sub-linear: slope decreases but doesn't fully plateau
            elif second_half_slope < 0.7 * first_half_slope:
                return "sub-linear"
            # Super-linear: slope increases
            elif second_half_slope > 1.3 * first_half_slope:
                return "super-linear"

    # Simple monotonic
    if all_increasing or mostly_increasing:
        return "monotonically increasing"
    if all_decreasing or mostly_decreasing:
        return "monotonically decreasing"

    return "complex"


def _compare_magnitude(model_val: float, exp_val: float) -> str:
    """
    Compare model vs experimental value and return qualitative description.
    """
    if exp_val == 0:
        if model_val == 0:
            return "both zero"
        return "model predicts activity where none observed"

    ratio = model_val / exp_val

    if ratio < 0.2:
        return "severely underestimates (model < 20% of experiment)"
    elif ratio < 0.5:
        return f"significantly underestimates (model ~{ratio * 100:.0f}% of experiment)"
    elif ratio < 0.8:
        return "underestimates (model ~{:.0f}% of experiment)".format(ratio * 100)
    elif ratio <= 1.2:
        return "matches well (model ~{:.0f}% of experiment)".format(ratio * 100)
    elif ratio <= 2.0:
        return "overestimates (model ~{:.0f}% of experiment)".format(ratio * 100)
    elif ratio <= 5.0:
        return "significantly overestimates (model ~{:.0f}x experiment)".format(ratio)
    else:
        return f"severely overestimates (model ~{ratio:.0f}x experiment)"


def _assess_trend_qualitatively(
    param_name: str,
    pred_dict: Dict[float, List[float]],
    exp_dict: Dict[float, List[float]],
    param_display_name: str | None = None,
) -> str:
    """
    Generate a qualitative text assessment of how model predictions compare
    to experimental data for a given parameter trend.

    Args:
        param_name: Short name of parameter (e.g., "c_Ru")
        pred_dict: {param_value: [list of predicted rates]}
        exp_dict: {param_value: [list of experimental rates]}
        param_display_name: Human-readable name (e.g., "Ruthenium concentration")

    Returns:
        Multi-line string with qualitative assessment
    """
    if param_display_name is None:
        param_display_name = param_name

    # Get common parameter values
    common_vals = sorted(set(pred_dict.keys()) & set(exp_dict.keys()))

    if len(common_vals) < 2:
        return f"{param_display_name}: Insufficient data points for comparison.\n"

    # Extract mean rates at each parameter value
    x_vals = np.array(common_vals)
    exp_means = np.array([np.mean(exp_dict[v]) for v in common_vals])
    pred_means = np.array([np.mean(pred_dict[v]) for v in common_vals])

    # Detect shapes
    exp_shape = _detect_shape(x_vals, exp_means)
    pred_shape = _detect_shape(x_vals, pred_means)

    # Split into LOW, MID, HIGH regimes (roughly thirds)
    n = len(common_vals)
    if n >= 3:
        low_idx = n // 3
        high_idx = 2 * n // 3

        low_vals = common_vals[: low_idx + 1]
        mid_vals = common_vals[low_idx : high_idx + 1]
        high_vals = common_vals[high_idx:]
    else:
        low_vals = [common_vals[0]]
        high_vals = [common_vals[-1]]
        mid_vals = common_vals

    # Calculate average rates in each regime
    def avg_rate(vals, rate_dict):
        rates = [np.mean(rate_dict[v]) for v in vals if v in rate_dict]
        return np.mean(rates) if rates else 0

    exp_low = avg_rate(low_vals, exp_dict)
    exp_high = avg_rate(high_vals, exp_dict)
    pred_low = avg_rate(low_vals, pred_dict)
    pred_high = avg_rate(high_vals, pred_dict)

    # Compare at low and high ends
    low_comparison = _compare_magnitude(pred_low, exp_low)
    high_comparison = _compare_magnitude(pred_high, exp_high)

    # Build the assessment text
    lines = [
        f"=== {param_display_name} Trend Assessment ===",
        "",
        f"Shape Analysis:",
        f"  - Experimental data shape: {exp_shape}",
        f"  - Model prediction shape: {pred_shape}",
    ]

    # Add shape mismatch commentary
    if exp_shape != pred_shape:
        lines.append(
            f"  - SHAPE MISMATCH: Model shows '{pred_shape}' but data is '{exp_shape}'"
        )

        # Specific diagnostics based on shape mismatch
        if exp_shape == "bell-shaped" and pred_shape in [
            "monotonically increasing",
            "saturating",
        ]:
            lines.append(
                f"  - The model fails to capture INHIBITION at high {param_display_name}"
            )
            lines.append(
                f"  - Consider: self-quenching, substrate inhibition, or competing pathways"
            )
        elif exp_shape == "saturating" and pred_shape == "monotonically increasing":
            lines.append(f"  - The model fails to capture SATURATION behavior")
            lines.append(
                "  - Consider: rate-limiting steps or binding site saturation"
            )
        elif exp_shape == "sub-linear" and pred_shape in [
            "super-linear",
            "monotonically increasing",
        ]:
            lines.append(
                "  - The model shows stronger dependence than experiment at high values"
            )
            lines.append(
                f"  - Consider: limiting reagents or competing consumption pathways"
            )
    else:
        lines.append(f"  - Shape match: Good qualitative agreement")

    lines.extend(
        [
            f"",
            f"Magnitude Comparison:",
            f"  - At LOW {param_display_name} ({low_vals[0]:.4g}): {low_comparison}",
            f"  - At HIGH {param_display_name} ({high_vals[-1]:.4g}): {high_comparison}",
        ]
    )

    # Identify the biggest problem region
    lines.append(f"")
    lines.append(f"Key Discrepancies:")

    # Check each point for discrepancy
    worst_ratio = 1.0
    worst_val = None
    worst_direction = None

    for v in common_vals:
        exp_v = np.mean(exp_dict[v])
        pred_v = np.mean(pred_dict[v])
        if exp_v > 0:
            ratio = pred_v / exp_v
            deviation = abs(ratio - 1.0)
            if deviation > abs(worst_ratio - 1.0):
                worst_ratio = ratio
                worst_val = v
                worst_direction = "over" if ratio > 1 else "under"

    if worst_val is not None and abs(worst_ratio - 1.0) > 0.3:
        lines.append(
            f"  - Worst fit at {param_display_name} = {worst_val:.4g}: model {worst_direction}estimates by {abs(worst_ratio - 1) * 100:.0f}%"
        )
    else:
        lines.append(f"  - No severe point-wise discrepancies (all within 30%)")

    # Add specific actionable insight
    lines.append(f"")
    lines.append(f"Diagnostic Summary:")

    # Generate diagnostic based on pattern
    if "underestimates" in low_comparison and "overestimates" in high_comparison:
        lines.append(
            f"  - Model UNDER-predicts at low [{param_name}] and OVER-predicts at high [{param_name}]"
        )
        lines.append(
            f"  - This suggests the model's dependence on {param_display_name} is TOO STRONG"
        )
    elif "overestimates" in low_comparison and "underestimates" in high_comparison:
        lines.append(
            f"  - Model OVER-predicts at low [{param_name}] and UNDER-predicts at high [{param_name}]"
        )
        lines.append(
            f"  - This suggests the model's dependence on {param_display_name} is TOO WEAK"
        )
    elif "underestimates" in low_comparison and "underestimates" in high_comparison:
        lines.append(
            f"  - Model UNDER-predicts across the entire {param_display_name} range"
        )
        lines.append(
            f"  - This suggests overall reaction rates are too slow or a pathway is missing"
        )
    elif "overestimates" in low_comparison and "overestimates" in high_comparison:
        lines.append(
            f"  - Model OVER-predicts across the entire {param_display_name} range"
        )
        lines.append(
            f"  - This suggests reaction rates are too fast or a consumption pathway is missing"
        )
    elif "matches" in low_comparison and "matches" in high_comparison:
        lines.append(f"  - Model captures the {param_display_name} dependence well")
    else:
        lines.append(f"  - Mixed agreement across the {param_display_name} range")

    lines.append("")
    return "\n".join(lines)


def generate_full_qualitative_assessment(
    trends_pred: Dict[str, Dict[float, List[float]]],
    trends_exp: Dict[str, Dict[float, List[float]]],
) -> str:
    """
    Generate a complete qualitative assessment for all parameter trends.

    Args:
        trends_pred: Dictionary of predicted trends {param: {value: [rates]}}
        trends_exp: Dictionary of experimental trends {param: {value: [rates]}}

    Returns:
        Multi-line string with full qualitative assessment
    """
    param_display_names = {
        "c_Ru": "Ruthenium concentration [Ru]",
        "c_S2O8": "Persulfate concentration [S₂O₈²⁻]",
        "irradiance": "Light irradiance",
        "pH": "pH",
    }

    sections = [
        "=" * 70,
        "QUALITATIVE PHENOMENOLOGICAL TREND ASSESSMENT",
        "=" * 70,
        "",
        "This assessment compares model predictions against experimental data",
        "to identify systematic discrepancies that suggest missing chemistry.",
        "",
    ]

    # Assess each parameter
    for param in ["c_Ru", "c_S2O8", "irradiance", "pH"]:
        pred = trends_pred.get(param, {})
        exp = trends_exp.get(param, {})
        display_name = param_display_names.get(param, param)

        assessment = _assess_trend_qualitatively(param, pred, exp, display_name)
        sections.append(assessment)

    # Add overall summary
    sections.extend(
        [
            "=" * 70,
            "OVERALL SUMMARY",
            "=" * 70,
            "",
            "Based on the above analysis, the key areas where the model disagrees",
            "with experiment are identified above. Use these diagnostics to guide",
            "modifications to the reaction network.",
            "",
        ]
    )

    return "\n".join(sections)
