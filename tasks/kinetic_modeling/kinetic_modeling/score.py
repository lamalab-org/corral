## The scoring function should be compare with the original results.


def score_base_model() -> float:
    # TODO add the Jacob model and compare with the scores from that model
    """Score the submitted base model against the ground truth."""
    raise NotImplementedError("score_base_model is not yet implemented")


def score_parameters(submitted: dict, ground_truth: dict) -> float:
    """Score the submitted parameters against the ground truth."""
    if not submitted:
        return 0.0

    if (
        not ground_truth
        or not isinstance(ground_truth, dict)
        or not all(isinstance(v, int | float) for v in ground_truth.values())
        or not all(isinstance(k, str) for k in ground_truth)
    ):
        raise ValueError("Invalid ground truth format")

    # Compare each parameter in the submitted answer with the ground truth
    scores = []
    for key, true_value in ground_truth.items():
        submitted_value = submitted.get(key)
        if submitted_value is not None:
            # Simple comparison (could be improved with more sophisticated metrics)
            scores.append(1.0 if submitted_value == true_value else 0.0)
        else:
            scores.append(0.0)

    # all() expects an iterable of booleans; compare each element to 1.0
    return 1.0 if all(s == 1.0 for s in scores) else 0.0
