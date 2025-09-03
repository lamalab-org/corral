def check_numerical(target: float, tolerance: float):
    def score_fn(result: str) -> float:
        # Parse result depending on type
        if isinstance(result, str):
            result = result.strip()
            answer = float(result)

        tol = tolerance * abs(target)
        numerical_ok = (target - tol) <= answer <= (target + tol)
        return 1.0 if numerical_ok else 0.0

    return score_fn
