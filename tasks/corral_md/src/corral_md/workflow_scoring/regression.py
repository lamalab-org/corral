"""Numerical checks for exported linear models; no training or deserialization."""

from __future__ import annotations

import numpy as np

from .common import EvidenceError, UnsupportedEvidence, finite_array


def metrics(reference, prediction) -> dict:
    y = finite_array(reference, ndim=1)
    p = finite_array(prediction, shape=y.shape)
    residual = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    # Match sklearn's finite convention for a constant reference target.
    r2 = (
        1 - np.sum(residual**2) / denominator
        if denominator > 0
        else float(np.allclose(y, p, rtol=0, atol=0))
    )
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2),
    }


def transformed_features(features, model: dict) -> np.ndarray:
    """Evaluate portable preprocessing without executing submitted code."""
    x = finite_array(features, ndim=2)
    d = x.shape[1]
    kind = model.get("preprocessing", "StandardScaler")
    if kind == "MinMaxScaler":
        scale = finite_array(model["scale"], shape=(d,))
        offset = finite_array(model["offset"], shape=(d,))
        if np.any(scale <= 0):
            raise EvidenceError("MinMaxScaler scales must be positive")
        result = x * scale + offset
        if type(model.get("clip", False)) is not bool:
            raise EvidenceError("MinMaxScaler clip must be boolean")
        if model.get("clip", False):
            bounds = finite_array(model.get("feature_range", [0, 1]), shape=(2,))
            result = np.clip(result, *bounds)
        return result
    if kind == "PCA":
        mean = finite_array(model["mean"], shape=(d,))
        components = finite_array(model["components"], ndim=2)
        if components.shape[1] != d or not 0 < len(components) <= d:
            raise EvidenceError("PCA component dimensions differ from raw features")
        result = (x - mean) @ components.T
        if type(model.get("whiten", False)) is not bool:
            raise EvidenceError("PCA whiten must be boolean")
        if model.get("whiten", False):
            variance = finite_array(
                model["explained_variance"], shape=(len(components),)
            )
            if np.any(variance <= 0):
                raise EvidenceError("Whitened components need positive variance")
            result /= np.sqrt(variance)
        return result
    if kind not in {"identity", "StandardScaler", "standard_scaler"}:
        raise UnsupportedEvidence(f"Preprocessing {kind!r} needs independent review")
    mean = finite_array(model.get("mean", np.zeros(d)), shape=(d,))
    scale = finite_array(model.get("scale", np.ones(d)), shape=(d,))
    if np.any(scale <= 0):
        raise EvidenceError("Feature scales must be positive")
    return (x - mean) / scale


def predict(features, model: dict) -> np.ndarray:
    """Apply saved preprocessing and Ridge coefficients; never load model objects."""
    z = transformed_features(features, model)
    coef = finite_array(model["coef"], shape=(z.shape[1],))
    intercept = finite_array(model.get("intercept", 0), shape=()).item()
    return finite_array(z @ coef + intercept, ndim=1)


def preprocessing_matches(features, model: dict, *, rtol=1e-5, atol=1e-8) -> bool:
    """Verify fitted preprocessing from only the provided training partition."""
    x = finite_array(features, ndim=2)
    if not len(x):
        raise EvidenceError("Preprocessing needs nonempty training rows")
    kind = model.get("preprocessing", "StandardScaler")
    d = x.shape[1]
    if kind == "identity":
        return bool(
            np.allclose(model.get("mean", np.zeros(d)), np.zeros(d))
            and np.allclose(model.get("scale", np.ones(d)), np.ones(d))
        )
    if kind in {"StandardScaler", "standard_scaler"}:
        if "mean" not in model or "scale" not in model:
            raise EvidenceError("StandardScaler requires fitted mean and scale")
        return scaler_matches(x, model, rtol=rtol, atol=atol)
    if kind == "MinMaxScaler":
        bounds = finite_array(model.get("feature_range", [0, 1]), shape=(2,))
        if bounds[0] >= bounds[1]:
            raise EvidenceError("MinMaxScaler feature_range must increase")
        data_min = finite_array(model["data_min"], shape=(d,))
        data_max = finite_array(model["data_max"], shape=(d,))
        width = x.max(axis=0) - x.min(axis=0)
        width[width < 10 * np.finfo(float).eps] = 1
        expected_scale = (bounds[1] - bounds[0]) / width
        pairs = (
            (data_min, x.min(axis=0)),
            (data_max, x.max(axis=0)),
            (model["scale"], expected_scale),
            (model["offset"], bounds[0] - x.min(axis=0) * expected_scale),
        )
        transformed_features(x, model)
        return all(np.allclose(a, b, rtol=rtol, atol=atol) for a, b in pairs)
    if kind == "PCA":
        if len(x) < 2:
            raise EvidenceError("PCA variance requires at least two fitting rows")
        transformed_features(x, model)
        components = np.asarray(model["components"])
        variance = finite_array(model["explained_variance"], shape=(len(components),))
        if np.any(variance < 0):
            raise EvidenceError("PCA explained variance must be nonnegative")
        centered = x - x.mean(axis=0)
        covariance = centered.T @ centered / (len(x) - 1)
        # Eigenvector and leading-eigenvalue certificates allow sign changes and
        # arbitrary bases of degenerate eigenspaces without refitting a pipeline.
        if not np.allclose(model["mean"], x.mean(axis=0), rtol=rtol, atol=atol):
            return False
        if not np.allclose(
            components @ components.T, np.eye(len(components)), rtol=rtol, atol=atol
        ):
            return False
        exact = bool(
            np.allclose(
                components @ covariance,
                variance[:, None] * components,
                rtol=rtol,
                atol=atol,
            )
            and np.allclose(
                variance,
                np.linalg.eigvalsh(covariance)[::-1][: len(components)],
                rtol=rtol,
                atol=atol,
            )
        )
        solver = model.get("svd_solver", "full")
        if not isinstance(solver, str) or not solver.strip():
            raise EvidenceError("PCA svd_solver must identify the recorded method")
        if not exact and solver.strip().lower() not in {"full", "covariance_eigh"}:
            raise UnsupportedEvidence(
                "The declared PCA solver does not satisfy the exact eigenspace certificate; its approximation and training-only fit require independent review"
            )
        return exact
    raise UnsupportedEvidence(f"Preprocessing {kind!r} needs independent review")


def partition(
    train, test, count: int, *, exhaustive=False
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for values in (train, test):
        raw = finite_array(values, ndim=1)
        indices = raw.astype(int)
        if (
            not np.array_equal(raw, indices)
            or np.any(indices < 0)
            or np.any(indices >= count)
            or len(np.unique(indices)) != len(indices)
        ):
            raise EvidenceError(
                "Partition indices must be distinct integers within the dataset"
            )
        arrays.append(indices)
    train, test = arrays
    if np.intersect1d(train, test).size:
        raise EvidenceError("Training and evaluation rows overlap")
    if exhaustive and len(train) + len(test) != count:
        raise EvidenceError("Partition does not cover the complete dataset")
    return train, test


def scaler_matches(features, model: dict, *, rtol=1e-5, atol=1e-8) -> bool:
    """Check a StandardScaler export against only the supplied training rows."""
    x = finite_array(features, ndim=2)
    d = x.shape[1]
    if any(type(model.get(key, True)) is not bool for key in ("with_mean", "with_std")):
        raise EvidenceError("Standardization flags must be booleans")
    mean = finite_array(model.get("mean", np.zeros(d)), shape=(d,))
    scale = finite_array(model.get("scale", np.ones(d)), shape=(d,))
    expected_mean = x.mean(axis=0) if model.get("with_mean", True) else np.zeros(d)
    expected_scale = x.std(axis=0) if model.get("with_std", True) else np.ones(d)
    expected_scale[expected_scale < np.finfo(float).eps] = 1
    return bool(
        np.allclose(mean, expected_mean, rtol=rtol, atol=atol)
        and np.allclose(scale, expected_scale, rtol=rtol, atol=atol)
    )


def ridge_stationarity(features, labels, model: dict, *, rtol=1e-4, atol=1e-7) -> bool:
    """Check Ridge normal equations or nonnegative-coefficient KKT conditions.

    This does not fit a model. It checks whether the supplied coefficients are
    consistent with minimizing the stated squared-loss/Ridge objective.
    """
    x = finite_array(features, ndim=2)
    y = finite_array(labels, shape=(len(x),))
    prediction = predict(x, model)
    alpha = float(model["alpha"])
    if not np.isfinite(alpha) or alpha < 0:
        raise EvidenceError("Ridge regularization must be finite and nonnegative")
    fit_intercept = model.get("fit_intercept", True)
    if type(fit_intercept) is not bool:
        raise EvidenceError("fit_intercept must be a boolean")
    if not fit_intercept and abs(float(model.get("intercept", 0))) > atol:
        return False
    z = transformed_features(x, model)
    residual = prediction - y
    gradient = z.T @ residual + alpha * np.asarray(model["coef"])
    scale = max(1.0, np.linalg.norm(z.T @ y))
    positive = model.get("positive", False)
    if type(positive) is not bool:
        raise EvidenceError("Ridge positive must be boolean")
    tolerance = atol + rtol * scale
    if positive:
        coef = np.asarray(model["coef"])
        active = coef > atol
        slope_ok = bool(
            np.all(coef >= -atol)
            and np.linalg.norm(gradient[active]) <= tolerance
            and np.all(gradient[~active] >= -tolerance)
        )
    else:
        slope_ok = np.linalg.norm(gradient) <= tolerance
    intercept_ok = not fit_intercept or abs(residual.sum()) <= atol + rtol * max(
        1.0, np.linalg.norm(y)
    )
    return bool(slope_ok and intercept_ok)
