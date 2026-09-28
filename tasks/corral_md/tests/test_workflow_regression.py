import numpy as np
import pytest
from corral_md.workflow_scoring.common import UnsupportedEvidence
from corral_md.workflow_scoring.regression import (
    metrics,
    partition,
    predict,
    preprocessing_matches,
    ridge_stationarity,
    scaler_matches,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


def test_metrics_are_recomputed_and_constant_targets_supported():
    result = metrics([0, 1, 2], [0, 2, 2])
    assert result == pytest.approx({"mae": 1 / 3, "rmse": np.sqrt(1 / 3), "r2": 0.5})
    assert metrics([2, 2], [2, 2])["r2"] == 1
    assert metrics([2, 2], [3, 3])["r2"] == 0


def test_coefficient_and_training_row_checks_do_not_fit_a_model():
    x = np.array([[-1.0], [0.0], [1.0]])
    y = 2 * x[:, 0] + 1
    model = {"coef": [1.0], "intercept": 1, "mean": [0], "scale": [1], "alpha": 2}
    np.testing.assert_allclose(predict(x, model), [0, 1, 2])
    assert ridge_stationarity(x, y, model)
    assert not ridge_stationarity(x, y, {**model, "fit_intercept": False})
    assert not ridge_stationarity(x, y, {**model, "coef": [4]})
    standardized = {"mean": [0], "scale": [np.sqrt(2 / 3)]}
    assert scaler_matches(x, standardized)
    assert not scaler_matches(x[:2], standardized)


def test_partition_rejects_leakage_duplicates_and_noninteger_indices():
    partition([0, 2], [1, 3], 4, exhaustive=True)
    for train, test in [([0, 1], [1, 2]), ([0, 0], [1, 2]), ([0.5], [1]), ([0], [4])]:
        with pytest.raises(ValueError):
            partition(train, test, 4)


def test_nonfinite_metrics_and_coefficients_rejected():
    with pytest.raises(ValueError):
        metrics([1, 2], [1, np.nan])
    with pytest.raises(ValueError):
        predict([[1]], {"coef": [np.inf]})


@pytest.mark.parametrize("kind", ["MinMaxScaler", "PCA"])
def test_training_only_transform_certificates_and_prediction(kind):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 5)) * np.arange(1, 6)
    y = x @ np.arange(5) + rng.normal(size=30)
    transform = (
        MinMaxScaler(feature_range=(-2, 3))
        if kind == "MinMaxScaler"
        else PCA(n_components=3, whiten=True)
    )
    z = transform.fit_transform(x)
    ridge = Ridge(alpha=0.2).fit(z, y)
    model = {
        "preprocessing": kind,
        "coef": ridge.coef_.tolist(),
        "intercept": ridge.intercept_,
        "alpha": ridge.alpha,
    }
    if kind == "MinMaxScaler":
        model.update(
            data_min=transform.data_min_.tolist(),
            data_max=transform.data_max_.tolist(),
            scale=transform.scale_.tolist(),
            offset=transform.min_.tolist(),
            feature_range=[-2, 3],
        )
    else:
        model.update(
            mean=transform.mean_.tolist(),
            components=transform.components_.tolist(),
            explained_variance=transform.explained_variance_.tolist(),
            whiten=True,
        )
    assert preprocessing_matches(x, model)
    assert not preprocessing_matches(x[:20], model)
    assert ridge_stationarity(x, y, model)
    np.testing.assert_allclose(predict(x, model), ridge.predict(z))
    if kind == "PCA":
        model["components"][0] = (-np.asarray(model["components"][0])).tolist()
        model["coef"][0] *= -1
        assert preprocessing_matches(x, model)
        np.testing.assert_allclose(predict(x, model), ridge.predict(z))


def test_positive_ridge_kkt_accepts_boundary_optimum_and_rejects_corruption():
    rng = np.random.default_rng(72)
    x = rng.normal(size=(200, 4))
    y = x @ [1, -2, 0.3, -0.8] + 3
    fitted = Ridge(alpha=0.4, positive=True, tol=1e-10, max_iter=1000).fit(x, y)
    model = {
        "preprocessing": "identity",
        "positive": True,
        "coef": fitted.coef_.tolist(),
        "intercept": fitted.intercept_,
        "alpha": 0.4,
    }
    assert ridge_stationarity(x, y, model)
    model["coef"][1] = -0.2
    assert not ridge_stationarity(x, y, model)


def test_declared_approximate_pca_is_reviewable_without_ignoring_training_leakage():
    x = np.random.default_rng(934).normal(size=(100, 60))
    fitted = PCA(
        n_components=3, svd_solver="randomized", iterated_power=0, random_state=4
    ).fit(x)
    model = {
        "preprocessing": "PCA",
        "mean": fitted.mean_.tolist(),
        "components": fitted.components_.tolist(),
        "explained_variance": fitted.explained_variance_.tolist(),
        "svd_solver": "randomized",
    }
    with pytest.raises(UnsupportedEvidence, match="approximation"):
        preprocessing_matches(x, model)
    # A claim to an exact fit is contradicted by this numerical evidence.
    assert not preprocessing_matches(x, {**model, "svd_solver": "full"})
    model["mean"][0] += 0.2
    assert not preprocessing_matches(x, model)
