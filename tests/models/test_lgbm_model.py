import numpy as np
import pytest

from pvnet_summation.models.lgbm_model import LightGBMModel


@pytest.fixture()
def lgbm_model(model_kwargs):
    return LightGBMModel(
        output_quantiles=None,
        predict_difference_from_sum=True,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
    )


@pytest.fixture()
def lgbm_quantile_model(model_kwargs):
    return LightGBMModel(
        output_quantiles=[0.1, 0.5, 0.9],
        predict_difference_from_sum=True,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
    )


def test_model_forward_before_fit(lgbm_model, batch):
    """Should return zeros without crashing before fitting."""
    y = lgbm_model.forward(batch)
    assert tuple(y.shape) == (2, 16), y.shape


def test_quantile_model_forward_before_fit(lgbm_quantile_model, batch):
    """Should return zeros without crashing before fitting."""
    y = lgbm_quantile_model.forward(batch)
    assert tuple(y.shape) == (2, 16, 3), y.shape


def test_batch_to_numpy(lgbm_model, batch):
    """Check X and y shapes from _batch_to_numpy."""
    X, y = lgbm_model._batch_to_numpy(batch)
    assert X.shape[0] == 2 * 16
    assert y.shape == (2 * 16,)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_model_fit_and_forward(lgbm_model, batch):
    """Fit on dummy data then check forward output shape."""
    X, y = lgbm_model._batch_to_numpy(batch)
    lgbm_model.fit(X, y)
    assert lgbm_model._is_fitted

    y_hat = lgbm_model.forward(batch)
    assert tuple(y_hat.shape) == (2, 16), y_hat.shape


def test_quantile_model_fit_and_forward(lgbm_quantile_model, batch):
    """Fit quantile model and check output shape."""
    X, y = lgbm_quantile_model._batch_to_numpy(batch)
    lgbm_quantile_model.fit(X, y)
    assert lgbm_quantile_model._is_fitted

    y_hat = lgbm_quantile_model.forward(batch)
    assert tuple(y_hat.shape) == (2, 16, 3), y_hat.shape


def test_quantile_model_num_boosters(lgbm_quantile_model, batch):
    """One booster per quantile."""
    X, y = lgbm_quantile_model._batch_to_numpy(batch)
    lgbm_quantile_model.fit(X, y)
    assert len(lgbm_quantile_model.boosters) == 3


def test_model_num_boosters(lgbm_model, batch):
    """Single booster for mean regression."""
    X, y = lgbm_model._batch_to_numpy(batch)
    lgbm_model.fit(X, y)
    assert len(lgbm_model.boosters) == 1
