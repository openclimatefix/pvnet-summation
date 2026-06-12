import numpy as np
import pytest

from pvnet_summation.models.xgb_model import XGBModel


@pytest.fixture()
def xgb_model(model_kwargs):
    return XGBModel(
        output_quantiles=None,
        predict_difference_from_sum=True,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
    )


@pytest.fixture()
def xgb_quantile_model(model_kwargs):
    return XGBModel(
        output_quantiles=[0.1, 0.5, 0.9],
        predict_difference_from_sum=True,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
    )


def test_model_forward_before_fit(xgb_model, batch):
    """Should return zeros without crashing before fitting."""
    y = xgb_model.forward(batch)
    assert tuple(y.shape) == (2, 16), y.shape


def test_quantile_model_forward_before_fit(xgb_quantile_model, batch):
    """Should return zeros without crashing before fitting."""
    y = xgb_quantile_model.forward(batch)
    assert tuple(y.shape) == (2, 16, 3), y.shape


def test_batch_to_numpy(xgb_model, batch):
    """Check X and y shapes from _batch_to_numpy."""
    X, y = xgb_model._batch_to_numpy(batch)
    # batch_size=2, forecast_len=16 -> 32 rows
    assert X.shape[0] == 2 * 16
    assert y.shape == (2 * 16,)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_model_fit_and_forward(xgb_model, batch):
    """Fit on dummy data then check forward output shape."""
    X, y = xgb_model._batch_to_numpy(batch)
    xgb_model.fit(X, y)
    assert xgb_model._is_fitted

    y_hat = xgb_model.forward(batch)
    assert tuple(y_hat.shape) == (2, 16), y_hat.shape


def test_quantile_model_fit_and_forward(xgb_quantile_model, batch):
    """Fit quantile model and check output shape."""
    X, y = xgb_quantile_model._batch_to_numpy(batch)
    xgb_quantile_model.fit(X, y)
    assert xgb_quantile_model._is_fitted

    y_hat = xgb_quantile_model.forward(batch)
    assert tuple(y_hat.shape) == (2, 16, 3), y_hat.shape


def test_quantile_model_num_boosters(xgb_quantile_model, batch):
    """One booster per quantile."""
    X, y = xgb_quantile_model._batch_to_numpy(batch)
    xgb_quantile_model.fit(X, y)
    assert len(xgb_quantile_model.boosters) == 3


def test_model_num_boosters(xgb_model, batch):
    """Single booster for mean regression."""
    X, y = xgb_model._batch_to_numpy(batch)
    xgb_model.fit(X, y)
    assert len(xgb_model.boosters) == 1
