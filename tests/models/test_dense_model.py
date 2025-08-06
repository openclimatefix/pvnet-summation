import pytest
from pvnet_summation.models.dense_model import DenseModel


@pytest.fixture()
def quantile_model(model_kwargs):
    return DenseModel(output_quantiles=[0.1, 0.5, 0.9], **model_kwargs)


def test_model_forward(model, batch):
    y = model.forward(batch)

    # batch size=2, forecast_len=16
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(model, batch):

    y = model(batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(quantile_model, batch):
    y_quantiles = quantile_model(batch)

    # batch size=2, forecast_len=16, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(quantile_model, batch):

    y_quantiles = quantile_model(batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
