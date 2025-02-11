from torch.optim import SGD
import pytest
from pvnet_summation.models.flat_model import FlatModel


@pytest.fixture()
def quantile_model(flat_model_kwargs):
    model = FlatModel(output_quantiles=[0.1, 0.5, 0.9], **flat_model_kwargs)
    return model


def test_model_forward(model, pvnet_inputs_batch):
    y = model.forward(pvnet_inputs_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=4
    assert tuple(y.shape) == (2, 4), y.shape


def test_model_backward(model, pvnet_inputs_batch):
    opt = SGD(model.parameters(), lr=0.001)

    y = model(pvnet_inputs_batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(quantile_model, pvnet_inputs_batch):
    y_quantiles = quantile_model(pvnet_inputs_batch)

    # check output is the correct shape
    # batch size=2, forecast_len=4, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 4, 3), y_quantiles.shape


def test_quantile_model_backward(quantile_model, pvnet_inputs_batch):
    opt = SGD(quantile_model.parameters(), lr=0.001)

    y_quantiles = quantile_model(pvnet_inputs_batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
