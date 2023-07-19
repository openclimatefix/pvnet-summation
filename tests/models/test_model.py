from torch.optim import SGD
import pytest


def test_model_forward(model, sample_batch):
    y = model.forward(sample_batch["pvnet_inputs"])

    # check output is the correct shape
    # batch size=2, forecast_len=16
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(model, sample_batch):
    opt = SGD(model.parameters(), lr=0.001)

    y = model(sample_batch["pvnet_inputs"])

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(quantile_model, sample_batch):
    y_quantiles = quantile_model(sample_batch["pvnet_inputs"])

    # check output is the correct shape
    # batch size=2, forecast_len=15, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(quantile_model, sample_batch):
    opt = SGD(quantile_model.parameters(), lr=0.001)

    y_quantiles = quantile_model(sample_batch["pvnet_inputs"])

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
