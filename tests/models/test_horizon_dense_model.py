import pytest
from pvnet_summation.models.horizon_dense_model import HorizonDenseModel


@pytest.fixture()
def horizon_dense_quantile_model(model_kwargs):
    return HorizonDenseModel(
        output_quantiles=[0.1, 0.5, 0.9], 
        force_non_crossing=True,
        use_horizon_encoding=True, 
        **model_kwargs,
)


@pytest.fixture()
def horizon_dense_model(model_kwargs):
    return HorizonDenseModel(
        output_quantiles=None,
        use_horizon_encoding=True,  
        **model_kwargs,
    )

def test_model_forward(model, batch):
    y = model.forward(batch)

    # batch size=2, forecast_len=16
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(horizon_dense_model, batch):

    y = horizon_dense_model(batch)

    # Backwards on sum drives sum to zero
    y.sum().backward()


def test_quantile_model_forward(horizon_dense_quantile_model, batch):
    y_quantiles = horizon_dense_quantile_model(batch)

    # batch size=2, forecast_len=16, num_quantiles=3
    assert tuple(y_quantiles.shape) == (2, 16, 3), y_quantiles.shape


def test_quantile_model_backward(horizon_dense_quantile_model, batch):

    y_quantiles = horizon_dense_quantile_model(batch)

    # Backwards on sum drives sum to zero
    y_quantiles.sum().backward()
