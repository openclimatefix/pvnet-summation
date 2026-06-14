import pytest

from pvnet_summation.models.attention_model import LocationAttentionModel


@pytest.fixture()
def attention_model(model_kwargs):
    return LocationAttentionModel(
        output_quantiles=None,
        predict_difference_from_sum=True,
        use_horizon_encoding=True,
        num_heads=4,
        embed_dim=64,
        num_attention_layers=2,
        num_horizon_attention_layers=2,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
        output_network=model_kwargs["output_network"],
    )


@pytest.fixture()
def attention_quantile_model(model_kwargs):
    return LocationAttentionModel(
        output_quantiles=[0.1, 0.5, 0.9],
        predict_difference_from_sum=True,
        use_horizon_encoding=True,
        num_heads=4,
        embed_dim=64,
        num_attention_layers=2,
        num_horizon_attention_layers=2,
        **{k: v for k, v in model_kwargs.items() if k not in ("output_network", "predict_difference_from_sum")},
        output_network=model_kwargs["output_network"],
    )


def test_model_forward(attention_model, batch):
    y = attention_model.forward(batch)
    assert tuple(y.shape) == (2, 16), y.shape


def test_model_backward(attention_model, batch):
    y = attention_model(batch)
    y.sum().backward()


def test_quantile_model_forward(attention_quantile_model, batch):
    y = attention_quantile_model.forward(batch)
    assert tuple(y.shape) == (2, 16, 3), y.shape


def test_quantile_model_backward(attention_quantile_model, batch):
    y = attention_quantile_model(batch)
    y.sum().backward()
