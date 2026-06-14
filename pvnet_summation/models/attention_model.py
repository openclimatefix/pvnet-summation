"""Attention over both locations and horizons"""

import torch
import torch.nn.functional as F
from torch import nn

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.models.base_model import BaseModel


class AttentionBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialise transformer block with given embedding dimension, heads and dropout."""
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run transformer block forward."""
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ff(x)))


class LocationAttentionModel(BaseModel):
    """Attention over locations and horizons.

    At each forecast horizon, multi-head attention is applied over locations.
    A second attention block then attends across horizon steps.
    """

    def __init__(
        self,
        output_quantiles: list[float] | None,
        num_input_locations: int,
        input_quantiles: list[float] | None,
        history_minutes: int,
        forecast_minutes: int,
        interval_minutes: int,
        output_network: torch.nn.Module,
        num_heads: int = 4,
        embed_dim: int = 64,
        num_attention_layers: int = 2,
        num_horizon_attention_layers: int = 2,
        predict_difference_from_sum: bool = False,
        use_horizon_encoding: bool = False,
    ):
        """Attention over locations and horizons.

        Args:
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            num_input_locations: The number of input locations (e.g. number of GSPs)
            input_quantiles: A list of float (0.0, 1.0) quantiles which PVNet predicts for. If set
                to None we assume PVNet predicts a single value
            history_minutes: Length of the GSP history period in minutes
            forecast_minutes: Length of the GSP forecast period in minutes
            interval_minutes: The interval in minutes between each timestep in the data
            output_network: A partially instantiated pytorch Module class used to predict the
                outturn at each horizon from the attended features.
            num_heads: Number of attention heads over the location dimension
            embed_dim: Embedding dimension for attention. Must be divisible by num_heads.
            num_attention_layers: Number of stacked location attention layers
            num_horizon_attention_layers: Number of stacked cross-horizon attention layers
            predict_difference_from_sum: Whether to predict the difference from the sum of
                locations, else the total is predicted directly
            use_horizon_encoding: Whether to use the forecast horizon as an input feature
        """
        super().__init__(
            output_quantiles,
            num_input_locations,
            input_quantiles,
            history_minutes,
            forecast_minutes,
            interval_minutes,
        )

        self.predict_difference_from_sum = predict_difference_from_sum
        self.use_horizon_encoding = use_horizon_encoding

        loc_features = 1 if input_quantiles is None else len(input_quantiles)

        self.loc_embedding = nn.Embedding(num_input_locations, embed_dim)
        self.loc_proj = nn.Sequential(nn.Linear(loc_features, embed_dim), nn.Dropout(0.1))
        self.loc_attention = nn.Sequential(
            *[AttentionBlock(embed_dim, num_heads) for _ in range(num_attention_layers)]
        )
        self.loc_aggregate = nn.Sequential(
            nn.Linear(embed_dim * num_input_locations, embed_dim), nn.Dropout(0.1)
        )

        self.horizon_embedding = nn.Embedding(self.forecast_len, embed_dim)
        self.horizon_attention = nn.Sequential(
            *[AttentionBlock(embed_dim, num_heads) for _ in range(num_horizon_attention_layers)]
        )

        in_features = embed_dim + (1 if use_horizon_encoding else 0)
        out_features = len(self.output_quantiles) if self.use_quantile_regression else 1

        self.model = output_network(in_features=in_features, out_features=out_features)

        if predict_difference_from_sum:
            self.model = nn.Sequential(
                self.model,
                nn.Linear(out_features, out_features),
            )

    def forward(self, x: SumTensorBatch) -> torch.Tensor:
        """Run model forward"""

        pv = x["pvnet_outputs"]
        batch_size = pv.shape[0]
        pv = torch.swapaxes(pv, 1, 2)

        if pv.dim() == 3:
            pv = pv.unsqueeze(-1)

        pv = pv.flatten(0, 1)
        pv = self.loc_proj(pv)

        loc_ids = torch.arange(self.num_input_locations, device=pv.device)
        pv = pv + self.loc_embedding(loc_ids).unsqueeze(0)
        pv = self.loc_attention(pv)

        pv = self.loc_aggregate(pv.flatten(1))
        pv = pv.view(batch_size, self.forecast_len, -1)

        horizon_ids = torch.arange(self.forecast_len, device=pv.device)
        pv = pv + self.horizon_embedding(horizon_ids).unsqueeze(0)
        pv = self.horizon_attention(pv)

        if self.use_horizon_encoding:
            horizon_enc = torch.linspace(
                start=0,
                end=1,
                steps=self.forecast_len,
                device=pv.device,
                dtype=pv.dtype,
            )
            horizon_enc = horizon_enc.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
            pv = torch.cat([pv, horizon_enc], dim=2)

        pv = pv.flatten(0, 1)
        out = self.model(pv)
        out = out.view(batch_size, *self.output_shape)

        if self.predict_difference_from_sum:
            loc_sum = self.sum_of_locations(x)
            if self.use_quantile_regression:
                loc_sum = loc_sum.unsqueeze(-1)
            out = loc_sum + out

        return F.leaky_relu(out, negative_slope=0.01)