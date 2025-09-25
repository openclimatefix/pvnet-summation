"""Simple model which only uses outputs of PVNet for all GSPs"""

import torch
import torch.nn.functional as F
from torch import nn

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.models.base_model import BaseModel


class NewModel(BaseModel):
    """Neural network architecture based on dense layers applied independently at each horizon

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
        predict_difference_from_sum: bool = False,
        add_horizon_encoding: bool = False,
        add_solar_position: bool = False,
        force_non_crossing: bool = False,
        beta: float = 3,
    ):
        """Neural network architecture based on naive dense layers

        """

        super().__init__(
            output_quantiles, 
            num_input_locations,
            input_quantiles,
            history_minutes,
            forecast_minutes,
            interval_minutes,
        )

        self.add_horizon_encoding = add_horizon_encoding
        self.predict_difference_from_sum = predict_difference_from_sum
        self.force_non_crossing = force_non_crossing
        self.beta = beta
        self.add_solar_position = add_solar_position


        if force_non_crossing:
            assert self.use_quantile_regression

        if input_quantiles is None:
            in_features = self.num_input_locations
        else:
            in_features = self.num_input_locations * len(input_quantiles)

        if add_horizon_encoding:
            in_features += 1

        if add_solar_position:
            in_features += 2

        if self.use_quantile_regression:
            self._out_features = len(self.output_quantiles)
        else:
            self._out_features = 1


        self.model = output_network(
            in_features=in_features,
            out_features=self._out_features,
        )

        # Add linear layer if predicting difference from sum
        # This allows difference to be positive or negative
        if predict_difference_from_sum:
            self.model = nn.Sequential(
                self.model, 
                nn.Linear(self._out_features, self._out_features),
            )

    def forward(self, x: SumTensorBatch) -> torch.Tensor:
        """Run model forward"""

        b, l, h = x["pvnet_outputs"].size()[:3]
        # x["pvnet_outputs"] has shape [batch, locs, horizon, (quantile)]
        x_in = torch.swapaxes(x["pvnet_outputs"], 1, 2) # -> [batch, horizon, locs, (quantile)]
        x_in = torch.flatten(x_in, start_dim=2) # -> [batch, horizon, locs*(quantile)]

        if self.add_horizon_encoding:
            horizon_encoding = torch.arange(
                start=0, 
                end=self.forecast_len,
                device=x_in.device, 
                dtype=x_in.dtype,
            ) / self.forecast_len
            horizon_encoding = horizon_encoding.tile((b,1)).unsqueeze(-1)
            x_in = torch.cat([x_in, horizon_encoding], dim=2)

        if self.add_solar_position:
            azimuth = x["azimuth"]
            elevation = x["elevation"]
            x_in = torch.cat([x_in, azimuth.unsqueeze(-1), elevation.unsqueeze(-1)], dim=2)

        x_in = torch.flatten(x_in, start_dim=0, end_dim=1) # -> [batch*horizon, locs*(quantile)]

        out = self.model(x_in)
        out = out.view(b, h, self._out_features)

        if not self.use_quantile_regression:
            # Shape: [batch_size, horizon, {quantiles, 1}]
            out = out.squeeze(axis=-1)

        if self.predict_difference_from_sum:
            loc_sum = self.sum_of_locations(x)

            if self.force_non_crossing:
                loc_sum = loc_sum.unsqueeze(-1)
                idx = self.input_quantiles.index(0.5)

                y_mid = loc_sum + out[..., idx:idx+1]
                if self.beta is None:
                    dy_below = F.relu(out[..., :idx])
                    dy_above = F.relu(out[..., idx+1:])
                else:
                    dy_below = F.softplus(out[..., :idx], beta=self.beta)
                    dy_above = F.softplus(out[..., idx+1:], beta=self.beta)

                y_below = []
                y = y_mid
                for i in range(dy_below.shape[-1]):
                    y = y.detach() - dy_below[..., i:i+1]
                    y_below.append(y)

                y_below = y_below[::-1]


                y_above = []
                y = y_mid
                for i in range(dy_above.shape[-1]):
                    y = y.detach() + dy_above[..., i:i+1]
                    y_above.append(y)

                out = F.leaky_relu(torch.cat(y_below + [y_mid,] + y_above, dim=-1), negative_slope=0.01)

            else:

                if self.use_quantile_regression:
                    loc_sum = loc_sum.unsqueeze(-1)

                out = F.leaky_relu(loc_sum + out, negative_slope=0.01)

        else:
            if self.force_non_crossing:
                out = F.softplus(out, beta=self.beta)
                out = torch.cumsum(out, dim=-1)

        return out
