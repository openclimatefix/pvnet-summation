"""Neural network architecture based on dense layers applied independently at each horizon"""


import torch
import torch.nn.functional as F
from torch import nn

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.models.base_model import BaseModel


class HorizonDenseModel(BaseModel):
    """Neural network architecture based on dense layers applied independently at each horizon.
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
        use_horizon_encoding: bool = False,
        use_solar_position: bool = False,
        force_non_crossing: bool = False,
        beta: float = 3,
    ):
        """Neural network architecture based on dense layers applied independently at each horizon.

        Args:
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            num_input_locations: The number of input locations (e.g. number of GSPs)
            input_quantiles: A list of float (0.0, 1.0) quantiles which PVNet predicts for. If set 
                to None we assume PVNet predicts a single value
            history_minutes (int): Length of the GSP history period in minutes
            forecast_minutes (int): Length of the GSP forecast period in minutes
            interval_minutes: The interval in minutes between each timestep in the data
            output_network: A partially instantiated pytorch Module class used top predict the 
                outturn at each horizon.
            predict_difference_from_sum: Whether to predict the difference from the sum of locations
                else the total is predicted directly
            use_horizon_encoding: Whether to use the forecast horizon as an input feature
            use_solar_position: Whether to use the solar coordinates as input features
            force_non_crossing: If predicting quantile, whether to predict the quantiles other than
                the median by predicting the distance between them and integrating.
            beta: If using force_non_crossing, the beta value to use in the softplus activation
        """

        super().__init__(
            output_quantiles, 
            num_input_locations,
            input_quantiles,
            history_minutes,
            forecast_minutes,
            interval_minutes,
        )

        if force_non_crossing:
            assert self.use_quantile_regression

        self.use_horizon_encoding = use_horizon_encoding
        self.predict_difference_from_sum = predict_difference_from_sum
        self.force_non_crossing = force_non_crossing
        self.beta = beta
        self.use_solar_position = use_solar_position

        in_features = 1 if self.input_quantiles is None else len(self.input_quantiles)
        in_features = in_features * self.num_input_locations

        if use_horizon_encoding:
            in_features += 1

        if use_solar_position:
            in_features += 2

        out_features = (len(self.output_quantiles) if self.use_quantile_regression else 1)

        model = output_network(in_features=in_features, out_features=out_features)

        # Add linear layer if predicting difference from sum
        # - This allows difference to be positive or negative
        # Also add linear layer if we are applying force_non_crossing since a softplus will be used
        if predict_difference_from_sum or force_non_crossing:
            model = nn.Sequential(
                model,
                nn.Linear(out_features, out_features),
            )
        
        self.model = model


    def forward(self, x: SumTensorBatch) -> torch.Tensor:
        """Run model forward"""

        # x["pvnet_outputs"] has shape [batch, locs, horizon, (quantile)]
        batch_size = x["pvnet_outputs"].shape[0]
        x_in = torch.swapaxes(x["pvnet_outputs"], 1, 2) # -> [batch, horizon, locs, (quantile)]
        x_in = torch.flatten(x_in, start_dim=2) # -> [batch, horizon, locs*(quantile)]

        if self.use_horizon_encoding:
            horizon_encoding = torch.linspace(
                start=0,
                end=1,
                steps=self.forecast_len,
                device=x_in.device,
                dtype=x_in.dtype,
            )
            horizon_encoding = horizon_encoding.tile((batch_size,1)).unsqueeze(-1)
            x_in = torch.cat([x_in, horizon_encoding], dim=2)

        if self.use_solar_position:
            x_in = torch.cat(
                [x_in, x["azimuth"].unsqueeze(-1), x["elevation"].unsqueeze(-1)],
                dim=2
            )

        x_in = torch.flatten(x_in, start_dim=0, end_dim=1) # -> [batch*horizon, features]

        out = self.model(x_in)
        out = out.view(batch_size, *self.output_shape) # -> [batch, horizon, (quantile)]

        if self.force_non_crossing:
            
            # Get the prediction of the median
            idx = self.output_quantiles.index(0.5)
            if self.predict_difference_from_sum:
                loc_sum = self.sum_of_locations(x).unsqueeze(-1)
                y_median = loc_sum + out[..., idx:idx+1]
            else:
                y_median = out[..., idx:idx+1]

            # These are the differences between the remaining quantiles
            dy_below = F.softplus(out[..., :idx], beta=self.beta)
            dy_above = F.softplus(out[..., idx+1:], beta=self.beta)

            # Find the absolute value of the quantile predictions from the differences
            y_below = []
            y = y_median
            for i in range(dy_below.shape[-1]):
                # We detach y to avoid the gradients caused by errors from one quantile 
                # prediction  flowing back to affect the other quantile predictions.
                # For example if the 0.9 quantile prediction was too low, we don't want the
                # gradient to pull the 0.5 quantile prediction higher to compensate.
                y = y.detach() - dy_below[..., i:i+1]
                y_below.append(y)

            y_above = []
            y = y_median
            for i in range(dy_above.shape[-1]):
                y = y.detach() + dy_above[..., i:i+1]
                y_above.append(y)

            # Compile the quantile predictions in the correct order
            out = torch.cat(y_below[::-1] + [y_median,] + y_above, dim=-1)

        else:

            if self.predict_difference_from_sum:
                loc_sum = self.sum_of_locations(x)

                if self.use_quantile_regression:
                    loc_sum = loc_sum.unsqueeze(-1)

                out = loc_sum + out

        # Use leaky relu as a soft clip to 0
        return F.leaky_relu(out, negative_slope=0.01)