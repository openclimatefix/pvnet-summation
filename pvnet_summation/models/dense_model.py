"""Simple model which only uses outputs of PVNet for all GSPs"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.models.base_model import BaseModel


class DenseModel(BaseModel):
    """Neural network architecture based on naive dense layers

    This model flattens all the features into a 1D vector before feeding them into the sub network
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

        self.predict_difference_from_sum = predict_difference_from_sum

        self.model = output_network(
            in_features=np.prod(self.input_shape),
            out_features=self.num_output_features,
        )

        # Add linear layer if predicting difference from sum
        # This allows difference to be positive or negative
        if predict_difference_from_sum:
            self.model = nn.Sequential(
                self.model, 
                nn.Linear(self.num_output_features, self.num_output_features),
            )

    def forward(self, x: SumTensorBatch) -> torch.Tensor:
        """Run model forward"""

        x_in = torch.flatten(x["pvnet_outputs"], start_dim=1)
        out = self.model(x_in)

        if self.use_quantile_regression:
            # Shape: [batch_size, seq_length * num_quantiles]
            out = out.reshape(out.shape[0], self.forecast_len, len(self.output_quantiles))

        if self.predict_difference_from_sum:
            loc_sum = self.sum_of_locations(x)

            if self.use_quantile_regression:
                loc_sum = loc_sum.unsqueeze(-1)

            out = F.leaky_relu(loc_sum + out)

        return out
