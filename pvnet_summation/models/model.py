"""Simple model which only uses outputs of PVNet for all GSPs"""

from typing import Optional

import numpy as np
import pvnet
import torch
import torch.nn.functional as F
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.multimodal.linear_networks.networks import DefaultFCNet
from pvnet.optimizers import AbstractOptimizer
from torch import nn

from pvnet_summation.models.base_model import BaseModel

_default_optimizer = pvnet.optimizers.Adam()


class Model(BaseModel):
    """Neural network which combines GSP predictions from PVNet"""

    name = "pvnet_summation_model"

    def __init__(
        self,
        model_name: str,
        model_version: Optional[str],
        output_quantiles: Optional[list[float]] = None,
        output_network: AbstractLinearNetwork = DefaultFCNet,
        output_network_kwargs: Optional[dict] = None,
        relative_scale_pvnet_outputs: bool = False,
        predict_difference_from_sum: bool = False,
        optimizer: AbstractOptimizer = _default_optimizer,
    ):
        """Neural network which combines GSP predictions from PVNet

        Args:
            model_name: Model path either locally or on huggingface.
            model_version: Model version if using huggingface. Set to None if using local.
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            output_network: Pytorch Module class used to combine the 1D features to produce the
                forecast.
            output_network_kwargs: Dictionary of optional kwargs for the `output_network` module.
            relative_scale_pvnet_outputs: If true, the PVNet predictions are scaled by a factor
                which is proportional to their capacities.
            predict_difference_from_sum: Whether to use the sum of GSPs as an estimate for the
                national sum and train the model to correct this estimate. Otherwise the model tries
                to learn the national sum from the PVNet outputs directly.
            optimizer (AbstractOptimizer): Optimizer
        """

        super().__init__(model_name, model_version, optimizer, output_quantiles)

        self.relative_scale_pvnet_outputs = relative_scale_pvnet_outputs
        self.predict_difference_from_sum = predict_difference_from_sum

        if output_network_kwargs is None:
            output_network_kwargs = dict()

        self.model = output_network(
            in_features=np.product(self.pvnet_output_shape),
            out_features=self.num_output_features,
            **output_network_kwargs,
        )

        # Add linear layer if predicting difference from sum
        # This allows difference to be positive or negative
        if predict_difference_from_sum:
            self.model = nn.Sequential(
                self.model, nn.Linear(self.num_output_features, self.num_output_features)
            )

        self.save_hyperparameters()

    def forward(self, x):
        """Run model forward"""

        if "pvnet_outputs" not in x:
            x["pvnet_outputs"] = self.predict_pvnet_batch(x["pvnet_inputs"])

        if self.relative_scale_pvnet_outputs:
            if self.pvnet_model.use_quantile_regression:
                eff_cap = x["effective_capacity"].unsqueeze(-1)
            else:
                eff_cap = x["effective_capacity"]

            # Multiply by (effective capacity / 100) since the capacities are roughly of magnitude
            # of 100 MW. We still want the inputs to the network to be order of magnitude 1.
            x_in = x["pvnet_outputs"] * (eff_cap / 100)
        else:
            x_in = x["pvnet_outputs"]

        x_in = torch.flatten(x_in, start_dim=1)
        out = self.model(x_in)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len_30, len(self.output_quantiles))

        if self.predict_difference_from_sum:
            gsp_sum = self.sum_of_gsps(x)

            if self.use_quantile_regression:
                gsp_sum = gsp_sum.unsqueeze(-1)

            out = F.leaky_relu(gsp_sum + out)

        return out
