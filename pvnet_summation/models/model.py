"""Simple model which only uses outputs of PVNet for all GSPs"""

from typing import Optional

import numpy as np
import torch

import pvnet
from pvnet_summation.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
from pvnet.models.multimodal.linear_networks.networks import DefaultFCNet
from pvnet.models.multimodal.linear_networks.basic_blocks import AbstractLinearNetwork




class Model(BaseModel):
    """Neural network which combines GSP predictions from PVNet

    """

    name = "pvnet_summation_model"

    def __init__(
        self,
        model_name: str,
        forecast_minutes: int,
        model_version: Optional[str],
        output_quantiles: Optional[list[float]] = None,
        output_network: AbstractLinearNetwork = DefaultFCNet,
        output_network_kwargs: dict = dict(),
        optimizer: AbstractOptimizer  = pvnet.optimizers.Adam(),

    ):
        """Neural network which combines GSP predictions from PVNet

        Args:
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            output_network: Pytorch Module class used to combine the 1D features to produce the
                forecast.
            output_network_kwargs: Dictionary of optional kwargs for the `output_network` module.
            model_name: Model path either locally or on huggingface.
            model_version: Model version if using huggingface. Set to None if using local.
            forecast_minutes (int): Length of the GSP forecast period in minutes
            optimizer (AbstractOptimizer): Optimizer
        """

        super().__init__(
            forecast_minutes, 
            model_name, 
            model_version, 
            optimizer, 
            output_quantiles
        )

        in_features = np.product(self.pvnet_output_shape)
        
        self.model = output_network(
            in_features=in_features,
            out_features=self.num_output_features,
            **output_network_kwargs,
        )

        self.save_hyperparameters()


    def forward(self, x):
        """Run model forward"""
        
        if "pvnet_outputs" in x:
            pvnet_out = x["pvnet_outputs"]
        else:
            pvnet_out = self.predict_pvnet_batch(x['pvnet_inputs'])
        
        pvnet_out = torch.flatten(pvnet_out, start_dim=1)
        out = self.model(pvnet_out)
        
        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len_30, len(self.output_quantiles))
        
        return out

