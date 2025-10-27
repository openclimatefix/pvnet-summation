"""Pytorch lightning module for training PVNet models"""

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from torch.utils.data import default_collate

from pvnet_summation.models.base_model import BaseModel
from pvnet_summation.optimizers import AbstractOptimizer
from pvnet_summation.training.plots import plot_sample_forecasts, wandb_line_plot


class PVNetSummationLightningModule(pl.LightningModule):
    """Lightning module for training PVNet models"""

    def __init__(
        self,
        model: BaseModel,
        optimizer: AbstractOptimizer,
    ):
        """Lightning module for training PVNet models

        Args:
            model: The PVNet model
            optimizer: Optimizer
        """
        super().__init__()

        self.model = model
        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None


    def _calculate_quantile_loss(self, y_quantiles: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate quantile loss.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network
            y: Target values

        Returns:
            Quantile loss
        """
        losses = []
        for i, q in enumerate(self.model.output_quantiles):
            errors = y - y_quantiles[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses.mean()
    
    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self.model)

    def _calculate_common_losses(
        self, 
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses common to train, and val"""

        losses = {}

        if self.model.use_quantile_regression:
            losses["quantile_loss"] = self._calculate_quantile_loss(y_hat, y)
            y_hat = self.model._quantiles_to_prediction(y_hat)

        losses.update({"MSE":  F.mse_loss(y_hat, y), "MAE": F.l1_loss(y_hat, y)})

        return losses
    
    def training_step(self, batch: TensorBatch, batch_idx: int) -> torch.Tensor:
        """Run training step"""

        y_hat = self.model(batch)

        y = batch["target"]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self.log_dict(losses, on_step=True, on_epoch=True)

        if self.model.use_quantile_regression:
            opt_target = losses["quantile_loss/train"]
        else:
            opt_target = losses["MAE/train"]
        return opt_target
    
    def _calculate_val_losses(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate additional losses only run in validation"""

        losses = {}

        if self.model.use_quantile_regression:
            metric_name = "val_fraction_below/fraction_below_{:.2f}_quantile"
            # Add fraction below each quantile for calibration
            for i, quantile in enumerate(self.model.output_quantiles):
                below_quant = y <= y_hat[..., i]
                # Mask values small values, which are dominated by night
                mask = y >= 0.01
                losses[metric_name.format(quantile)] = below_quant[mask].float().mean()

        return losses

    def _calculate_step_metrics(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor, 
    ) -> tuple[np.array, np.array]:
        """Calculate the MAE and MSE at each forecast step"""

        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0).cpu().numpy()
        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0).cpu().numpy()
       
        return mae_each_step, mse_each_step
    
    def on_validation_epoch_start(self):
        """Run at start of val period"""
        # Set up stores which we will fill during validation
        self._val_horizon_maes: list[np.array] = []
        if self.current_epoch==0:
            self._val_persistence_horizon_maes: list[np.array] = []
            self._val_loc_sum_horizon_maes: list[np.array] = []
        
        # Plot some sample forecasts
        val_dataset = self.trainer.val_dataloaders.dataset

        plots_per_figure = 16
        num_figures = 2

        for plot_num in range(num_figures):
            idxs = np.arange(plots_per_figure) + plot_num * plots_per_figure
            idxs = idxs[idxs<len(val_dataset)]

            if len(idxs)==0:
                continue

            batch = default_collate([val_dataset[i] for i in idxs])
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            with torch.no_grad():
                y_hat = self.model(batch)

            y_loc_sum = self.model.sum_of_locations(batch)
            
            fig = plot_sample_forecasts(batch, y_hat, y_loc_sum, self.model.output_quantiles)

            plot_name = f"val_forecast_samples/sample_set_{plot_num}"

            self.logger.experiment.log({plot_name: wandb.Image(fig)})

            plt.close(fig)

    def validation_step(self, batch: TensorBatch, batch_idx: int) -> None:
        """Run validation step"""

        y_hat = self.model(batch)

        y = batch["target"]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/val": v for k, v in losses.items()}

        losses.update(self._calculate_val_losses(y, y_hat))

        # Calculate the horizon MAE/MSE metrics
        if self.model.use_quantile_regression:
            y_hat_mid = self.model._quantiles_to_prediction(y_hat)
        else:
            y_hat_mid = y_hat

        mae_step, mse_step = self._calculate_step_metrics(y, y_hat_mid)

        # Store to make horizon-MAE plot
        self._val_horizon_maes.append(mae_step)

        # Also add each step to logged metrics
        losses.update({f"val_step_MAE/step_{i:03}": m for i, m in enumerate(mae_step)})
        losses.update({f"val_step_MSE/step_{i:03}": m for i, m in enumerate(mse_step)})

        # Calculate the persistence and sum-of-locations losses - we only need to do this once per 
        # training run not every epoch
        if self.current_epoch==0:
            
            # Persistence
            y_persist = batch["last_outturn"].unsqueeze(1).expand(-1, self.model.forecast_len)
            mae_step_persist, mse_step_persist = self._calculate_step_metrics(y, y_persist)
            self._val_persistence_horizon_maes.append(mae_step_persist)
            losses.update(
                {
                    "MAE/val_persistence": mae_step_persist.mean(), 
                    "MSE/val_persistence": mse_step_persist.mean()
                }
            )

            # Sum of Locations
            y_loc_sum = self.model.sum_of_locations(batch)
            mae_step_loc_sum, mse_step_loc_sum = self._calculate_step_metrics(y, y_loc_sum)
            self._val_loc_sum_horizon_maes.append(mae_step_loc_sum)
            losses.update(
                {
                    "MAE/val_location_sum": mae_step_loc_sum.mean(), 
                    "MSE/val_location_sum": mse_step_loc_sum.mean()
                }
            )

        # Log the metrics
        self.log_dict(losses, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Run on epoch end"""

        val_horizon_maes = np.mean(self._val_horizon_maes, axis=0)
        self._val_horizon_maes = []

        if isinstance(self.logger, pl.loggers.WandbLogger):
            
            # Create the horizon accuracy curve
            horizon_mae_plot = wandb_line_plot(
                x=np.arange(self.model.forecast_len), 
                y=val_horizon_maes,
                xlabel="Horizon step",
                ylabel="MAE",
                title="Val horizon loss curve",
            )
            
            wandb.log({"val_horizon_mae_plot": horizon_mae_plot})

            # Create persistence and location-sum horizon accuracy curve on first epoch
            if self.current_epoch==0:
                val_persistence_horizon_maes = np.mean(self._val_persistence_horizon_maes, axis=0)
                del self._val_persistence_horizon_maes

                val_loc_sum_horizon_maes = np.mean(self._val_loc_sum_horizon_maes, axis=0)
                del self._val_loc_sum_horizon_maes

                persist_horizon_mae_plot = wandb_line_plot(
                    x=np.arange(self.model.forecast_len), 
                    y=val_persistence_horizon_maes,
                    xlabel="Horizon step",
                    ylabel="MAE",
                    title="Val persistence horizon loss curve",
                )

                loc_sum_horizon_mae_plot = wandb_line_plot(
                    x=np.arange(self.model.forecast_len), 
                    y=val_loc_sum_horizon_maes,
                    xlabel="Horizon step",
                    ylabel="MAE",
                    title="Val location-sum horizon loss curve",
                )

                wandb.log(
                    {
                        "persistence_val_horizon_mae_plot": persist_horizon_mae_plot,
                        "location_sum_val_horizon_mae_plot": loc_sum_horizon_mae_plot,
                    }
                )
