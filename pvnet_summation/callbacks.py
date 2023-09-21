# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stochastic weight averaging callback

Modified from:
    lightning.pytorch.callbacks.StochasticWeightAveraging
"""

from typing import Any, Callable, List, Optional, Union, cast

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities.types import LRScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging as _StochasticWeightAveraging
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch import Tensor
from torch.optim.swa_utils import SWALR

_AVG_FN = Callable[[Tensor, Tensor, Tensor], Tensor]
_DEFAULT_DEVICE = torch.device("cpu")


class StochasticWeightAveraging(_StochasticWeightAveraging):
    """Stochastic weight averaging callback

    Modified from:
        lightning.pytorch.callbacks.StochasticWeightAveraging
    """

    def __init__(
        self,
        swa_lrs: Union[float, List[float]],
        swa_epoch_start: Union[int, float] = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = _DEFAULT_DEVICE,
    ):
        r"""Implements the Stochastic Weight Averaging (SWA) Callback to average a model.

        Stochastic Weight Averaging was proposed in ``Averaging Weights Leads to
        Wider Optima and Better Generalization`` by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        This documentation is highly inspired by PyTorch's work on SWA.
        The callback arguments follow the scheme defined in PyTorch's ``swa_utils`` package.

        For a SWA explanation, please take a look
        `here <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`_.

        .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

        .. warning:: ``StochasticWeightAveraging`` is currently not supported for multiple
            optimizers/schedulers.

        .. warning:: ``StochasticWeightAveraging`` is currently only supported on every epoch.

        Arguments:
            swa_lrs: The SWA learning rate to use:

                - ``float``. Use this value for all parameter groups of the optimizer.
                - ``List[float]``. A list values for each parameter group of the optimizer.

            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch

            annealing_epochs: number of epochs in the annealing phase (default: 10)

            annealing_strategy: Specifies the annealing strategy (default: "cos"):

                - ``"cos"``. For cosine annealing.
                - ``"linear"`` For linear annealing

            avg_fn: the averaging function used to update the parameters;
                the function must take in the current value of the
                :class:`AveragedModel` parameter, the current value of :attr:`model`
                parameter and the number of models already averaged; if None,
                equally weighted average is used (default: ``None``)

            device: if provided, the averaged model will be stored on the ``device``.
                When None is provided, it will infer the `device` from ``pl_module``.
                (default: ``"cpu"``)

        """
        # Add this so we can use iterative datapipe
        self._train_batches = 0

        super()._init_(
            swa_lrs,
            swa_epoch_start,
            annealing_epochs,
            annealing_strategy,
            avg_fn,
            device,
        )

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Run at start of each train epoch"""
        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            self._initialized = True

            # move average model to request device.
            assert self._average_model is not None
            self._average_model = self._average_model.to(self._device or pl_module.device)

            optimizer = trainer.optimizers[0]
            if isinstance(self._swa_lrs, float):
                self._swa_lrs = [self._swa_lrs] * len(optimizer.param_groups)

            for lr, group in zip(self._swa_lrs, optimizer.param_groups, strict=True):
                group["initial_lr"] = lr

            assert trainer.max_epochs is not None
            self._swa_scheduler = cast(
                LRScheduler,
                SWALR(
                    optimizer,
                    swa_lr=self._swa_lrs,  # type: ignore[arg-type]
                    anneal_epochs=self._annealing_epochs,
                    anneal_strategy=self._annealing_strategy,
                    last_epoch=trainer.max_epochs if self._annealing_strategy == "cos" else -1,
                ),
            )
            if self._scheduler_state is not None:
                # Restore scheduler state from checkpoint
                self._swa_scheduler.load_state_dict(self._scheduler_state)
            elif trainer.current_epoch != self.swa_start:
                # Log a warning if we're initializing after start without any checkpoint data,
                # as behaviour will be different compared to having checkpoint data.
                rank_zero_warn(
                    "SWA is initializing after swa_start without any checkpoint data. "
                    "This may be caused by loading a checkpoint from an older version of PyTorch"
                    " Lightning."
                )

            # We assert that there is only one optimizer on fit start
            default_scheduler_cfg = LRSchedulerConfig(self._swa_scheduler)
            assert default_scheduler_cfg.interval == "epoch"
            assert default_scheduler_cfg.frequency == 1

            if trainer.lr_scheduler_configs:
                scheduler_cfg = trainer.lr_scheduler_configs[0]
                if scheduler_cfg.interval != "epoch" or scheduler_cfg.frequency != 1:
                    rank_zero_warn(
                        f"SWA is currently only supported every epoch. Found {scheduler_cfg}"
                    )
                rank_zero_info(
                    f"Swapping scheduler `{scheduler_cfg.scheduler.__class__.__name__}`"
                    f" for `{self._swa_scheduler.__class__.__name__}`"
                )
                trainer.lr_scheduler_configs[0] = default_scheduler_cfg
            else:
                trainer.lr_scheduler_configs.append(default_scheduler_cfg)

            if self.n_averaged is None:
                self.n_averaged = torch.tensor(
                    self._init_n_averaged, dtype=torch.long, device=pl_module.device
                )

        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (
            trainer.current_epoch > self._latest_update_epoch
        ):
            assert self.n_averaged is not None
            assert self._average_model is not None
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self._avg_fn)
            self._latest_update_epoch = trainer.current_epoch

        # Note: No > here in case the callback is saved with the model and training continues
        if trainer.current_epoch == self.swa_end + 1:
            # Transfer weights from average model to pl_module
            assert self._average_model is not None
            self.transfer_weights(self._average_model, pl_module)

            # Reset BatchNorm for update
            self.reset_batch_norm_and_save_state(pl_module)

            # There is no need to perform either backward or optimizer.step as we are
            # performing only one pass over the train data-loader to compute activation statistics
            # Therefore, we will virtually increase the number of training batches by 1 and
            # skip backward.
            trainer.fit_loop.max_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
            trainer.accumulate_grad_batches = self._train_batches

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args: Any) -> None:
        """Run at end of each train epoch"""
        if trainer.current_epoch == 0:
            self._train_batches = trainer.global_step
        trainer.fit_loop._skip_backward = False
