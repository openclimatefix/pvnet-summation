"""Pytorch lightning datamodules for loading pre-saved samples and predictions."""

from glob import glob
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.load.gsp import open_gsp
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample, TensorBatch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset
from ocf_data_sampler.utils import minutes
from torch.utils.data import DataLoader, Dataset, default_collate
from typing_extensions import override

SumNumpySample: TypeAlias = dict[str, np.ndarray | NumpyBatch]
SumTensorSample: TypeAlias = dict[str, torch.Tensor | TensorBatch]
SumTensorBatch: TypeAlias = dict[str, torch.Tensor | list[TensorBatch]]


class StreamedDataset(PVNetUKConcurrentDataset):
    """A torch dataset for creating concurrent PVNet inputs and national targets."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A torch dataset for creating concurrent PVNet inputs and national targets.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        super().__init__(config_filename, start_time, end_time, gsp_ids=None)

        # Load and nornmalise the national GSP data to use as target values
        national_gsp_data = (
            open_gsp(
                zarr_path=self.config.input_data.gsp.zarr_path, 
                boundaries_version=self.config.input_data.gsp.boundaries_version
            )
            .sel(gsp_id=0)
            .compute()
        )
        self.national_gsp_data = national_gsp_data / national_gsp_data.effective_capacity_mwp


    def _get_sample(self, t0: pd.Timestamp) -> SumNumpySample:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """

        pvnet_inputs: NumpySample = super()._get_sample(t0)

        location_capacities = pvnet_inputs["gsp_effective_capacity_mwp"]

        valid_times = pd.date_range(
            t0+minutes(self.config.input_data.gsp.time_resolution_minutes), 
            t0+minutes(self.config.input_data.gsp.interval_end_minutes),
            freq=minutes(self.config.input_data.gsp.time_resolution_minutes)
        )

        total_outturns = self.national_gsp_data.sel(time_utc=valid_times).values
        total_capacity = self.national_gsp_data.sel(time_utc=t0).effective_capacity_mwp.item()

        relative_capacities = location_capacities / total_capacity

        return {
            # NumpyBatch object with batch size = num_locations
            "pvnet_inputs": pvnet_inputs,
            # Shape: [time]
            "target": total_outturns,
            # Shape: [time]
            "valid_times": valid_times.values.astype(int),
            # Shape: 
            "last_outturn": self.national_gsp_data.sel(time_utc=t0).values,
            # Shape: [num_locations]
            "relative_capacity": relative_capacities,
        }

    @override
    def __getitem__(self, idx: int) -> SumNumpySample:
        return super().__getitem__(idx)

    @override
    def get_sample(self, t0: pd.Timestamp) -> SumNumpySample:
        return super().get_sample(t0)


class StreamedDataModule(LightningDataModule):
    """Datamodule for training pvnet_summation."""

    def __init__(
        self,
        configuration: str,
        train_period: list[str | None] = [None, None],
        val_period: list[str | None] = [None, None],
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ):
        """Datamodule for creating concurrent PVNet inputs and national targets.

        Args:
            configuration: Path to ocf-data-sampler configuration file.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            persistent_workers: If True, the data loader will not shut down the worker processes 
                after a dataset has been consumed once. This allows to maintain the workers Dataset 
                instances alive.
        """
        super().__init__()
        self.configuration = configuration
        self.train_period = train_period
        self.val_period = val_period

        self._dataloader_kwargs = dict(
            batch_size=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct train dataloader"""
        dataset = StreamedDataset(self.configuration, *self.train_period)
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        dataset = StreamedDataset(self.configuration, *self.val_period)
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)


class PresavedDataset(Dataset):
    """Dataset for loading pre-saved PVNet predictions from disk"""

    def __init__(self, sample_dir: str):
        """"Dataset for loading pre-saved PVNet predictions from disk.
        
        Args:
            sample_dir: The directory containing the saved samples
        """
        self.sample_filepaths = sorted(glob(f"{sample_dir}/*.pt"))

    def __len__(self) -> int:
        return len(self.sample_filepaths)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.sample_filepaths[idx], weights_only=True)


class PresavedDataModule(LightningDataModule):
    """Datamodule for loading pre-saved PVNet predictions."""

    def __init__(
        self,
        sample_dir: str,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
    ):
        """Datamodule for loading pre-saved PVNet predictions.

        Args:
            sample_dir: Path to the directory of pre-saved samples.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            persistent_workers: If True, the data loader will not shut down the worker processes 
                after a dataset has been consumed once. This allows to maintain the workers Dataset 
                instances alive.
        """
        super().__init__()
        self.sample_dir = sample_dir

        self._dataloader_kwargs = dict(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None if batch_size is None else default_collate,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Construct train dataloader"""
        dataset = PresavedDataset(f"{self.sample_dir}/train")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        dataset = PresavedDataset(f"{self.sample_dir}/val")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)
