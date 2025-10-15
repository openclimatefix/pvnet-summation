"""Pytorch lightning datamodules for loading pre-saved samples and predictions."""

import os
from glob import glob
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.load.gsp import get_gsp_boundaries, open_gsp
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch
from ocf_data_sampler.numpy_sample.sun_position import calculate_azimuth_and_elevation
from ocf_data_sampler.select.geospatial import osgb_to_lon_lat
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset
from ocf_data_sampler.utils import minutes
from torch.utils.data import DataLoader, Dataset, Subset, default_collate
from typing_extensions import override

SumNumpySample: TypeAlias = dict[str, np.ndarray | NumpyBatch]
SumTensorBatch: TypeAlias = dict[str, torch.Tensor]

def get_gb_centroid_lon_lat() -> tuple[float, float]:
    """Get the longitude and latitude of the centroid of Great Britain"""
    row = get_gsp_boundaries("20250109").loc[0]
    x_osgb = row.x_osgb.item()
    y_osgb = row.y_osgb.item()
    return osgb_to_lon_lat(x_osgb, y_osgb)

LON, LAT = get_gb_centroid_lon_lat()


def construct_sample(
    pvnet_inputs: NumpyBatch,
    valid_times: pd.DatetimeIndex,
    relative_capacities: np.ndarray,
    target: np.ndarray | None,
    last_outturn: float | None = None,
) -> SumNumpySample:
    """Construct an input sample for the summation model

    Args:
        pvnet_inputs: The PVNet batch for all GSPs
        valid_times: An array of valid-times for the forecast
        relative_capacities: Array of capacities of all GSPs normalised by the total capacity
        target: The target national outturn. This is only needed during training.
        last_outturn: The previous national outturn. This is only needed during training.
    """
    
    azimuth, elevation = calculate_azimuth_and_elevation(valid_times, LON, LAT)

    sample = {
        # NumpyBatch object with batch size = num_locations
        "pvnet_inputs": pvnet_inputs,
        # Shape: [time]
        "valid_times": valid_times.values.astype(int),
        # Shape: [num_locations]
        "relative_capacity": relative_capacities,
        # Shape: [time]
        "azimuth": azimuth.astype(np.float32) / 360,
        # Shape: [time]
        "elevation": elevation.astype(np.float32) / 180 + 0.5,
    }

    if target is not None:
        # Shape: [time]
        sample["target"] = target
    if last_outturn is not None:
        # Shape: scalar
        sample["last_outturn"] = last_outturn
    return sample


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
        self.national_gsp_data = (
            open_gsp(
                zarr_path=self.config.input_data.gsp.zarr_path, 
                boundaries_version=self.config.input_data.gsp.boundaries_version
            )
            .sel(gsp_id=0)
            .compute()
        )

    def _get_sample(self, t0: pd.Timestamp) -> SumNumpySample:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """

        # Get the PVNet input batch
        pvnet_inputs: NumpyBatch = super()._get_sample(t0)

        # Construct an array of valid times for eahc forecast horizon
        valid_times = pd.date_range(
            t0+minutes(self.config.input_data.gsp.time_resolution_minutes), 
            t0+minutes(self.config.input_data.gsp.interval_end_minutes),
            freq=minutes(self.config.input_data.gsp.time_resolution_minutes)
        )

        # Get the GSP and national capacities
        location_capacities = pvnet_inputs["gsp_effective_capacity_mwp"]
        total_capacity = self.national_gsp_data.sel(time_utc=t0).effective_capacity_mwp.item()
        
        # Calculate requited inputs for the sample
        relative_capacities = location_capacities / total_capacity
        target = self.national_gsp_data.sel(time_utc=valid_times).values / total_capacity
        last_outturn = self.national_gsp_data.sel(time_utc=t0).values / total_capacity

        return construct_sample(
            pvnet_inputs=pvnet_inputs,
            valid_times=valid_times,
            relative_capacities=relative_capacities,
            target=target,
            last_outturn=last_outturn,
        )

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
        seed: int | None = None,
        dataset_pickle_dir: str | None = None,
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
            seed: Random seed used in shuffling datasets.
            dataset_pickle_dir: Directory in which the val and train set will be presaved as
                pickle objects. Setting this speeds up instantiation of multiple workers a lot.
        """
        super().__init__()
        self.configuration = configuration
        self.train_period = train_period
        self.val_period = val_period
        self.seed = seed
        self.dataset_pickle_dir = dataset_pickle_dir

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
            multiprocessing_context="spawn" if num_workers>0 else None,
        )

    def setup(self, stage: str | None = None):
        """Called once to prepare the datasets."""

        # This logic runs only once at the start of training, therefore the val dataset is only
        # shuffled once
        if self.dataset_pickle_dir is not None:
            os.makedirs(self.dataset_pickle_dir, exist_ok=True)

            train_dataset_path = f"{self.dataset_pickle_dir}/train_dataset.pkl"
            val_dataset_path = f"{self.dataset_pickle_dir}/val_dataset.pkl"

            # For safety, these pickled datasets cannot be overwritten.
            # See: https://github.com/openclimatefix/pvnet/pull/445
            for path in [train_dataset_path, val_dataset_path]:
                if os.path.exists(path):
                    raise FileExistsError(
                        f"The pickled dataset path '{path}' already exists. Make sure that "
                        "this can be safely deleted (i.e. not currently being used by any "
                        "training run) and delete it manually. Else change the "
                        "`dataset_pickle_dir` to a different directory."
                    )

        # Prepare the train dataset
        self.train_dataset = StreamedDataset(self.configuration, *self.train_period)

        # Prepare and pre-shuffle the val dataset and set seed for reproducibility
        val_dataset = StreamedDataset(self.configuration, *self.val_period)
        shuffled_indices = np.random.default_rng(seed=self.seed).permutation(len(val_dataset))
        self.val_dataset = Subset(val_dataset, shuffled_indices)
    
        if self.dataset_pickle_dir is not None:
            self.train_dataset.presave_pickle(train_dataset_path)
            self.train_dataset.presave_pickle(val_dataset_path)

    def teardown(self, stage: str | None = None) -> None:
        """Clean up the pickled datasets"""
        if self.dataset_pickle_dir is not None:
            for filename in ["val_dataset.pkl", "train_dataset.pkl"]:
                filepath = f"{self.dataset_pickle_dir}/{filename}"
                if os.path.exists(filepath):
                    os.remove(filepath)

    def train_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct train dataloader"""
        return DataLoader(self.train_dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        return DataLoader(self.val_dataset, shuffle=shuffle, **self._dataloader_kwargs)


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
            multiprocessing_context="spawn" if num_workers>0 else None,
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Construct train dataloader"""
        dataset = PresavedDataset(f"{self.sample_dir}/train")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        dataset = PresavedDataset(f"{self.sample_dir}/val")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)
