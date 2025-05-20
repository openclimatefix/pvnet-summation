"""Pytorch lightning datamodules for loading pre-saved samples and predictions.

The pre-saced samplea can be prepared using the save_concurrent_samples.py script from the PVNet
library: https://github.com/openclimatefix/PVNet
"""

from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.load.gsp import open_gsp


# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_summation_samples(samples: list[dict]) -> dict:
    batch_dict = {}
    batch_dict["pvnet_inputs"] = [s["pvnet_inputs"] for s in samples]
    for key in ["effective_capacity", "national_targets", "times"]:
        batch_dict[key] = torch.stack([s[key] for s in samples])
    return batch_dict


def get_national_outturns(gsp_data, times) -> torch.Tensor:
    return torch.as_tensor(
        gsp_data.sel(time_utc=times.cpu().numpy().astype("datetime64[ns]")).values
    )


def get_sample_valid_times(sample: dict) -> torch.Tensor:
    id0 = int(sample["gsp_t0_idx"])
    return sample["gsp_time_utc"][0, id0 + 1 :]


def get_sample_capacities(sample: dict) -> torch.Tensor:
    return sample["gsp_effective_capacity_mwp"].float().unsqueeze(-1)


class SavedSampleDataset(Dataset):
    def __init__(self, sample_dir: str, gsp_zarr_path: str, gsp_boundaries_version="20220314"):
        self.sample_filepaths = glob(f"{sample_dir}/*.pt")

        # Load and nornmalise the national GSP data to use as target values
        gsp_data = (
            open_gsp(zarr_path=gsp_zarr_path, boundaries_version=gsp_boundaries_version)
            .sel(gsp_id=0)
            .compute()
        )
        gsp_data = gsp_data / gsp_data.effective_capacity_mwp

        self.gsp_data = gsp_data

    def __len__(self) -> int:
        return len(self.sample_filepaths)

    def __getitem__(self, idx) -> dict:
        sample = torch.load(self.sample_filepaths[idx], weights_only=False)

        sample_valid_times = get_sample_valid_times(sample)

        national_outturns = get_national_outturns(self.gsp_data, sample_valid_times)

        national_capacity = get_national_outturns(
            self.gsp_data.effective_capacity_mwp, sample_valid_times
        )[0]

        gsp_capacities = get_sample_capacities(sample)

        gsp_relative_capacities = gsp_capacities / national_capacity

        return dict(
            pvnet_inputs=sample,
            effective_capacity=gsp_relative_capacities,
            national_targets=national_outturns,
            times=sample_valid_times,
        )


class SavedSampleDataModule(LightningDataModule):
    """Datamodule for training pvnet_summation."""

    def __init__(
        self,
        sample_dir: str,
        gsp_zarr_path: str,
        gsp_boundaries_version: str = "20220314",
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
    ):
        """Datamodule for training pvnet_summation.

        Args:
            sample_dir: Path to the directory of pre-saved samples.
            gsp_zarr_path: Path to zarr file containing GSP ID 0 outputs
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.gsp_zarr_path = gsp_zarr_path
        self.gsp_boundaries_version = gsp_boundaries_version
        self.sample_dir = sample_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    @property
    def _dataloader_kwargs(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size is None else collate_summation_samples,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct train dataloader"""
        dataset = SavedSampleDataset(
            f"{self.sample_dir}/train",
            self.gsp_zarr_path,
            self.gsp_boundaries_version,
        )
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        dataset = SavedSampleDataset(
            f"{self.sample_dir}/val",
            self.gsp_zarr_path,
            self.gsp_boundaries_version,
        )
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)


class SavedPredictionDataset(Dataset):
    def __init__(self, sample_dir: str):
        self.sample_filepaths = sorted(glob(f"{sample_dir}/*.pt"))

    def __len__(self) -> int:
        return len(self.sample_filepaths)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.sample_filepaths[idx], weights_only=False)


class SavedPredictionDataModule(LightningDataModule):
    """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation."""

    def __init__(
        self,
        sample_dir: str,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
    ):
        """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation.

        Args:
            sample_dir: Path to the directory of pre-saved batches.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.sample_dir = sample_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    @property
    def _dataloader_kwargs(self) -> dict:
        return dict(
            batch_size=self.batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size is None else default_collate,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Construct train dataloader"""
        dataset = SavedPredictionDataset(f"{self.sample_dir}/train")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Construct val dataloader"""
        dataset = SavedPredictionDataset(f"{self.sample_dir}/val")
        return DataLoader(dataset, shuffle=shuffle, **self._dataloader_kwargs)
