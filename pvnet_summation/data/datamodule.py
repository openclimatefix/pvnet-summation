"""Pytorch lightning datamodules for loading pre-saved samples and predictions."""

from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from lightning.pytorch import LightningDataModule
from ocf_data_sampler.load.gsp import open_gsp


# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_summation_samples(samples: list):
    batch_dict = {}
    batch_dict["pvnet_inputs"] = [s["pvnet_inputs"] for s in samples]
    for key in ["effective_capacity", "national_targets", "times"]:
        batch_dict[key] = torch.stack([s[key] for s in samples])
    return batch_dict


def get_national_outturns(gsp_data, times):
    return torch.as_tensor(
        gsp_data.sel(time_utc=times.cpu().numpy().astype("datetime64[ns]")).values
    )


def get_sample_valid_times(sample: dict):
    id0 = int(sample["gsp_t0_idx"])
    return sample["gsp_time_utc"][0, id0 + 1 :]


def get_sample_capacities(sample):
    return sample["gsp_effective_capacity_mwp"].float().unsqueeze(-1)


class SavedSampleDataset(Dataset):
    def __init__(self, sample_dir, gsp_zarr_path):
        self.sample_filepaths = glob(f"{sample_dir}/*.pt")

        # Load and nornmalise the national GSP data to use as target values
        gsp_data = open_gsp(zarr_path=gsp_zarr_path).sel(gsp_id=0).compute()
        gsp_data = gsp_data / gsp_data.effective_capacity_mwp

        self.gsp_data = gsp_data

    def __len__(self):
        return len(self.sample_filepaths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_filepaths[idx])

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
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
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
        self.sample_dir = sample_dir

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=collate_summation_samples,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def train_dataloader(self, shuffle=True):
        """Construct train dataloader"""
        dataset = SavedSampleDataset(f"{self.sample_dir}/train", self.gsp_zarr_path)
        return DataLoader(dataset, shuffle=shuffle, **self._common_dataloader_kwargs)

    def val_dataloader(self, shuffle=False):
        """Construct val dataloader"""
        dataset = SavedSampleDataset(f"{self.sample_dir}/val", self.gsp_zarr_path)
        return DataLoader(dataset, shuffle=shuffle, **self._common_dataloader_kwargs)


class SavedPredictionDataset(Dataset):
    def __init__(self, sample_dir):
        self.sample_filepaths = glob(f"{sample_dir}/*.pt")

    def __len__(self):
        return len(self.sample_filepaths)

    def __getitem__(self, idx):
        return torch.load(self.sample_filepaths[idx])


class SavedPredictionDataModule(LightningDataModule):
    """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation."""

    def __init__(self, sample_dir: str, batch_size=16, num_workers=0, prefetch_factor=None):
        """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation.

        Args:
            sample_dir: Path to the directory of pre-saved batches.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.sample_dir = sample_dir

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
        )

    def train_dataloader(self, shuffle=True):
        """Construct train dataloader"""
        dataset = SavedPredictionDataset(f"{self.sample_dir}/train")
        return DataLoader(dataset, shuffle=shuffle, **self._common_dataloader_kwargs)

    def val_dataloader(self, shuffle=False):
        """Construct val dataloader"""
        dataset = SavedPredictionDataset(f"{self.sample_dir}/val")
        return DataLoader(dataset, shuffle=shuffle, **self._common_dataloader_kwargs)
