""" Data module for pytorch lightning """

import torch
from lightning.pytorch import LightningDataModule
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import FileLister, IterDataPipe
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.load import OpenGSP
from ocf_datapipes.training.pvnet import normalize_gsp

from pvnet.data.datamodule import (
    copy_batch_to_device,
    batch_to_tensor,
    split_batches,
)
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class GetNationalPVLive(IterDataPipe):
    def __init__(self, gsp_data, sample_datapipe, return_times=False):
        self.gsp_data = gsp_data
        self.sample_datapipe = sample_datapipe
        self.return_times = return_times
    
    def __iter__(self):
        gsp_data = self.gsp_data
        for sample in self.sample_datapipe:
            # Times for each GSP in the sample batch should be the same - take first
            id0 = sample[BatchKey.gsp_t0_idx]
            times = sample[BatchKey.gsp_time_utc][0, id0+1:]
            national_outputs = torch.as_tensor(
                gsp_data.sel(time_utc=times.cpu().numpy().astype("datetime64[s]")).values
            )
            
            if self.return_times:
                yield national_outputs, times
            else:
                yield national_outputs

        
class ReorganiseBatch(IterDataPipe):
    """Reoragnise batches for pvnet_summation"""
    def __init__(self, source_datapipe):
        """Reoragnise batches for pvnet_summation
        
        Args:
            source_datapipe: Zipped datapipe of list[tuple(NumpyBatch, national_outputs)]
        """
        self.source_datapipe = source_datapipe
        
    def __iter__(self):
        for batch in self.source_datapipe:
            yield dict(
                pvnet_inputs = [sample[0] for sample in batch],
                national_targets = torch.stack([sample[1] for sample in batch]),
                times = torch.stack([sample[2] for sample in batch]),
            )
            
class DataModule(LightningDataModule):
    """Datamodule for training pvnet_summation."""

    def __init__(
        self,
        batch_dir: str,
        gsp_zarr_path: str,
        batch_size=16,
        num_workers=0,
        prefetch_factor=2,
    ):
        """Datamodule for training pvnet_summation.

        Args:
            batch_dir: Path to the directory of pre-saved batches.
            gsp_zarr_path: Path to zarr file containing GSP ID 0 outputs
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.gsp_zarr_path = gsp_zarr_path
        self.batch_size = batch_size
        self.batch_dir = batch_dir

        self.readingservice_config = dict(
            num_workers=num_workers,
            multiprocessing_context="spawn",
            worker_prefetch_cnt=prefetch_factor,
        )
    
    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        data_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)
        if shuffle:
            data_pipeline = data_pipeline.shuffle(buffer_size=1000)
        
        data_pipeline = data_pipeline.sharding_filter().map(torch.load)
        
        # Add the national target
        data_pipeline, dp = data_pipeline.fork(2, buffer_size=5)
        
        gsp_datapipe = OpenGSP(gsp_pv_power_zarr_path=self.gsp_zarr_path).map(normalize_gsp)
        gsp_data = next(iter(gsp_datapipe)).sel(gsp_id=0).compute()
        
        national_targets_datapipe, times_datapipe = (
            GetNationalPVLive(gsp_data, dp, return_times=True).unzip(sequence_length=2)
        )
        data_pipeline = data_pipeline.zip(national_targets_datapipe, times_datapipe)
        
        data_pipeline = ReorganiseBatch(data_pipeline.batch(self.batch_size))
        
        return data_pipeline

    def train_dataloader(self):
        """Construct train dataloader"""
        datapipe = self._get_premade_batches_datapipe("train", shuffle=True)

        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)

    def val_dataloader(self):
        """Construct val dataloader"""
        datapipe = self._get_premade_batches_datapipe("val")
        
        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)

    def test_dataloader(self):
        """Construct test dataloader"""
        datapipe = self._get_premade_batches_datapipe("test")

        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)
