""" Data module for pytorch lightning """

import torch
from lightning.pytorch import LightningDataModule
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import FileLister, IterDataPipe
from ocf_datapipes.utils.consts import BatchKey
from ocf_datapipes.load import OpenGSP
from ocf_datapipes.training.pvnet import normalize_gsp
from torchdata.datapipes.iter import Zipper

from pvnet.data.datamodule import (
    copy_batch_to_device,
    batch_to_tensor,
    split_batches,
)
# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class GetNationalPVLive(IterDataPipe):
    """Select national output targets for given times"""
    def __init__(self, gsp_data, times_datapipe):
        """Select national output targets for given times
        
        Args:
            gsp_data: xarray Dataarray of the national outputs
            times_datapipe: IterDataPipe yeilding arrays of target times.
        """
        self.gsp_data = gsp_data
        self.times_datapipe = times_datapipe
    
    def __iter__(self):
        
        gsp_data = self.gsp_data
        for times in self.times_datapipe:
            national_outputs = torch.as_tensor(
                gsp_data.sel(time_utc=times.cpu().numpy().astype("datetime64[s]")).values
            )
            yield national_outputs


class GetBatchTime(IterDataPipe):
    """Extract the valid times from the concurrent sample batch"""
    
    def __init__(self, sample_datapipe):
        """Extract the valid times from the concurrent sample batch
        
        Args:
            sample_datapipe: IterDataPipe yeilding concurrent sample batches
        """
        self.sample_datapipe = sample_datapipe
    
    def __iter__(self):
        for sample in self.sample_datapipe:
            # Times for each GSP in the sample batch should be the same - take first
            id0 = sample[BatchKey.gsp_t0_idx]
            times = sample[BatchKey.gsp_time_utc][0, id0+1:]
            yield times
        

class PivotDictList(IterDataPipe):
    """Convert list of dicts to dict of lists"""
    
    def __init__(self, source_datapipe):
        """Convert list of dicts to dict of lists
        
        Args:
            source_datapipe: 
        """
        self.source_datapipe = source_datapipe
        
    def __iter__(self):
        for list_of_dicts in self.source_datapipe:
            keys = list_of_dicts[0].keys()
            batch_dict = {k: [d[k] for d in list_of_dicts] for k in keys}
            yield batch_dict
            
            
class DictApply(IterDataPipe):
    """Apply functions to elements of a dictionary and return processed dictionary."""
    
    def __init__(self, source_datapipe, **transforms):
        """Apply functions to elements of a dictionary and return processed dictionary.
        
        Args:
            source_datapipe: Datapipe which yields dicts
            **transforms: key-function pairs
        """
        self.source_datapipe = source_datapipe
        self.transforms = transforms
        
    def __iter__(self):
        for d in self.source_datapipe:
            for key, function in self.transforms.items():
                d[key] = function(d[key])
            yield d


class ZipperDict(IterDataPipe):
    """Yield samples from multiple datapipes as a dict"""
    
    def __init__(self, **datapipes):
        """Yield samples from multiple datapipes as a dict.
        
        Args:
            **datapipes: Named datapipes
        """
        self.keys = list(datapipes.keys())
        self.source_datapipes = Zipper(*[datapipes[key] for key in self.keys])
        
    def __iter__(self):
        for outputs in self.source_datapipes:
            yield {key: value for key, value in zip(self.keys, outputs)}

            
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
    
    def _get_premade_batches_datapipe(self, subdir, shuffle=False, add_filename=False):
        
        # Load presaved concurrent sample batches
        file_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)
        
        if shuffle:
            file_pipeline = file_pipeline.shuffle(buffer_size=1000)
        if add_filename:
            file_pipeline, file_pipeline_copy = file_pipeline.fork(2, buffer_size=5)
        
        sample_pipeline = file_pipeline.sharding_filter().map(torch.load)
        
        # Find national outout simultaneous to concurrent samples
        gsp_data = (
            next(iter(
                OpenGSP(gsp_pv_power_zarr_path=self.gsp_zarr_path)
                .map(normalize_gsp)
            ))
            .sel(gsp_id=0)
            .compute()
        )
        
        sample_pipeline, dp = sample_pipeline.fork(2, buffer_size=5)
        
        times_datapipe, dp = GetBatchTime(dp).fork(2, buffer_size=5)
        
        national_targets_datapipe = GetNationalPVLive(gsp_data, dp)
        
        # Compile the samples
        if add_filename:
            data_pipeline = ZipperDict(
                pvnet_inputs = sample_pipeline,
                national_targets = national_targets_datapipe, 
                times = times_datapipe, 
                filepath = file_pipeline_copy,
            )
        else:
            data_pipeline = ZipperDict(
                pvnet_inputs = sample_pipeline,
                national_targets = national_targets_datapipe, 
                times = times_datapipe, 
            )         
                    
        if self.batch_size is not None:
        
            data_pipeline = PivotDictList(data_pipeline.batch(self.batch_size))
            data_pipeline = DictApply(
                data_pipeline, 
                national_targets=torch.stack, 
                times=torch.stack,
            )
        
        return data_pipeline

    def train_dataloader(self, shuffle=True, add_filename=False):
        """Construct train dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "train", 
            shuffle=shuffle, 
            add_filename=add_filename
        )

        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)

    def val_dataloader(self, shuffle=False, add_filename=False):
        """Construct val dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "val", 
            shuffle=shuffle, 
            add_filename=add_filename
        )        
        rs = MultiProcessingReadingService(**self.readingservice_config)
        return DataLoader2(datapipe, reading_service=rs)

    def test_dataloader(self):
        """Construct test dataloader"""
        raise NotImplementedError

