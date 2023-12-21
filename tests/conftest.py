import os

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import math
import glob
import tempfile
from pvnet_summation.models.model import Model


from ocf_datapipes.batch import BatchKey
from datetime import timedelta

from pvnet_summation.data.datamodule import DataModule


def construct_batch_by_sample_duplication(og_batch, i):
    """From a batch of data, take the ith sample and repeat it 317 to create a new batch"""
    new_batch = {}

    # Need to loop through these keys and add to batch
    ununsed_keys = list(og_batch.keys())

    # NWP is nested so needs to be treated differently
    if BatchKey.nwp in og_batch:
        og_nwp_batch = og_batch[BatchKey.nwp]
        new_nwp_batch = {}
        for nwp_source, og_nwp_source_batch in og_nwp_batch.items():
            new_nwp_source_batch = {}
            for key, value in og_nwp_source_batch.items():
                if isinstance(value, torch.Tensor):
                    n_dims = len(value.shape)
                    repeats = (317,) + tuple(1 for dim in range(n_dims - 1))
                    new_nwp_source_batch[key] = value[i : i + 1].repeat(repeats)[:317]
                else:
                    new_nwp_source_batch[key] = value
            new_nwp_batch[nwp_source] = new_nwp_source_batch

        new_batch[BatchKey.nwp] = new_nwp_batch
        ununsed_keys.remove(BatchKey.nwp)

    for key in ununsed_keys:
        if isinstance(og_batch[key], torch.Tensor):
            n_dims = len(og_batch[key].shape)
            repeats = (317,) + tuple(1 for dim in range(n_dims - 1))
            new_batch[key] = og_batch[key][i : i + 1].repeat(repeats)[:317]
        else:
            new_batch[key] = og_batch[key]

    return new_batch


@pytest.fixture()
def sample_data():
    # Copy small batches to fake 317 GSPs in each
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(f"{tmpdirname}/train")
        os.makedirs(f"{tmpdirname}/val")

        # Grab times from batch to make national output zarr
        times = []

        file_n = 0
        for file in glob.glob("tests/test_data/sample_batches/train/*.pt"):
            og_batch = torch.load(file)

            for i in range(og_batch[BatchKey.gsp_time_utc].shape[0]):
                # Duplicate sample to fake 317 GSPs
                new_batch = construct_batch_by_sample_duplication(og_batch, i)

                # Save fopr both train and val
                torch.save(new_batch, f"{tmpdirname}/train/{file_n:06}.pt")
                torch.save(new_batch, f"{tmpdirname}/val/{file_n:06}.pt")

                file_n += 1

                times += [new_batch[BatchKey.gsp_time_utc][i].numpy().astype("datetime64[s]")]

        times = np.unique(np.sort(np.concatenate(times)))

        da_output = xr.DataArray(
            data=np.random.uniform(size=(len(times), 1)),
            dims=["datetime_gmt", "gsp_id"],
            coords=dict(
                datetime_gmt=times,
                gsp_id=[0],
            ),
        )

        da_cap = xr.DataArray(
            data=np.ones((len(times), 1)),
            dims=["datetime_gmt", "gsp_id"],
            coords=dict(
                datetime_gmt=times,
                gsp_id=[0],
            ),
        )

        ds = xr.Dataset(
            data_vars=dict(
                generation_mw=da_output,
                installedcapacity_mwp=da_cap,
                capacity_mwp=da_cap,
            ),
        )

        ds.to_zarr(f"{tmpdirname}/gsp.zarr")

        yield tmpdirname, f"{tmpdirname}/gsp.zarr"


@pytest.fixture()
def sample_datamodule(sample_data):
    batch_dir, gsp_zarr_dir = sample_data

    dm = DataModule(
        batch_dir=batch_dir,
        gsp_zarr_path=gsp_zarr_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )

    return dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    return batch


@pytest.fixture()
def model_kwargs():
    # These kwargs define the pvnet model which the summation model uses
    kwargs = dict(
        model_name="openclimatefix/pvnet_v2",
        model_version="22e577100d55787eb2547d701275b9bb48f7bfa0",
    )
    return kwargs


@pytest.fixture()
def model(model_kwargs):
    model = Model(**model_kwargs)
    return model


@pytest.fixture()
def quantile_model(model_kwargs):
    model = Model(output_quantiles=[0.1, 0.5, 0.9], **model_kwargs)
    return model
