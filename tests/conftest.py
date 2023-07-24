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


from ocf_datapipes.utils.consts import BatchKey
from datetime import timedelta

from pvnet_summation.data.datamodule import DataModule


@pytest.fixture()
def sample_data():
    # Copy small batches to fake 317 GSPs in each
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.makedirs(f"{tmpdirname}/train")
        os.makedirs(f"{tmpdirname}/val")

        # Grab times from batch to make national output zarr
        times = []

        file_n = 0
        for file in glob.glob("tests/data/sample_batches/train/*.pt"):
            batch = torch.load(file)

            this_batch = {}
            for i in range(batch[BatchKey.gsp_time_utc].shape[0]):
                # Duplicate sample to fake 317 GSPs
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        n_dims = len(batch[key].shape)
                        repeats = (317,) + tuple(1 for dim in range(n_dims - 1))
                        this_batch[key] = batch[key][i : i + 1].repeat(repeats)[:317]
                    else:
                        this_batch[key] = batch[key]

                # Save fopr both train and val
                torch.save(this_batch, f"{tmpdirname}/train/{file_n:06}.pt")
                torch.save(this_batch, f"{tmpdirname}/val/{file_n:06}.pt")

                file_n += 1

                times += [batch[BatchKey.gsp_time_utc][i].numpy().astype("datetime64[s]")]

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
        prefetch_factor=2,
    )

    return dm


@pytest.fixture()
def sample_batch(sample_datamodule):
    batch = next(iter(sample_datamodule.train_dataloader()))
    return batch


@pytest.fixture()
def model_kwargs():
    kwargs = dict(
        model_name="openclimatefix/pvnet_v2",
        model_version="898630f3f8cd4e8506525d813dd61c6d8de86144",
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
