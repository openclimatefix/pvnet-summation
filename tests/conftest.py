import pytest

import os
import shutil
import yaml
import hydra

import numpy as np
import pandas as pd
import xarray as xr
import dask
import torch
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset

from pvnet_summation.models.dense_model import DenseModel
from pvnet_summation.data.datamodule import SavedSampleDataset, SavedSampleDataModule


@pytest.fixture(scope="session", autouse=True)
def test_root_directory(request):
    return f"{request.config.rootpath}/tests"


@pytest.fixture(scope="session")
def num_samples():
    return 100


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):
    # Define coords for satellite-like dataset
    variables = [
        "IR_016",
        "IR_039",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
        "VIS006",
        "VIS008",
        "WV_062",
        "WV_073",
    ]
    x = np.linspace(start=15002, stop=-1824245, num=100)
    y = np.linspace(start=4191563, stop=5304712, num=100)
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")

    area_string = """msg_seviri_rss_3km:
        description: MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution
        projection:
            proj: geos
            lon_0: 9.5
            h: 35785831
            x_0: 0
            y_0: 0
            a: 6378169
            rf: 295.488065897014
            no_defs: null
            type: crs
        shape:
            height: 298
            width: 615
        area_extent:
            lower_left_xy: [28503.830075263977, 5090183.970808983]
            upper_right_xy: [-1816744.1169023514, 4196063.827395439]
            units: m
        """

    # Create satellite-like data with some NaNs
    data = dask.array.zeros(
        shape=(len(variables), len(times), len(y), len(x)),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )
    data[:, 10, :, :] = np.nan

    ds = xr.DataArray(
        data=data,
        coords=dict(
            variable=variables,
            time=times,
            y_geostationary=y,
            x_geostationary=x,
        ),
        attrs=dict(area=area_string),
    ).to_dataset(name="data")

    zarr_path = f"{session_tmp_path}/test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(session_tmp_path):
    init_times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="180min")
    steps = pd.timedelta_range("0h", "18h", freq="1h")

    x = np.linspace(-239_000, 857_000, 50)
    y = np.linspace(-183_000, 1225_000, 100)
    variables = ["si10", "dswrf", "t", "prate"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    nwp_array_shape = tuple(len(coord_values) for _, coord_values in coords)

    ds = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape).astype(np.float32),
        coords=coords,
    ).to_dataset(name="UKV")

    ds = ds.chunk(
        {
            "init_time": 1,
            "step": -1,
            "variable": -1,
            "x": -1,
            "y": -1,
        }
    )

    zarr_path = f"{session_tmp_path}/ukv_nwp.zarr"
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def uk_gsp_zarr_path(session_tmp_path):
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

    coords = (
        ("datetime_gmt", times),
        ("gsp_id", gsp_ids),
    )

    da_cap = xr.DataArray(
        capacity,
        coords=coords,
    )

    da_gen = xr.DataArray(
        generation,
        coords=coords,
    )

    ds = xr.Dataset(
        {"capacity_mwp": da_cap, "installedcapacity_mwp": da_cap, "generation_mw": da_gen}
    )

    zarr_path = f"{session_tmp_path}/uk_gsp.zarr"
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def pvnet_config_filename(
    session_tmp_path, nwp_ukv_zarr_path, uk_gsp_zarr_path, sat_zarr_path, test_root_directory
):
    config = load_yaml_configuration(f"{test_root_directory}/test_data/data_config.yaml")
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.gsp.zarr_path = uk_gsp_zarr_path

    filename = f"{session_tmp_path}/configuration.yaml"
    save_yaml_configuration(config, filename)
    return filename


@pytest.fixture(scope="session")
def presaved_samples_dir(session_tmp_path, pvnet_config_filename, num_samples):
    # Set up the sample directory
    samples_dir = f"{session_tmp_path}/presaved_samples"
    os.makedirs(samples_dir, exist_ok=False)
    os.makedirs(f"{samples_dir}/train", exist_ok=False)
    shutil.copyfile(pvnet_config_filename, f"{samples_dir}/data_configuration.yaml")

    # Save a single sample and symlink it to other sample paths to save disk space
    dataset = PVNetUKConcurrentDataset(pvnet_config_filename)
    sample = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    torch.save(sample, f"{samples_dir}/train/000000.pt")

    for i in range(1, num_samples):
        os.system(f"ln -s {samples_dir}/train/000000.pt {samples_dir}/train/{i:06}.pt")

    # Use the same samples for validation
    os.system(f"ln -s {samples_dir}/train {samples_dir}/val")

    return samples_dir


@pytest.fixture(scope="session")
def saved_pvnet_model_path(session_tmp_path, test_root_directory):
    # Create the PVNet model
    model_config_path = f"{test_root_directory}/test_data/pvnet_model_config.yaml"
    with open(model_config_path, "r") as stream:
        model_config = yaml.safe_load(stream)

    model = hydra.utils.instantiate(model_config)

    # Save the model
    model_output_dir = f"{session_tmp_path}/pvnet_model"

    model.save_pretrained(
        model_output_dir,
        config=model_config,
        data_config=f"{test_root_directory}/test_data/data_config.yaml",
        wandb_repo=None,
        wandb_ids="excluded-for-text",
        push_to_hub=False,
        repo_id="openclimatefix/pvnet_uk_region",
    )

    return model_output_dir


@pytest.fixture(scope="session")
def flat_model_kwargs(saved_pvnet_model_path):
    kwargs = dict(
        # These kwargs define the pvnet model which the summation model uses
        model_name=saved_pvnet_model_path,
        model_version=None,
        # These kwargs define the structure of the summation model
        output_network=dict(
            _target_="pvnet.models.multimodal.linear_networks.networks.ResFCNet2",
            _partial_=True,
            fc_hidden_features=128,
            n_res_blocks=2,
            res_block_layers=2,
            dropout_frac=0.0,
        ),
    )
    return hydra.utils.instantiate(kwargs)


@pytest.fixture(scope="session")
def model(flat_model_kwargs):
    return DenseModel(**flat_model_kwargs)


@pytest.fixture(scope="session")
def presaved_predictions_dir(session_tmp_path, presaved_samples_dir, uk_gsp_zarr_path, model):
    # Set up the sample directory
    presaved_preds_dir = f"{session_tmp_path}/presaved_predictions"
    os.makedirs(presaved_preds_dir, exist_ok=False)
    os.makedirs(f"{presaved_preds_dir}/train", exist_ok=False)
    os.makedirs(f"{presaved_preds_dir}/val", exist_ok=False)
    shutil.copyfile(
        f"{presaved_samples_dir}/data_configuration.yaml",
        f"{presaved_preds_dir}/data_configuration.yaml",
    )

    # Make PVNet predictions for all samples and save them
    for split in ["train", "val"]:
        dataset = SavedSampleDataset(
            sample_dir=f"{presaved_samples_dir}/{split}", gsp_zarr_path=uk_gsp_zarr_path
        )

        for i in range(len(dataset)):
            x = dataset[i]
            x["pvnet_outputs"] = model.predict_pvnet_batch([x["pvnet_inputs"]])[0].cpu()
            del x["pvnet_inputs"]

            torch.save(x, f"{presaved_preds_dir}/{split}/{i:06}.pt")

    return presaved_preds_dir


@pytest.fixture(scope="session")
def pvnet_inputs_batch(presaved_samples_dir, uk_gsp_zarr_path):
    datamodule = SavedSampleDataModule(
        sample_dir=presaved_samples_dir,
        gsp_zarr_path=uk_gsp_zarr_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )
    return next(iter(datamodule.train_dataloader()))
