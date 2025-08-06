import pytest

import yaml

import dask


from pvnet_summation.models import DenseModel

from pvnet_summation.data.datamodule import (
    StreamedDataModule,
    PresavedDataset,
    SumTensorBatch,
)

from torch.utils.data import default_collate

from ocf_data_sampler.torch_datasets.sample.base import batch_to_tensor, copy_batch_to_device


import os

import dask.array
import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
import hydra


from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration



_top_test_directory = os.path.dirname(os.path.realpath(__file__))


uk_sat_area_string = """msg_seviri_rss_3km:
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


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory) -> str:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path) -> str:
    variables = [
        "IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
        "IR_134", "VIS006", "VIS008", "WV_062", "WV_073",
    ]
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")
    y = np.linspace(start=4191563, stop=5304712, num=100)
    x = np.linspace(start=15002, stop=-1824245, num=100)

    coords = (
        ("variable", variables),
        ("time", times),
        ("y_geostationary", y),
        ("x_geostationary", x),
    )

    data = dask.array.zeros(
        shape=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )

    attrs = {"area": uk_sat_area_string}

    ds = xr.DataArray(data=data, coords=coords, attrs=attrs).to_dataset(name="data")

    zarr_path = session_tmp_path / "test_sat.zarr"
    ds.to_zarr(zarr_path)

    return zarr_path


@pytest.fixture(scope="session")
def ukv_zarr_path(session_tmp_path) -> str:
    init_times = pd.date_range(start="2023-01-01 00:00", freq="180min", periods=24 * 7)
    variables = ["si10", "dswrf", "t", "prate"]
    steps = pd.timedelta_range("0h", "24h", freq="1h")
    x = np.linspace(-239_000, 857_000, 200)
    y = np.linspace(-183_000, 1425_000, 200)
    
    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    data = dask.array.random.uniform(
        low=0,
        high=200,
        size=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(1, -1, -1, 50, 50),
    ).astype(np.float32)

    ds = xr.DataArray(data=data, coords=coords).to_dataset(name="UKV")

    zarr_path = session_tmp_path / "ukv_nwp.zarr"
    ds.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def ecmwf_zarr_path(session_tmp_path) -> str:
    init_times = pd.date_range(start="2023-01-01 00:00", freq="6h", periods=24 * 7)
    variables = ["t2m", "dswrf", "mcc"]
    steps = pd.timedelta_range("0h", "14h", freq="1h")
    lons = np.arange(-12.0, 3.0, 0.1)
    lats = np.arange(48.0, 65.0, 0.1)

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("longitude", lons),
        ("latitude", lats),
    )

    data = dask.array.random.uniform(
        low=0,
        high=200,
        size=tuple(len(coord_values) for _, coord_values in coords),
        chunks=(1, -1, -1, 50, 50),
    ).astype(np.float32)

    ds = xr.DataArray(data=data, coords=coords).to_dataset(name="ECMWF_UK")

    zarr_path = session_tmp_path / "ukv_ecmwf.zarr"
    ds.to_zarr(zarr_path)
    yield zarr_path


@pytest.fixture(scope="session")
def gsp_zarr_path(session_tmp_path) -> str:
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

    coords = (
        ("datetime_gmt", times),
        ("gsp_id", gsp_ids),
    )

    ds_uk_gsp = xr.Dataset({
        "capacity_mwp": xr.DataArray(capacity, coords=coords),
        "installedcapacity_mwp": xr.DataArray(capacity, coords=coords),
        "generation_mw": xr.DataArray(generation, coords=coords),
    })

    zarr_path = session_tmp_path / "uk_gsp.zarr"
    ds_uk_gsp.to_zarr(zarr_path)
    return zarr_path


@pytest.fixture(scope="session")
def data_config_path(
    session_tmp_path, 
    sat_zarr_path, 
    ukv_zarr_path, 
    ecmwf_zarr_path, 
    gsp_zarr_path
) -> str:  
    
    # Populate the config with the generated zarr paths
    config = load_yaml_configuration(f"{_top_test_directory}/test_data/data_config.yaml")
    config.input_data.nwp["ukv"].zarr_path = str(ukv_zarr_path)
    config.input_data.nwp["ecmwf"].zarr_path = str(ecmwf_zarr_path)
    config.input_data.satellite.zarr_path = str(sat_zarr_path)
    config.input_data.gsp.zarr_path = str(gsp_zarr_path)

    filename = f"{session_tmp_path}/data_config.yaml"
    save_yaml_configuration(config, filename)
    return filename


@pytest.fixture(scope="session")
def pvnet_model_config() -> dict:
    model_config_path = f"{_top_test_directory}/test_data/pvnet_model_config.yaml"
    with open(model_config_path, "r") as stream:
        model_config = yaml.safe_load(stream)
    return model_config


@pytest.fixture(scope="session")
def presaved_samples_dir(session_tmp_path, data_config_path, pvnet_model_config) -> str:
        
    num_samples = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a PVNet model
    pvnet_model_config["location_id_mapping"] = {i:i for i in range(1, 318)}
    pvnet_model = hydra.utils.instantiate(pvnet_model_config)
    pvnet_model.to(device).requires_grad_(False)

    # Save PVNet predictions
    sample_dir = f"{session_tmp_path}/samples"
    os.makedirs(f"{sample_dir}/train")
    os.makedirs(f"{sample_dir}/val")

    datamodule = StreamedDataModule(
        configuration=data_config_path,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    )

    dataloader = datamodule.train_dataloader(shuffle=True)

    sample = next(iter(dataloader))

    del dataloader

    # Run PVNet inputs though model
    x = copy_batch_to_device(batch_to_tensor(sample["pvnet_inputs"]), device)
    pvnet_outputs = pvnet_model(x).detach().cpu()

    # Create version of sample without the PVNet inputs and save
    sample_to_save = {k: v.clone() for k, v in sample.items() if k!="pvnet_inputs"}

    sample_to_save["pvnet_outputs"] = pvnet_outputs

    for i in range(num_samples):
        torch.save(sample_to_save, f"{sample_dir}/train/{i:06}.pt")
        torch.save(sample_to_save, f"{sample_dir}/val/{i:06}.pt")
        
    return sample_dir


@pytest.fixture()
def model_kwargs(pvnet_model_config) -> dict:
    kwargs = dict(
        output_network = dict(
            _target_ = "pvnet.models.late_fusion.linear_networks.networks.ResFCNet",
            _partial_ = True,
            fc_hidden_features = 128,
            n_res_blocks = 2,
            res_block_layers = 2,
            dropout_frac = 0.0,
        ),
        predict_difference_from_sum=True,
        history_minutes = pvnet_model_config["history_minutes"],
        forecast_minutes = pvnet_model_config["forecast_minutes"],
        interval_minutes = pvnet_model_config["interval_minutes"],
        num_input_locations = len(pvnet_model_config["location_id_mapping"]),
        input_quantiles = pvnet_model_config["output_quantiles"],
    )

    return hydra.utils.instantiate(kwargs)


@pytest.fixture()
def model(model_kwargs) -> DenseModel:
    return DenseModel(output_quantiles=None, **model_kwargs)


@pytest.fixture(scope="session")
def batch(presaved_samples_dir) -> SumTensorBatch:
    dataset = PresavedDataset(sample_dir=f"{presaved_samples_dir}/train")
    sample = dataset[0]
    return default_collate([sample, sample])
