from pvnet_summation.data.datamodule import DataModule
from ocf_datapipes.batch import BatchKey


def test_init(sample_data):
    batch_dir, gsp_zarr_dir = sample_data

    dm = DataModule(
        batch_dir=batch_dir,
        gsp_zarr_path=gsp_zarr_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )


def test_iter(sample_data):
    batch_dir, gsp_zarr_dir = sample_data

    dm = DataModule(
        batch_dir=batch_dir,
        gsp_zarr_path=gsp_zarr_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )

    batch = next(iter(dm.train_dataloader()))

    # batch size is 2
    assert len(batch["pvnet_inputs"]) == 2

    # 317 GSPs in each sample
    # 21 timestamps for each GSP from -120 mins to +480 mins
    assert batch["pvnet_inputs"][0][BatchKey.gsp_time_utc].shape == (317, 21)

    assert batch["times"].shape == (2, 16)

    assert batch["national_targets"].shape == (2, 16)


def test_iter_multiprocessing(sample_data):
    batch_dir, gsp_zarr_dir = sample_data

    dm = DataModule(
        batch_dir=batch_dir,
        gsp_zarr_path=gsp_zarr_dir,
        batch_size=2,
        num_workers=2,
        prefetch_factor=2,
    )

    for batch in dm.train_dataloader():
        # batch size is 2
        assert len(batch["pvnet_inputs"]) == 2

        # 317 GSPs in each sample
        # 21 timestamps for each GSP from -120 mins to +480 mins
        assert batch["pvnet_inputs"][0][BatchKey.gsp_time_utc].shape == (317, 21)

        assert batch["times"].shape == (2, 16)

        assert batch["national_targets"].shape == (2, 16)
