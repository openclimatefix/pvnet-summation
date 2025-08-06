from pvnet_summation.data.datamodule import StreamedDataset, PresavedDataset


def test_streameddataset(data_config_path):
    dataset = StreamedDataset(config_filename=data_config_path)
    sample = dataset[0]

    expected_keys = set(
        ["pvnet_inputs", "target", "valid_times", "last_outturn", "relative_capacity",]
    )
    assert set(sample.keys())==expected_keys


def test_presaveddataset(presaved_samples_dir):
    dataset = PresavedDataset(sample_dir=f"{presaved_samples_dir}/train")
    sample = dataset[0]

    expected_keys = set(
        ["pvnet_outputs", "target", "valid_times", "last_outturn", "relative_capacity",]
    )
    assert set(sample.keys())==expected_keys