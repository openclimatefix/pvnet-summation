from pvnet_summation.data.datamodule import (
    SavedSampleDataset,
    SavedSampleDataModule,
    SavedPredictionDataset,
    SavedPredictionDataModule,
)


def test_saved_sample_dataset(presaved_samples_dir, uk_gsp_zarr_path, num_samples):
    dataset = SavedSampleDataset(
        sample_dir=f"{presaved_samples_dir}/train", gsp_zarr_path=uk_gsp_zarr_path
    )
    assert len(dataset) == num_samples

    sample = dataset[0]
    assert isinstance(sample, dict)


def test_saved_sample_datamodule(presaved_samples_dir, uk_gsp_zarr_path, num_samples):
    batch_size = 2
    datamodule = SavedSampleDataModule(
        sample_dir=presaved_samples_dir,
        gsp_zarr_path=uk_gsp_zarr_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )

    dataloader = datamodule.train_dataloader()
    assert len(dataloader)==num_samples/batch_size

    batch = next(iter(dataloader))
    assert isinstance(batch, dict)


def test_saved_prediction_dataset(presaved_predictions_dir, num_samples):
    dataset = SavedPredictionDataset(sample_dir=f"{presaved_predictions_dir}/train")
    assert len(dataset) == num_samples

    sample = dataset[0]
    assert isinstance(sample, dict)


def test_saved_prediction_datamodule(presaved_predictions_dir, num_samples):
    batch_size = 2
    datamodule = SavedPredictionDataModule(
        sample_dir=presaved_predictions_dir,
        batch_size=batch_size,
        num_workers=0,
        prefetch_factor=None,
    )
    dataloader = datamodule.train_dataloader()
    assert len(dataloader)==num_samples/batch_size

    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
