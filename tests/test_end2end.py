import lightning
import pytest

from pvnet_summation.data.datamodule import SavedPredictionDataModule


@pytest.fixture()
def saved_prediction_datamodule(presaved_predictions_dir):
    return SavedPredictionDataModule(
        sample_dir=presaved_predictions_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )


def test_model_trainer_fit(model, saved_prediction_datamodule):
    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=saved_prediction_datamodule)
