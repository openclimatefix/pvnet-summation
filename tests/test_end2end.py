import lightning

from pvnet_summation.data.datamodule import PresavedDataModule
from pvnet_summation.optimizers import AdamW
from pvnet_summation.training.lightning_module import PVNetSummationLightningModule


def test_model_trainer_fit(model, presaved_samples_dir):

    datamodule = PresavedDataModule(
        sample_dir=presaved_samples_dir,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )

    model_module = PVNetSummationLightningModule(
        model=model,
        optimizer=AdamW()
    )

    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True)
    trainer.fit(model=model_module, datamodule=datamodule)
