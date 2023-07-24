import lightning


def test_model_trainer_fit(model, sample_datamodule):
    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=sample_datamodule)
