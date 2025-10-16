"""Training"""
import logging
import os

import hydra
import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)
from omegaconf import DictConfig, OmegaConf
from pvnet.models import BaseModel as PVNetBaseModel
from tqdm import tqdm

from pvnet_summation.data.datamodule import PresavedDataModule, StreamedDataModule
from pvnet_summation.utils import (
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CONFIG_NAME,
    create_pvnet_model_config,
)

log = logging.getLogger(__name__)


def resolve_monitor_loss(output_quantiles: list | None) -> str:
    """Return the desired metric to monitor based on whether quantile regression is being used.

    Adds the option to use
        monitor: "${resolve_monitor_loss:${model.model.output_quantiles}}"
    in early stopping and model checkpoint callbacks so the callbacks config does not need to be
    modified depending on whether quantile regression is being used or not.
    """
    if output_quantiles is None:
        return "MAE/val"
    else:
        return "quantile_loss/val"


OmegaConf.register_new_resolver("resolve_monitor_loss", resolve_monitor_loss)


def train(config: DictConfig) -> None:
    """Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Get the PVNet model
    pvnet_model = PVNetBaseModel.from_pretrained(
        model_id=config.datamodule.pvnet_model.model_id,
        revision=config.datamodule.pvnet_model.revision
    )
    
    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)
    # Set summation model parameters to align with the input PVNet model
    config.model.model.history_minutes = pvnet_model.history_minutes
    config.model.model.forecast_minutes = pvnet_model.forecast_minutes
    config.model.model.interval_minutes = pvnet_model.interval_minutes
    config.model.model.num_input_locations = len(pvnet_model.location_id_mapping)
    config.model.model.input_quantiles = pvnet_model.output_quantiles
    OmegaConf.set_struct(config, True)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Compute and save the PVNet predictions before training the summation model
    save_dir = (
        f"{config.sample_save_dir}/{config.datamodule.pvnet_model.model_id}"
        f"/{config.datamodule.pvnet_model.revision}"
    )

    if os.path.isdir(save_dir):
        log.info(
            f"PVNet output directory already exists: {save_dir}\n"
            "Skipping saving new outputs. The existing saved outputs will be loaded."
        )
    else:
        log.info(f"Saving PVNet outputs to {save_dir}")

        # Move to device and disable gradients for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pvnet_model.to(device).requires_grad_(False)

        os.makedirs(f"{save_dir}/train")
        os.makedirs(f"{save_dir}/val")

        pvnet_data_config_path = f"{save_dir}/pvnet_data_config.yaml"

        data_source_paths = OmegaConf.to_container(
            config.datamodule.data_source_paths, 
            resolve=True,
        )

        create_pvnet_model_config(
            save_path=pvnet_data_config_path,
            repo=config.datamodule.pvnet_model.model_id,
            commit=config.datamodule.pvnet_model.revision,
            data_source_paths=data_source_paths,
        )

        datamodule = StreamedDataModule(
            configuration=pvnet_data_config_path,
            num_workers=config.datamodule.num_workers,
            prefetch_factor=config.datamodule.prefetch_factor,
            train_period=config.datamodule.train_period,
            val_period=config.datamodule.val_period,
            persistent_workers=False,
            seed=config.datamodule.seed,
            dataset_pickle_dir=config.datamodule.dataset_pickle_dir,
        )

        datamodule.setup()

        for dataloader_func, max_num_samples, split in [
            (datamodule.train_dataloader, config.datamodule.max_num_train_samples, "train",),
            (datamodule.val_dataloader, config.datamodule.max_num_val_samples, "val"),
        ]:
            
            log.info(f"Saving {split} outputs")
            dataloader = dataloader_func(shuffle=True)

            # If max_num_samples set to None use all samples
            max_num_samples = max_num_samples or len(dataloader)

            for i, sample in tqdm(zip(range(max_num_samples), dataloader), total=max_num_samples):
                # Run PVNet inputs though model
                x = copy_batch_to_device(batch_to_tensor(sample["pvnet_inputs"]), device)
                pvnet_outputs = pvnet_model(x).detach().cpu()

                # Create version of sample without the PVNet inputs and save
                sample_to_save = {k: v.clone() for k, v in sample.items() if k!="pvnet_inputs"}

                sample_to_save["pvnet_outputs"] = pvnet_outputs
                torch.save(sample_to_save, f"{save_dir}/{split}/{i:06}.pt")
            
            del dataloader

        datamodule.teardown()


    datamodule = PresavedDataModule(
        sample_dir=save_dir,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
        prefetch_factor=config.datamodule.prefetch_factor,
        persistent_workers=config.datamodule.persistent_workers,
    )

    # Init lightning loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    # - this makes it easy to push the model to huggingface
    use_wandb_logger = False
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    # Set the output directory based in the wandb-id of the run
    if use_wandb_logger:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                # Calling the .experiment property instantiates a wandb run
                wandb_id = wandb_logger.experiment.id

                # Save the run results to the expected parent folder but with the folder name
                # set by the wandb ID
                save_dir = f"{os.path.dirname(callback.dirpath)}/{wandb_id}"

                callback.dirpath = save_dir
                
                # Save the model config
                os.makedirs(save_dir, exist_ok=True)
                OmegaConf.save(config.model, f"{save_dir}/{MODEL_CONFIG_NAME}")

                # Save the datamodule config
                OmegaConf.save(config.datamodule, f"{save_dir}/{DATAMODULE_CONFIG_NAME}")

                # Save the full hydra config to the output directory and to wandb
                OmegaConf.save(config, f"{save_dir}/{FULL_CONFIG_NAME}")
                wandb_logger.experiment.save(f"{save_dir}/{FULL_CONFIG_NAME}", base_path=save_dir)


    # Init lightning model
    model = hydra.utils.instantiate(config.model)
                
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)