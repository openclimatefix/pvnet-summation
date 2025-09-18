"""Load a model from its checkpoint directory"""

import glob
import os

import hydra
import torch
import yaml

from pvnet_summation.utils import (
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CONFIG_NAME,
)


def get_model_from_checkpoints(
    checkpoint_dir_path: str,
    val_best: bool = True,
) -> tuple[torch.nn.Module, dict, str | None, str | None]:
    """Load a model from its checkpoint directory

    Args:
        checkpoint_dir_path: str path to the directory with the model files
        val_best (optional): if True, load the best epoch model; otherwise, load the last

    Returns:
        tuple:
            model: nn.Module of pretrained model.
            model_config: dict of model config used to train the model.
            datamodule_config: path to datamodule used to create samples e.g train/test split info.
            experiment_configs: path to the full experimental config.

    """

    # Load lightning training module
    with open(f"{checkpoint_dir_path}/{MODEL_CONFIG_NAME}") as cfg:
        model_config = yaml.load(cfg, Loader=yaml.FullLoader)

    lightning_module = hydra.utils.instantiate(model_config)

    if val_best:
        # Only one epoch (best) saved per model
        files = glob.glob(f"{checkpoint_dir_path}/epoch*.ckpt")
        if len(files) != 1:
            raise ValueError(
                f"Found {len(files)} checkpoints @ {checkpoint_dir_path}/epoch*.ckpt. Expected one."
            )
        
        checkpoint = torch.load(files[0], map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(
            f"{checkpoint_dir_path}/last.ckpt", 
            map_location="cpu", 
            weights_only=True,
        )

    lightning_module.load_state_dict(state_dict=checkpoint["state_dict"])

    # Extract the model from the lightning module
    model = lightning_module.model
    model_config = model_config["model"]

    # Check for datamodule config
    # This only exists if the model was trained with presaved samples
    datamodule_config = f"{checkpoint_dir_path}/{DATAMODULE_CONFIG_NAME}"
    datamodule_config = datamodule_config if os.path.isfile(datamodule_config) else None

    # Check for experiment config
    # For backwards compatibility - this might not always exist
    experiment_config = f"{checkpoint_dir_path}/{FULL_CONFIG_NAME}"
    experiment_config = experiment_config if os.path.isfile(experiment_config) else None

    return model, model_config, datamodule_config, experiment_config
