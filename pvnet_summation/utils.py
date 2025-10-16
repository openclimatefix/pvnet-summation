"""Utils"""
import logging

import rich.syntax
import rich.tree
import yaml
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from pvnet.models.base_model import BaseModel as PVNetBaseModel

logger = logging.getLogger(__name__)


PYTORCH_WEIGHTS_NAME = "model_weights.safetensors"
MODEL_CONFIG_NAME = "model_config.yaml"
DATAMODULE_CONFIG_NAME = "datamodule_config.yaml"
FULL_CONFIG_NAME =  "full_experiment_config.yaml"
MODEL_CARD_NAME = "README.md"



def maybe_apply_debug_mode(config: DictConfig) -> None:
    """Check if debugging run is requested and force debug-frendly configuration

    Controlled by main config file

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # Force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        logger.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("prefetch_factor"):
            config.datamodule.prefetch_factor = None

    # Disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: tuple[str, ...] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (tuple[str, ...], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

def populate_config_with_data_data_filepaths(config: dict, data_source_paths: dict) -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
        data_source_paths: A dictionary of data paths for the different input sources
    """

    # Replace the GSP data path
    config["input_data"]["gsp"]["zarr_path"] =  data_source_paths["gsp"]

    # Replace satellite data path if using it
    if "satellite" in config["input_data"]:
        if config["input_data"]["satellite"]["zarr_path"] != "":
            config["input_data"]["satellite"]["zarr_path"] = data_source_paths["satellite"]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            provider = nwp_config[nwp_source]["provider"]
            assert provider in data_source_paths["nwp"], f"Missing NWP path: {provider}"
            nwp_config[nwp_source]["zarr_path"] = data_source_paths["nwp"][provider]

    return config


def create_pvnet_model_config(
    save_path: str, 
    repo: str, 
    commit: str, 
    data_source_paths: dict,
) -> None:
    """Create the data config needed to run the PVNet model"""
    data_config_path = PVNetBaseModel.get_data_config(repo, revision=commit)

    with open(data_config_path) as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)

    data_config = populate_config_with_data_data_filepaths(data_config, data_source_paths)

    with open(save_path, "w") as file:
        yaml.dump(data_config, file, default_flow_style=False)