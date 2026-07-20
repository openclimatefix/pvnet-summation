"""Run training.

This file can be run for example using
>>  python run.py experiment=example_simple
"""

import logging
import sys

import hydra
import torch
from omegaconf import DictConfig

from pvnet_summation.training import train
from pvnet_summation.utils import maybe_apply_debug_mode, print_config

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# Use TF32 tensor cores for fp32 matmul
# General speed up observed
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Runs training"""

    # Forcing debug friendly configuration if requested in config
    maybe_apply_debug_mode(config)

    print_config(config, resolve=True)

    return train(config)


if __name__ == "__main__":
    main()
