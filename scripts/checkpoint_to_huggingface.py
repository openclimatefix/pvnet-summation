"""Command line tool to push locally save model checkpoints to huggingface

use:
python checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import glob
import os
import tempfile
from pathlib import Path
from typing import Optional

import hydra
import torch
import typer
import wandb
from pyaml_env import parse_config

import pvnet_summation


def push_to_huggingface(
    checkpoint_dir_path: str,
    huggingface_repo: str = "openclimatefix/pvnet_v2_summation",
    wandb_repo: str = "openclimatefix/pvnet_summation",
    val_best: bool = True,
    wandb_id: Optional[str] = None,
    local_path: Optional[str] = None,
    push_to_hub: bool = True,
):
    """Push a local model to openclimatefix/pvnet_v2_summation huggingface model repo

    checkpoint_dir_path (str): Path of the chekpoint directory
    huggingface_repo: Name of the HuggingFace repo to push the model to
    wandb_repo: Name of the wandb repo which has training logs
    val_best (bool): Use best model according to val loss, else last saved model
    wandb_id (str): The wandb ID code
    local_path (str): Where to save the local copy of the model
    push_to_hub (bool): Whether to push the model to the hub or just create local version.
    """

    assert push_to_hub or local_path is not None

    os.path.dirname(os.path.abspath(__file__))

    # Check if checkpoint dir name is wandb run ID
    if wandb_id is None:
        all_wandb_ids = [run.id for run in wandb.Api().runs(path=wandb_repo)]
        dirname = checkpoint_dir_path.split("/")[-1]
        if dirname in all_wandb_ids:
            wandb_id = dirname

    # Load the model
    model_config = parse_config(f"{checkpoint_dir_path}/model_config.yaml")

    model = hydra.utils.instantiate(model_config)

    if val_best:
        # Only one epoch (best) saved per model
        files = glob.glob(f"{checkpoint_dir_path}/epoch*.ckpt")
        assert len(files) == 1
        checkpoint = torch.load(files[0], map_location="cpu")
    else:
        checkpoint = torch.load(f"{checkpoint_dir_path}/last.ckpt", map_location="cpu")

    model.load_state_dict(state_dict=checkpoint["state_dict"])

    # Push to hub
    if local_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        model_output_dir = temp_dir.name
    else:
        model_output_dir = local_path

    model.save_pretrained(
        model_output_dir,
        config=model_config,
        data_config=None,
        wandb_repo=wandb_repo,
        wandb_ids=[wandb_id],
        push_to_hub=push_to_hub,
        repo_id=huggingface_repo,
        card_template_path=(
            Path(pvnet_summation.__file__).parent / "models" / "model_card_template.md"
        ),
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    typer.run(push_to_huggingface)
