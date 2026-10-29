"""Tests and fixtures for CPU-only Trainer and offline W&B logging."""

import shutil
from pathlib import Path

import hydra
import pytest
import yaml
from omegaconf import DictConfig
from pvnet.utils import DATA_CONFIG_NAME, MODEL_CONFIG_NAME, PYTORCH_WEIGHTS_NAME
from safetensors.torch import save_file

from pvnet_summation.training.train import train as pvnetsummation_train


@pytest.fixture()
def wandb_save_dir(tmp_path) -> str:
    """Return W&B save dir under a per-test temp path."""
    save_dir = str(tmp_path / "wandb")
    return save_dir


@pytest.fixture()
def trainer_cfg_cpu() -> dict:
    """Tiny CPU-only Trainer config."""
    return {
        "_target_": "lightning.pytorch.Trainer",
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "accelerator": "cpu",
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
        "enable_progress_bar": False,
    }


@pytest.fixture()
def logger_cfg(wandb_save_dir: str) -> dict:
    """W&B logger config."""
    return {
        "wandb": {
            "_target_": "lightning.pytorch.loggers.wandb.WandbLogger",
            "project": "pvnet-summation-tests",
            "save_dir": wandb_save_dir,
            "offline": True,
            "name": "train-offline-integration",
            "log_model": False,
        }
    }


@pytest.fixture()
def ckpt_cfg(wandb_save_dir: str) -> dict:
    """ModelCheckpoint config."""
    ckpt_dir = str(Path(wandb_save_dir).parent / "ckpts")
    return {
        "ckpt": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": ckpt_dir,
            "save_top_k": 1,
            "monitor": "MAE/val",
            "mode": "min",
            "save_on_train_epoch_end": False,
        },
        "last_ckpt": {
            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
            "dirpath": ckpt_dir,
            "filename": "last",
            "enable_version_counter": False,
            "save_on_train_epoch_end": False,
        },
    }


@pytest.fixture()
def pvnet_model_dir(tmp_path, pvnet_model_config, data_config_path) -> str:
    """Save a locally built PVNet model to a directory in the layout used on HuggingFace.

    Passing this directory as the `model_id` makes PVNet load the model and data config from it
    rather than downloading from HuggingFace. `create_pvnet_model_config` then overwrites the zarr
    paths in the data config using `data_source_paths`.
    """
    model_dir = tmp_path / "pvnet_model"
    model_dir.mkdir()

    pvnet_model = hydra.utils.instantiate(pvnet_model_config)
    save_file(pvnet_model.state_dict(), model_dir / PYTORCH_WEIGHTS_NAME)

    with open(model_dir / MODEL_CONFIG_NAME, "w") as f:
        yaml.safe_dump(pvnet_model_config, f)

    shutil.copy(data_config_path, model_dir / DATA_CONFIG_NAME)

    return str(model_dir)


def build_model_cfg() -> dict:
    """Build config for PVNetSummationLightningModule + minimal DenseModel.

    The time settings, number of input locations and input quantiles are filled in by train()
    from the PVNet model, so are not set here.
    """
    return {
        "_target_": "pvnet_summation.training.lightning_module.PVNetSummationLightningModule",
        "model": {
            "_target_": "pvnet_summation.models.dense_model.DenseModel",
            "output_quantiles": [0.1, 0.5, 0.9],
            "output_network": {
                "_target_": "pvnet.models.late_fusion.linear_networks.networks.ResFCNet",
                "_partial_": True,
                "fc_hidden_features": 32,
                "n_res_blocks": 1,
                "res_block_layers": 2,
                "dropout_frac": 0.1,
            },
            "predict_difference_from_sum": True,
        },
        "optimizer": {
            "_target_": "pvnet_summation.optimizers.AdamW",
            "lr": 1e-3,
        },
    }


def test_train(
    tmp_path,
    pvnet_model_dir,
    sat_zarr_path,
    ukv_zarr_path,
    ecmwf_zarr_path,
    generation_zarr_path,
    trainer_cfg_cpu,
    logger_cfg,
    ckpt_cfg,
    wandb_save_dir,
):
    """Run PVNet over streamed samples, then train the summation model with W&B offline."""
    cfg = DictConfig({
        "seed": 42,
        "sample_save_dir": str(tmp_path / "pvnet_outputs"),
        "datamodule": {
            "data_source_paths": {
                "generation": str(generation_zarr_path),
                "satellite": str(sat_zarr_path),
                "nwp": {
                    "ukv": str(ukv_zarr_path),
                    "ecmwf": str(ecmwf_zarr_path),
                },
            },
            "batch_size": 2,
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "max_num_train_samples": 2,
            "max_num_val_samples": 2,
            "pvnet_model": {
                "model_id": pvnet_model_dir,
                "revision": None,
            },
            "train_period": None,
            "val_period": None,
            "seed": "${seed}",
            "dataset_pickle_dir": None,
        },
        "model": build_model_cfg(),
        "logger": logger_cfg,
        "callbacks": ckpt_cfg,
        "trainer": trainer_cfg_cpu,
        "model_name": "test_model",
    })

    pvnetsummation_train(cfg)

    # Check that the PVNet outputs were saved
    pvnet_output_dir = f"{tmp_path}/pvnet_outputs/{pvnet_model_dir}/None"
    assert len(list(Path(pvnet_output_dir, "train").glob("*.pt"))) == 2
    assert len(list(Path(pvnet_output_dir, "val").glob("*.pt"))) == 2

    # Check that checkpoint exists
    ckpt_paths = list(Path(wandb_save_dir).parent.glob("*/last.ckpt"))
    assert len(ckpt_paths) == 1, (
        f"expected one checkpoint called last.ckpt at end of epoch, got {ckpt_paths}"
    )
