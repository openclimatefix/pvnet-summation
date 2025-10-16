"""Base model for all PVNet submodels"""
import logging
import os
import shutil
import time
from importlib.metadata import version
from math import prod
from pathlib import Path

import hydra
import torch
import yaml
from huggingface_hub import ModelCard, ModelCardData, snapshot_download
from huggingface_hub.hf_api import HfApi
from safetensors.torch import load_file, save_file

from pvnet_summation.data.datamodule import SumTensorBatch
from pvnet_summation.utils import (
    DATAMODULE_CONFIG_NAME,
    FULL_CONFIG_NAME,
    MODEL_CARD_NAME,
    MODEL_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
)


def santize_datamodule(config: dict) -> dict:
    """Create new datamodule config which only keeps the details required for inference"""
    return {"pvnet_model": config["pvnet_model"]}


def download_from_hf(
    repo_id: str,
    filename: str | list[str],
    revision: str,
    cache_dir: str | None,
    force_download: bool,
    max_retries: int = 5,
    wait_time: int = 10,
) -> str | list[str]:
    """Tries to download one or more files from HuggingFace up to max_retries times.

    Args:
        repo_id: HuggingFace repo ID
        filename: Name of the file(s) to download
        revision: Specific model revision
        cache_dir: Cache directory
        force_download: Whether to force a new download
        max_retries: Maximum number of retry attempts
        wait_time: Wait time (in seconds) before retrying

    Returns:
        The local file path of the downloaded file(s)
    """
    for attempt in range(1, max_retries + 1):
        try:
            save_dir = snapshot_download(
                repo_id=repo_id,
                allow_patterns=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
            )

            if isinstance(filename, list):
                return [f"{save_dir}/{f}" for f in filename]
            else:
                return f"{save_dir}/{filename}"
        
        except Exception as e:
            if attempt == max_retries:
                raise Exception(
                    f"Failed to download {filename} from {repo_id} after {max_retries} attempts."
                ) from e
            logging.warning(
                (
                    f"Attempt {attempt}/{max_retries} failed to download {filename} "
                    f"from {repo_id}. Retrying in {wait_time} seconds..."
                )
            )
            time.sleep(wait_time)


class HuggingfaceMixin:
    """Mixin for saving and loading model to and from huggingface"""

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        cache_dir: str | None = None,
        force_download: bool = False,
        strict: bool = True,
    ) -> "BaseModel":
        """Load Pytorch pretrained weights and return the loaded model."""

        if os.path.isdir(model_id):
            print("Loading model from local directory")
            model_file = f"{model_id}/{PYTORCH_WEIGHTS_NAME}"
            config_file = f"{model_id}/{MODEL_CONFIG_NAME}"
        else:
            print("Loading model from huggingface repo")

            model_file, config_file = download_from_hf(
                repo_id=model_id,
                filename=[PYTORCH_WEIGHTS_NAME, MODEL_CONFIG_NAME],
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                max_retries=5,
                wait_time=10,
            )

        with open(config_file, "r") as f:
            model = hydra.utils.instantiate(yaml.safe_load(f))

        state_dict = load_file(model_file)
        model.load_state_dict(state_dict, strict=strict)  # type: ignore
        model.eval()  # type: ignore

        return model
    
    @classmethod
    def get_datamodule_config(
        cls,
        model_id: str,
        revision: str,
        cache_dir: str | None = None,
        force_download: bool = False,
    ) -> str:
        """Load data config file."""
        if os.path.isdir(model_id):
            print("Loading datamodule config from local directory")
            datamodule_config_file = os.path.join(model_id, DATAMODULE_CONFIG_NAME)
        else:
            print("Loading datamodule config from huggingface repo")
            datamodule_config_file = download_from_hf(
                repo_id=model_id,
                filename=DATAMODULE_CONFIG_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                max_retries=5,
                wait_time=10,
            )

        return datamodule_config_file

    def _save_model_weights(self, save_directory: str) -> None:
        """Save weights from a Pytorch model to a local directory."""
        save_file(self.state_dict(), f"{save_directory}/{PYTORCH_WEIGHTS_NAME}")

    def save_pretrained(
        self,
        save_directory: str,
        model_config: dict,
        wandb_repo: str,
        wandb_id: str,
        card_template_path: str,
        datamodule_config_path,
        experiment_config_path: str | None = None,
        hf_repo_id: str | None = None,
        push_to_hub: bool = False,
    ) -> None:
        """Save weights in local directory or upload to huggingface hub.

        Args:
            save_directory:
                Path to directory in which the model weights and configuration will be saved.
            model_config (`dict`):
                Model configuration specified as a key/value dictionary.
            wandb_repo: Identifier of the repo on wandb.
            wandb_id: Identifier of the model on wandb.
            datamodule_config_path:
                The path to the datamodule config.
            card_template_path: Path to the HuggingFace model card template. Defaults to card in
                PVNet library if set to None.
            experiment_config_path:
                The path to the full experimental config.
            hf_repo_id:
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to
                the folder name if not provided.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the HuggingFace Hub after saving it.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model weights/files
        self._save_model_weights(save_directory)

        # Save the model config
        if isinstance(model_config, dict):
            with open(save_directory / MODEL_CONFIG_NAME, "w") as outfile:
                yaml.dump(model_config, outfile, sort_keys=False, default_flow_style=False)

        # Sanitize and save the datamodule config
        with open(datamodule_config_path) as cfg:
            datamodule_config = yaml.load(cfg, Loader=yaml.FullLoader)

        datamodule_config = santize_datamodule(datamodule_config)

        with open(save_directory / DATAMODULE_CONFIG_NAME, "w") as outfile:
            yaml.dump(datamodule_config, outfile, sort_keys=False, default_flow_style=False)
        
        # Save the full experimental config
        if experiment_config_path is not None:
            shutil.copyfile(experiment_config_path, save_directory / FULL_CONFIG_NAME)

        card = self.create_hugging_face_model_card(card_template_path, wandb_repo, wandb_id)

        (save_directory / MODEL_CARD_NAME).write_text(str(card))

        if push_to_hub:
            api = HfApi()

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=save_directory,
                repo_type="model",
                commit_message=f"Upload model - {wandb_id}",
            )

            # Print the most recent commit hash
            c = api.list_repo_commits(repo_id=hf_repo_id, repo_type="model")[0]

            message = (
                f"The latest commit is now: \n"
                f"    date: {c.created_at} \n"
                f"    commit hash: {c.commit_id}\n"
                f"    by: {c.authors}\n"
                f"    title: {c.title}\n"
            )

            print(message)

    @staticmethod
    def create_hugging_face_model_card(
        card_template_path: str,
        wandb_repo: str,
        wandb_id: str,
    ) -> ModelCard:
        """
        Creates Hugging Face model card

        Args:
            card_template_path: Path to the HuggingFace model card template
            wandb_repo: Identifier of the repo on wandb.
            wandb_id: Identifier of the model on wandb.

        Returns:
            card: ModelCard - Hugging Face model card object
        """

        # Creating and saving model card.
        card_data = ModelCardData(language="en", license="mit", library_name="pytorch")

        link = f"https://wandb.ai/{wandb_repo}/runs/{wandb_id}"
        wandb_link = f" - [{link}]({link})\n"

        # Find package versions for OCF packages
        packages_to_display = ["pvnet_summation", "ocf-data-sampler"]
        packages_and_versions = {package: version(package) for package in packages_to_display}


        package_versions_markdown = ""
        for package, v in packages_and_versions.items():
            package_versions_markdown += f" - {package}=={v}\n"

        return ModelCard.from_template(
            card_data,
            template_path=card_template_path,
            wandb_link=wandb_link,
            package_versions=package_versions_markdown,
        )


class BaseModel(torch.nn.Module, HuggingfaceMixin):
    """Abstract base class for PVNet-summation submodels"""

    def __init__(
        self,
        output_quantiles: list[float] | None,
        num_input_locations: int,
        input_quantiles: list[float] | None,
        history_minutes: int,
        forecast_minutes: int,
        interval_minutes: int,
    ):
        """Abtstract base class for PVNet-summation submodels.

        """
        super().__init__()

        if (output_quantiles is not None):
            if output_quantiles != sorted(output_quantiles):
                raise ValueError("output_quantiles should be in ascending order")
            if 0.5 not in output_quantiles:
                raise ValueError("Quantiles must include 0.5")

        self.output_quantiles = output_quantiles

        self.num_input_locations = num_input_locations
        self.input_quantiles = input_quantiles

        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.interval_minutes = interval_minutes

        # Number of timestemps for 30 minutely data
        self.history_len = history_minutes // interval_minutes
        self.forecast_len = (forecast_minutes) // interval_minutes

        # Store whether the model should use quantile regression or simply predict the mean
        self.use_quantile_regression = self.output_quantiles is not None

        # Also store the final output shape
        if self.use_quantile_regression:
            self.output_shape = (self.forecast_len, len(input_quantiles))
        else:
            self.output_shape = (self.forecast_len,)

        # Store the number of output features and that the model should predict for
        self.num_output_features = prod(self.output_shape)

        # Store the expected input shape
        if input_quantiles is None:
            self.input_shape = (self.num_input_locations, self.forecast_len)
        else:
            self.input_shape = (self.num_input_locations, self.forecast_len, len(input_quantiles))
            

    def _quantiles_to_prediction(self, y_quantiles: torch.Tensor) -> torch.Tensor:
        """Convert network prediction into a point prediction.

        Args:
            y_quantiles: Quantile prediction of network

        Returns:
            torch.Tensor: Point prediction
        """
        # y_quantiles Shape: [batch_size, seq_length, num_quantiles]
        idx = self.output_quantiles.index(0.5)
        return y_quantiles[..., idx]

    def sum_of_locations(self, x: SumTensorBatch) -> torch.Tensor:
        """Compute the sum of the location-level predictions"""
        if self.input_quantiles is None:
            y_hat = x["pvnet_outputs"]
        else:
            idx = self.input_quantiles.index(0.5)
            y_hat = x["pvnet_outputs"][..., idx]

        return (y_hat * x["relative_capacity"].unsqueeze(-1)).sum(dim=1)