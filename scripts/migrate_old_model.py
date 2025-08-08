"""Script to migrate old pvnet-summation models (v0.3.7) which are hosted on huggingface to current 
"""

import datetime
import os
import tempfile
from importlib.metadata import version
import tempfile

import torch
import yaml
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, file_exists
from safetensors.torch import save_file

from pvnet_summation.models import BaseModel
from pvnet_summation.utils import (
    DATAMODULE_CONFIG_NAME,
    MODEL_CARD_NAME,
    MODEL_CONFIG_NAME,
    PYTORCH_WEIGHTS_NAME,
)

# ------------------------------------------
# USER SETTINGS

# The huggingface commit of the model you want to update
repo_id: str = "openclimatefix/pvnet_v2_summation"
revision: str = "175a71206cf89a2d8fcd180cfa60d132590f12cb"

# If set to a string, this allows you to change the commit of the PVNet model which the summation 
# model goes with. Else the commit hash is taken from what is already stored on huggingface.
# This is useful if both PVNet and PVNet-summation are being migrated simultaneously
pvnet_revision: str | None = None

# The local directory which will be downloaded to
# If set to None a temporary directory will be used
local_dir: str | None = None

# Whether to upload the migrated model back to the huggingface
upload: bool = True

# ------------------------------------------
# SETUP

if local_dir is None:
    temp_dir = tempfile.TemporaryDirectory()
    save_dir = temp_dir.name

else:
    os.makedirs(local_dir, exist_ok=False)
    save_dir = local_dir

# Set up huggingface API
api = HfApi()

# Download the model repo
_ = api.snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=save_dir,
    force_download=True,
)

# ------------------------------------------
# MIGRATION STEPS

# Modify the model config
with open(f"{save_dir}/{MODEL_CONFIG_NAME}") as cfg:
    model_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Get the PVNet model it was trained on
pvnet_model_id = model_config.pop("model_name")
if pvnet_revision is None:
    pvnet_revision = model_config["model_version"]
del model_config["model_version"]


with tempfile.TemporaryDirectory() as pvnet_dir:

    # Download the model repo
    _ = api.snapshot_download(
        repo_id=pvnet_model_id,
        revision=pvnet_revision,
        local_dir=str(pvnet_dir),
        force_download=True,
    )

    with open(f"{pvnet_dir}/{MODEL_CONFIG_NAME}") as cfg:
        pvnet_model_config = yaml.load(cfg, Loader=yaml.FullLoader)

# Get rid of the optimiser - we don't store this anymore
del model_config["optimizer"]

# Rename the top level model
if model_config["_target_"]=="pvnet_summation.models.flat_model.FlatModel":
    model_config["_target_"] = "pvnet_summation.models.dense_model.DenseModel"
else:
    raise Exception("Unknown model: " + model_config["_target_"])

# Models which used this setting are not supported any more
if model_config["relative_scale_pvnet_outputs"]:
    raise Exception("Models with `relative_scale_pvnet_outputs=True` are no longer supported")
else:
    del model_config["relative_scale_pvnet_outputs"]


model_config["num_input_locations"] = model_config.pop("num_locations")

# Re-find the model components in the new PVNet package structure
model_config["output_network"]["_target_"] = (
    model_config["output_network"]["_target_"] 
        .replace("multimodal", "late_fusion")
        .replace("ResFCNet2", "ResFCNet")
)

# Add entries from the PVNet model which are now required in the summation model
model_config["history_minutes"] = pvnet_model_config["history_minutes"]
model_config["forecast_minutes"] = pvnet_model_config["forecast_minutes"]
model_config["interval_minutes"] = pvnet_model_config.get("interval_minutes", 30)
model_config["input_quantiles"] = pvnet_model_config["output_quantiles"]

# Save the model config
with open(f"{save_dir}/{MODEL_CONFIG_NAME}", "w") as f:
    yaml.dump(model_config, f, sort_keys=False, default_flow_style=False)

# Create a datamodule
with open(f"{save_dir}/{DATAMODULE_CONFIG_NAME}", "w") as f:
    datamodule = {"pvnet_model": {"model_id": pvnet_model_id, "revision": pvnet_revision}}
    yaml.dump(datamodule, f, sort_keys=False, default_flow_style=False)

# Resave the model weights as safetensors and remove the PVNet weights which we no longer need
state_dict = torch.load(f"{save_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)
new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("pvnet_model")}
save_file(new_state_dict, f"{save_dir}/{PYTORCH_WEIGHTS_NAME}")
os.remove(f"{save_dir}/pytorch_model.bin")

# Add a note to the model card to say the model has been migrated
with open(f"{save_dir}/{MODEL_CARD_NAME}", "a") as f:
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    summation_version = version("pvnet_summation")
    f.write(
        f"\n\n---\n**Migration Note**: This model was migrated on {current_date} "
        f"to pvnet-summation version {summation_version}\n"
    )

# ------------------------------------------
# CHECKS

# Check the model can be loaded
model = BaseModel.from_pretrained(model_id=save_dir, revision=None)

print("Model checkpoint successfully migrated")

# ------------------------------------------
# UPLOAD TO HUGGINGFACE

if upload:
    print("Uploading migrated model to huggingface")

    operations = []
    for file in [MODEL_CARD_NAME, MODEL_CONFIG_NAME, PYTORCH_WEIGHTS_NAME, DATAMODULE_CONFIG_NAME]:
        # Stage modified files for upload
        operations.append(
            CommitOperationAdd(
                path_in_repo=file, # Name of the file in the repo
                path_or_fileobj=f"{save_dir}/{file}", # Local path to the file
            ),
        )
    
    # Remove old pytorch weights file if it exists in the most recent commit
    if file_exists(repo_id, "pytorch_model.bin"):
        operations.append(
            CommitOperationDelete(path_in_repo="pytorch_model.bin")
        )

    commit_info = api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Migrate model to pvnet-summation version {summation_version}",
    )

    # Print the most recent commit hash
    c = api.list_repo_commits(repo_id=repo_id, repo_type="model")[0]

    print(
        f"\nThe latest commit is now: \n"
        f"    date: {c.created_at} \n"
        f"    commit hash: {c.commit_id}\n"
        f"    by: {c.authors}\n"
        f"    title: {c.title}\n"
    )
