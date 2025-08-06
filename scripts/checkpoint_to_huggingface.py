"""Command line tool to push locally save model checkpoints to huggingface

To use this script, you will need to write a custom model card. You can copy and fill out
`pvnet/model_cards/empty_model_card_template.md` to get you started.

These model cards should not be added to and version controlled in the repo since they are specific
to each user.

Then run using:

```
python checkpoint_to_huggingface.py "path/to/model/checkpoints" \
    --huggingface-repo="openclimatefix/pvnet_summation" \
    --wandb-repo="openclimatefix/pvnet_uk_summation" \
    --card-template-path="pvnet_summation/models/model_cards/my_custom_model_card.md" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
```
"""

import tempfile

import typer
import wandb

from pvnet_summation.load_model import get_model_from_checkpoints

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def push_to_huggingface(
    checkpoint_dir_paths: str = typer.Argument(...,),
    huggingface_repo: str = typer.Option(..., "--huggingface-repo"),
    wandb_repo: str = typer.Option(..., "--wandb-repo"),
    card_template_path: str = typer.Option(..., "--card-template-path"),
    wandb_id: str = typer.Option(None, "--wandb-id"),
    val_best: bool = typer.Option(True),
    local_path: str | None = typer.Option(None, "--local-path"),
    push_to_hub: bool = typer.Option(True),
):
    """Push a local model to a huggingface model repo

    Args:
        checkpoint_dir_path: Path of the checkpoint directory
        huggingface_repo: Name of the HuggingFace repo to push the model to
        wandb_repo: Name of the wandb repo which has training logs
        card_template_path: Path to the model card template.
        wandb_id: The wandb ID - if not filled given, we try to derive this from 
            `checkpoint_dir_paths` 
        val_best: Use best model according to val loss, else last saved model
        local_path: Where to save the local copy of the model
        push_to_hub: Whether to push the model to the hub or just create local version.
    """

    assert push_to_hub or local_path is not None

    # Check that the wandb-IDs are correct
    all_wandb_ids = [run.id for run in wandb.Api().runs(path=wandb_repo)]

    # If the ID is not supplied try and pull it from the checkpoint dir name
    if wandb_id is None:
        dirname = checkpoint_dir_path.split("/")[-1]
        if dirname in all_wandb_ids:
            wandb_id  = dirname
        else:
            raise Exception(f"Could not find wand run for {path} within {wandb_repo}")
    
    # Else if it is provided check that it exists
    else:
        if wandb_id not in all_wandb_ids:
            raise Exception(f"Could not find wand run for {wandb_id} within {wandb_repo}")

    (
        model, 
        model_config, 
        data_config_path, 
        datamodule_config_path, 
        experiment_config_path,
    ) = get_model_from_checkpoints(checkpoint_dir_paths, val_best)

    # Push to hub
    if local_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        model_output_dir = temp_dir.name
    else:
        model_output_dir = local_path

    model.save_pretrained(
        save_directory=model_output_dir,
        model_config=model_config,
        data_config_path=data_config_path,
        datamodule_config_path=datamodule_config_path,
        experiment_config_path=experiment_config_path,
        wandb_repo=wandb_repo,
        wandb_id=wandb_id,
        card_template_path=card_template_path,
        push_to_hub=push_to_hub,
        hf_repo_id=huggingface_repo if push_to_hub else None,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    app()