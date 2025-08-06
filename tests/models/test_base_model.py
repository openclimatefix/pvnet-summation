"""Test the methods which stem from the BaseModel class"""
from pvnet_summation.models import BaseModel
import pvnet_summation.model_cards

card_path = f"{pvnet_summation.model_cards.__path__[0]}/empty_model_card_template.md"


def test_save_load_model(tmp_path, model, raw_model_kwargs):
    """Test loading and saving the model"""

    # Construct the model config
    model_config = {
        "_target_": "pvnet_summation.models.DenseModel",
        "output_quantiles": None,
        **raw_model_kwargs,
    }

    # Save the model
    model_output_dir = f"{tmp_path}/saved_model"
    model.save_pretrained(
        save_directory=model_output_dir,
        model_config=model_config,
        wandb_repo="test",
        wandb_id="abc",
        card_template_path=card_path,
        push_to_hub=False,
    )

    # Load the model
    _ = BaseModel.from_pretrained(model_id=model_output_dir, revision=None)