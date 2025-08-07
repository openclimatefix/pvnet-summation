# PVNet summation
[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#overview-of-ocfs-nowcasting-repositories)

This project is used for training a model to sum the GSP predictions of [PVNet](https://github.com/openclimatefix/pvnet) into a national estimate.

Using the summation model to sum the GSP predictions rather than doing a simple sum increases the accuracy of the national predictions and can be configured to produce estimates of the uncertainty range of the national estimate. See the [PVNet](https://github.com/openclimatefix/pvnet) repo for more details and our paper.


## Setup / Installation

```bash
git clone https://github.com/openclimatefix/PVNet_summation
cd PVNet_summation
pip install .
```

### Additional development dependencies

```bash
pip install ".[dev]"
```

## Getting started with running PVNet summation

In order to run PVNet summation, we assume that you are already set up with
[PVNet](https://github.com/openclimatefix/pvnet) and have a trained PVNet model already available either locally or pushed to HuggingFace.

Before running any code, copy the example configuration to a configs directory:

```
cp -r configs.example configs
```

You will be making local amendments to these configs.

### Datasets

The datasets required are the same as documented in
[PVNet](https://github.com/openclimatefix/pvnet). The only addition is that you will need PVLive
data for the national sum i.e. GSP ID 0.


### Training PVNet_summation

How PVNet_summation is run is determined by the extensive configuration in the config files. The
configs stored in `configs.example`.

Make sure to update the following config files before training your model:


1. At the very start of training we loop over all of the input samples and make predictions for them using PVNet. These predictions are saved to disk and will be loaded in the training loop for more efficient training. In `configs/config.yaml` update `sample_save_dir` to set where the predictions will be saved to.

2. In `configs/datamodule/default.yaml`:
  - Update `pvnet_model.model_id` and `pvnet_model.revision` to point to the Huggingface commit or local directory where the exported PVNet model is.
  - Update `configuration` to point to a data configuration compatible with the PVNet model whose outputs will be fed into the summation model.
  - Set `train_period` and `val_period` to control the time ranges of the train and val period
  - Optionally set `max_num_train_samples` and `max_num_val_samples` to limit the number of possible train and validation example which will be used.

3. In `configs/model/default.yaml`:
    - Update the hyperparameters and structure of the summation model
4. In `configs/trainer/default.yaml`:
    - Set `accelerator: 0` if running on a system without a supported GPU


Assuming you have updated the configs, you should now be able to run:

```
python run.py
```


## Testing

You can use `python -m pytest tests` to run tests
