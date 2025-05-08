# PVNet summation
[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#overview-of-ocfs-nowcasting-repositories)

This project is used for training a model to sum the GSP predictions of [PVNet](https://github.com/openclimatefix/PVNet) into a national estimate.

Using the summation model to sum the GSP predictions rather than doing a simple sum increases the accuracy of the national predictions and can be configured to produce estimates of the uncertainty range of the national estimate. See the [PVNet](https://github.com/openclimatefix/PVNet) repo for more details and our paper.


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
[PVNet](https://github.com/openclimatefix/PVNet) and have a trained PVNet model already available either locally or pushed to HuggingFace.

Before running any code, copy the example configuration to a
configs directory:

```
cp -r configs.example configs
```

You will be making local amendments to these configs.

### Datasets

The datasets required are the same as documented in
[PVNet](https://github.com/openclimatefix/PVNet). The only addition is that you will need PVLive
data for the national sum i.e. GSP ID 0.


## Generating pre-made concurrent batches of data for PVNet

It is required that you preprepare batches using the `save_concurrent_batches.py` script from
PVNet. This saves the batches as required by the PVNet model to make predictions for all GSPs for
a single forecast init time. Seen the PVNet package for more details on this.


### Set up and config example for batch creation


The concurrent batches created in the step above will be augmented with a few additional pieces of
data required for the summation model. Within your copy of `PVNet_summation/configs` make sure you
have replaced all of the items marked with `PLACEHOLDER`

### Training PVNet_summation

How PVNet_summation is run is determined by the extensive configuration in the config files. The
configs stored in `PVNet/configs.example` should work with batches created using the steps and
batch creation config mentioned above.

Make sure to update the following config files before training your model:

1. In `configs/datamodule/default.yaml`:
    - update `batch_dir` to point to the directory you stored your concurrent batches in during
      batch creation.
    - update `gsp_zarr_path` to point to the PVLive data containing the national estimate
2. In `configs/model/default.yaml`:
    - update the PVNet model for which you are training a summation model for. A new summation model
      should be trained for each PVNet model
    - update the hyperparameters and structure of the summation model
3. In `configs/trainer/default.yaml`:
    - set `accelerator: 0` if running on a system without a supported GPU
4. In `configs.config.yaml`:
    - It is recommended that you set `presave_pvnet_outputs` to `True`. This means that the
      concurrent batches that you create will only be run through the PVNet model once before
      training and their outputs saved, rather than being run on the fly on each batch throughout
      training. This can speed up training significantly.


Assuming you have updated the configs, you should now be able to run:

```
python run.py
```

This will then use the pretrained PVNet model to run inference on the concurrent batches, the outputs from this inference will then be used as the training data for the summation model alongside the national PVLive data (GSP ID 0).

## Testing

You can use `python -m pytest tests` to run tests
