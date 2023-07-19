---
{{ card_data }}
---






# PVNet_summation

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
This model class sums the output of the PVNet model's GSP level predictions to make a national forecast of UK PV output. More information can be found in the model repo [1], the PVNet model repo [2], and experimental notes in [this google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing).

- **Developed by:** openclimatefix
- **Model type:** Fusion model
- **Language(s) (NLP):** en
- **License:** mit


# Training Details

## Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model is trained on data from 2017-2020 and validated on data from 2021. See experimental notes in the [the google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing) for more details.


### Preprocessing

Data is prepared with the `ocf_datapipes.training.pvnet` datapipe [3].


## Results

The training logs for the current model can be found [here on wandb](https://wandb.ai/openclimatefix/pvnet_summation/runs/{{ wandb_model_code }}).

The training logs for all model runs of PVNet_summation can be found [here](https://wandb.ai/openclimatefix/pvnet_summation).

Some experimental notes can be found at in [the google doc](https://docs.google.com/document/d/1fbkfkBzp16WbnCg7RDuRDvgzInA6XQu3xh4NCjV-WDA/edit?usp=sharing)


### Hardware

Trained on a single NVIDIA Tesla T4

### Software

- [1] https://github.com/openclimatefix/PVNet_summation
- [2] https://github.com/openclimatefix/PVNet
- [3] https://github.com/openclimatefix/ocf_datapipes
