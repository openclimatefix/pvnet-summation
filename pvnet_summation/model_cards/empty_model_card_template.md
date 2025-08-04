---
{{ card_data }}
---
<!--
Do not remove elements like the above surrounded by two curly braces and do not add any more of them. These entries are required by the library and are automaticall infilled when the model is uploaded to huggingface
-->

<!-- Title - e.g. PVNet-summation -->
# TEMPLATE

<!-- Provide a longer summary of what this model is/does. -->
## Model Description

<!-- e.g.
This model uses the output predictions of PVNet to predict the sum from predictions of the parts
-->

- **Developed by:** openclimatefix
- **Language(s) (NLP):** en
- **License:** mit

# Training Details

## Data

<!-- eg.
The model is trained on data from 2019-2022 and validated on data from 2022-2023. It uses the
output predictions from PVNet - see the PVNet model for its inputs

-->

<!-- The preprocessing section is not strictly nessessary but perhaps nice to have -->
### Preprocessing

<!-- eg.
Data is prepared with the `ocf_data_sampler/torch_datasets/datasets/pvnet_uk` Dataset [2].
-->

## Results

<!-- Do not remove the lines below -->
The training logs for this model commit can be found here:
{{ wandb_links }}

<!-- The hardware section is also just nice to have -->
### Hardware
<!-- e.g.
Trained on a single NVIDIA Tesla T4
-->

<!-- Do not remove the section below -->
### Software

This model was trained using the following Open Climate Fix packages:

- [1] https://github.com/openclimatefix/pvnet-summation
- [2] https://github.com/openclimatefix/ocf-data-sampler

<!-- Especially do not change the two lines below -->
The versions of these packages can be found below:
{{ package_versions }}
