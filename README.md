**Status:** Archive (code is provided as-is, no updates expected)

# CantileverNN

#### [ [Paper] ](URL_TBD.com) [ [Demo] ](URL_TBD.com)

Training Convolutional Neural Networks to predict static and dynamic behaviors of cantilever beams from cross section images.

This repository provides the necessary files to reproduce the training results published in our paper: "Visual design intuition: Predicting dynamic properties of beams from raw cross section images".

![](https://avatars1.githubusercontent.com/u/11238785?s=60&v=4)

## Installation

Tested platform: Ubuntu 18.04, Python 3.7, PyTorch <= 1.3.0

- Install X and its dependencies, including Y.
- Clone the repo:
    ```
    git clone <PATH TO REPO>
    ```
## Downloading corresponding datasets
Download our corresponding datasets from Mendeley Data <REPO LINK>



## Training configuration file template
Create a training configuration file my_config.json of the format described below:

```
{
  "num_epochs": 20,
  "batch_size": 100, 
  "lbl_selection": "npf1_SUM",
  "learning_rate": 0.0001,
  "lr_isadaptive": false,
  "lr_decreasefactor": 0.5,
  "normalization_range": [
    1,
    100
  ],
  "logmask": [],
  "norm_mode": "pass_through",
  "dataset_path": "<Path_TO_DATASET>/TA50_DS/",
  "img_name": "img_agrayscale_128.jpg",
  "dataset_split": [64, 16, 20],
  "NeuralNetwork": "ConvNetExtended"
}
```
Note: lr_isadaptive should be set to "false" and norm_mode should be set to "pass_through", thereby ignoring the lr_decreasefactor, normalization_range, and logmask parameters. These parameters were included for future work.

## Visualizations

## Guide to interfaces

## Training models

## Citation

Please cite using the following BibTeX entry:
```
@article{VisualDesignIntuition2021,
  author = {Philippe M. Wyder, Hod Lipson},
  title = {Visual design intuition: Predicting dynamic properties of beams from raw cross-section images},
  journal = {Journal of the Royal Society Interface},
  year = {2021},
  note = {<URL_TBD>},
  doi = {<DOI_TBD>}
}
```
