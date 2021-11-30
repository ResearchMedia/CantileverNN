**Status:** Archive (code is provided as-is, no updates expected)

# CantileverNN

#### [ [Paper] ](https://doi.org/10.1098/rsif.2021.0571) [ [Video] ](TBD)

This repository provides the necessary files to reproduce the training results published in our paper: "Visual design intuition: Predicting dynamic properties of beams from raw cross-section images".

![3D Twisted Beam with Eigenvalues from FEA, analytical solution, and Neural Network Prediction](/figures/3DTwistedBeamExplained_V2.png "3D Twisted Beam")

## Neural Network Configuration
![Neural Network Architecture Graph](/figures/ConvNetExtended.png "Convolutional Neural Network Graph")

## Installation

Tested platform: Ubuntu 18.04, Python 3.7, PyTorch <= 1.3.0, cudatoolkit 9.0, 

- Install the packages listed in PyTorchNN.yml or simply create a conda environment using the PyTorchNN.yml as a template.
- Clone the repository:
    ```
    git clone https://github.com/ResearchMedia/CantileverNN/
    ```
## Downloading corresponding datasets
Download our corresponding datasets from Mendeley Data: https://doi.org/10.17632/y3m8xm6kfk


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
Coming Soon

## Guide to interfaces
Coming Soon

## Training models
Coming Soon

## Citation
Wyder Philippe M. and Lipson Hod. 2021 Visual design intuition: predicting dynamic properties of beams from raw cross-section images. J. R. Soc. Interface.182021057120210571
[https://doi.org/10.1098/rsif.2021.0571](https://doi.org/10.1098/rsif.2021.0571)

Please cite using the following BibTeX entry:
```
@article{VisualDesignIntuition2021,
  author = {Philippe M. Wyder, Hod Lipson},
  title = {Visual design intuition: predicting dynamic properties of beams from raw cross-section images},
  journal = {J. R. Soc. Interface},
  year = {2021},
  doi = {10.1098/rsif.2021.0571}
}
```
