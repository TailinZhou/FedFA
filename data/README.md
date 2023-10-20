# Data

MNIST & CIFAR-10 & CIFAR-100 datasets will be downloaded automatically by the torchvision package.

FEMNIST datasets will be downloaded and processed automatically by the code file for 'utils/femnist_dataset'.

sorry for this inconvenience, mixed_digit dataset needs to be downloaded according to the following instruction (sorry for this insconvenience):
''''
This experiment seting is following FedBN (this paper url={https://openreview.net/pdf?id=6YEQUn0QICG}).
Code souce of data process: https://github.com/med-air/FedBN

Before running this file, due to the requirement of the maximum file size of Supplementary Material,
please download the pre-processed datasets from the following url:
    https://drive.google.com/uc?export=download&id=1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf
    
and unzip it under 'data/mixed_digit_dataset' directory,
then you can start following experiments on mixed-digit dataset.
''''

