#!/bin/bash

# download model weights
mkdir weights
wget -O weights/checkpoint_95.pth https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=0

# create env and activate
conda create -n score-POCS_x python=3.8
conda activate score-POCS_x

# install dependencies
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt