#!/bin/bash

# download model weights
mkdir weights
wget -O weights/checkpoint_95.pth https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=0

# create env and activate
conda create -n score-POCS_ python=3.8
conda activate score-POCS_

# install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
