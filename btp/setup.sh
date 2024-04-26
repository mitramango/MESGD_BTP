#!/bin/bash


conda create -n mesgd_new python=3.8.18


source activate mesgd_new


pip install -r req.txt


conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
