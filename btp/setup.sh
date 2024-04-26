#!/bin/bash


conda create -n mesgd5 python=3.8.18

source activate mesgd5

pip install git+https://github.com/huggingface/transformers

pip install -r req.txt

pip install git+https://github.com/huggingface/transformers

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
