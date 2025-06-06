#!/bin/bash
# This script sets up the environment for running LLaVA converter on RunPod.

mkdir content
cd ./content
git clone https://github.com/roeibenzion/LLaVA_converter.git
cd ./LLaVA_converter
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda env create -f environment.yml
conda activate llava_yaml
huggingface-cli login 
wandb login d12da608f35a7fa002abd3cfb6958656dcc85425
python3 download.py
python3 unzip.py
rm -rf ./playground/data/LLaVA-Pretrain/images.zip
chmod +x ./scripts/v1_5/fga_pretrain.sh
./scripts/v1_5/fga_pretrain.sh

