#!/bin/bash

# === Configurable parameters ===
IMAGE_NAME="roeibenzion/llava:new"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="llava-fga-pretrain-$TIMESTAMP"
NUM_GPUS=1
PROJECT_NAME="aicenter"  # uncommented so it's usable
HF_TOKEN="hf"           # Set your Hugging Face token here
WANDB_API_KEY="d"      # Set your WandB API key here

runai submit $JOB_NAME \
  --image $IMAGE_NAME \
  --gpu $NUM_GPUS \
  --large-shm \
  --node-type h100 \
  --name $JOB_NAME \
  --project $PROJECT_NAME \
  --backoff-limit 0 \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY
