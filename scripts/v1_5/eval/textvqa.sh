#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /content/LLaVA_converter/checkpoints/checkpoint-500 \
    --model-base lmsys/vicuna-7b-v1.5
    --question-file ./playground/data/eval/textvqa/test.json \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --q_limit 300

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl
