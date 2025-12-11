#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/llava-v1.5-7b \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/mm-vet/images \
    --answers-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mm-vet/results/llava-v1.5-7b.json

