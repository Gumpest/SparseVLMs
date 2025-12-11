#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_science \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/llava-v1.5-7b \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/images/test \
    --answers-file  /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_science_qa.py \
    --base-dir /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa \
    --result-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --output-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl \
    --output-result /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/scienceqa/answers/llava-v1.5-7b_result.json
