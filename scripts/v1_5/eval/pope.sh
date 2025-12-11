#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/${MODEL} \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/pope/val2014 \
    --answers-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/pope/answers/${MODEL}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python3 llava/eval/eval_pope.py \
    --annotation-dir /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/pope/coco \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/pope/answers/${MODEL}.jsonl