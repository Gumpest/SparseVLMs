#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SPLIT="mmbench_dev_20230712"
CKPT=llava-v1.5-7b

python3 -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/llava-v1.5-7b \
    --question-file /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python3 scripts/convert_mmbench_for_submission.py \
    --annotation-file /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
