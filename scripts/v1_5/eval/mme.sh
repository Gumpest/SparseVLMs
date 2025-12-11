#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/$MODEL \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/MME/MME_Benchmark_release_version \
    --answers-file  /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/MME

python3 convert_answer_to_mme.py --experiment $MODEL

cd eval_tool

python3 calculation.py --results_dir answers/$MODEL