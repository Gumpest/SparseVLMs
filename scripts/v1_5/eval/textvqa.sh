#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/${MODEL} \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/textvqa/train_images \
    --answers-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/textvqa/answers/${MODEL}.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1 

python3 -m llava.eval.eval_textvqa \
    --annotation-file /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/textvqa/answers/${MODEL}.jsonl