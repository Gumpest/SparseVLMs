#!/bin/bash
MODEL="llava-v1.5-7b"

python -m llava.eval.model_vqa \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/$MODEL \
    --question-file ./playground/data/eval/llava-bench-coco/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-coco/images \
    --answers-file ./playground/data/eval/llava-bench-coco/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-coco/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-coco/questions.jsonl \
    --context playground/data/eval/llava-bench-coco/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-coco/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-coco/answers/$MODEL.jsonl \
    --output \
        playground/data/eval/llava-bench-coco/reviews/$MODEL.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-coco/reviews/$MODEL.jsonl
