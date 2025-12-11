#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.eval.model_vqa_loader \
        --model-path /mnt/bn/bes-mllm-shared/SparseVLM/llava-v1.5-7b \
        --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python3 scripts/convert_seed_for_submission.py \
    --annotation-file /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl

