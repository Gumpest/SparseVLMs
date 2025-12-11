#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
MODEL=llava-v1.5-7b

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/data"

CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python3 -m llava.eval.model_vqa_loader \
    --model-path /mnt/bn/bes-mllm-shared/SparseVLM/${MODEL} \
    --question-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/$SPLIT.jsonl \
    --image-folder /mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval/gqa/data/images \
    --answers-file /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/${MODEL}.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1 &
wait

output_file=/mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/${MODEL}.jsonl  >> "$output_file"
done

python3 scripts/convert_gqa_for_eval.py --src $output_file --dst /mnt/bn/bes-mllm-shared/SparseVLM/playground/data/eval/gqa/data/testdev_balanced_predictions.json

cd $GQADIR
python3 eval/eval.py --tier testdev_balanced
