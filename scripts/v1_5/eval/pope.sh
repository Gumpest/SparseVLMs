#!/bin/bash

CKPT=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$CKPT \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --scale 9 \
    --bias 6

python3 llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
