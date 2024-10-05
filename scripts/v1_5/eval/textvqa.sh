#!/bin/bash
MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --scale 0.8 \
    --bias 0

python3 -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl
