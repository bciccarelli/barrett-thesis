#!/bin/bash

git lfs install
git clone https://bkciccar:xxx@huggingface.co/bkciccar/llama-3.1-8b-instruct-culture-fft
cd llama-3.1-8b-instruct-culture-fft/
git lfs track "tokenizer.json"
huggingface-cli lfs-enable-largefiles .
cp /scratch/bkciccar/outputs/llama-3.1-8b-fft-1/checkpoint-1288/* .
git add .
git commit -m "Upload finetuned model checkpoint"
git push

cd ..

git lfs install
git clone https://bkciccar:xxx@huggingface.co/bkciccar/mistral-7b-instruct-culture-fft
cd mistral-7b-instruct-culture-fft/
git lfs track "tokenizer.json"
huggingface-cli lfs-enable-largefiles .
cp /scratch/bkciccar/outputs/mistral-7b-fft-1/checkpoint-1395/* .
git add .
git commit -m "Upload finetuned model checkpoint"
git push

cd ..

git lfs install
git clone https://bkciccar:xxx@huggingface.co/bkciccar/qwen2-7b-instruct-culture-fft
cd qwen2-7b-instruct-culture-fft/
git lfs track "tokenizer.json"
huggingface-cli lfs-enable-largefiles .
cp /scratch/bkciccar/outputs/qwen2-7b-fft-1/checkpoint-1284/* .
git add .
git commit -m "Upload finetuned model checkpoint"
git push

cd ..