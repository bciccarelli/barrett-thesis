lm_eval --model hf \
    --model_args pretrained=/scratch/bkciccar/outputs/llama-3.1-8b-fft/checkpoint-486 \
    --tasks culturalbench_hard \
    --device cuda:0 \
    --batch_size auto