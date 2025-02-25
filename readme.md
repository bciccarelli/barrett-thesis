# Barrett Thesis Project

## LLama 3.1 8B
### Base model

|      Tasks       |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  |0.2584|±  |0.0062|

### Finetuned
|      Tasks       |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  |0.3959|±  | 0.007|

## Qwen2 7B
### Base model
|      Tasks       |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|------------------|------:|------|-----:|------|---|----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  | 0.23|±  | 0.006|

### Finetuned
|      Tasks       |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  |0.2624|±  |0.0063|


## Mistral 7B v0.1
### Base model
|      Tasks       |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|------------------|------:|------|-----:|------|---|----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  |0.252|±  |0.0062|

### Finetuned
|      Tasks       |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|-----:|------|---|-----:|---|-----:|
|culturalbench_hard|      0|none  |     0|acc   |↑  |0.4806|±  |0.0071|



## Axolotl Setup:

```bash
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]
axolotl train llama-8b-test.yml
```
## Env variables:

API Keys: 
`WANDB_API_KEY`
`HF_TOKEN`

## Benchmarking
### Llama
`lm_eval --model hf --model_args pretrained=NousResearch/Meta-Llama-3.1-8B --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`
`lm_eval --model hf --model_args pretrained=/scratch/bkciccar/outputs/llama-3.1-8b-fft-1/checkpoint-1288 --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`

### Qwen
`lm_eval --model hf --model_args pretrained=Qwen/Qwen2-7B-Instruct --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`
`lm_eval --model hf --model_args pretrained=/scratch/bkciccar/outputs/qwen2-7b-fft-1/checkpoint-1215 --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`

### Mistral
`lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.1 --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`
`lm_eval --model hf --model_args pretrained=/scratch/bkciccar/outputs/mistral-7b-fft-1/checkpoint-1395 --tasks culturalbench_hard --device cuda:0 --batch_size auto --include_path ./lm_eval/tasks/culturalbench`