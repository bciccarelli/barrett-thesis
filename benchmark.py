import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NousResearch/Meta-Llama-3.1-8B"
dataset_name = "kellycyy/CulturalBench"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.eval()

gen_kwargs = {
    "max_new_tokens": 10,
    "do_sample": False,
    "temperature": 0.0,
    "eos_token_id": tokenizer.eos_token_id,
}

dataset = load_dataset(dataset_name, split="test")

def construct_prompt(sample):
    prompt = (
        f"Question: {sample['prompt_question']}\n"
        f"Option: {sample['prompt_option']}\n"
        "Answer (true/false):"
    )
    return prompt

def parse_prediction(generated_text):
    text = generated_text.lower()
    match = re.search(r'\b(true|false)\b', text)
    if match:
        return match.group(1)
    return None

correct = 0
total = 0

for sample in dataset.select(range(1)):
    prompt = construct_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    print(outputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = parse_prediction(generated_text)
    
    gold = str(sample['answer']).strip().lower()
    if prediction is not None and prediction == gold:
        correct += 1
    total += 1
    
    print(f"Prompt:\n{prompt}")
    print(f"Generated: {generated_text}")
    print(f"Prediction: {prediction} | Gold: {gold}\n{'-'*40}")

accuracy = correct / total if total > 0 else 0
print(f"\nAccuracy over {total} examples: {accuracy * 100:.2f}%")
