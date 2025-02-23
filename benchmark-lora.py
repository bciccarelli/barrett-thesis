import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

model_name = sys.argv[1]

if model_name == "llama":
    base_model_name = "NousResearch/Meta-Llama-3.1-8B"
    lora_weights_path = "outputs/llama-3.1-8b-lora"

dataset_name = "kellycyy/CulturalBench"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")

model = PeftModel.from_pretrained(base_model, lora_weights_path)
model.eval()


dataset = load_dataset(dataset_name, split="test")


true_id = tokenizer("true", add_special_tokens=False).input_ids[0]
false_id = tokenizer("false", add_special_tokens=False).input_ids[0]


def construct_prompt(sample):
    prompt = (
        f"Question: {sample['prompt_question']}\n"
        f"Option: {sample['prompt_option']}\n"
        "Answer (respond with ONLY 'true' or 'false'):"
    )
    return prompt

def predict_from_logits(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(logits, dim=-1)
    true_prob = probabilities[true_id].item()
    false_prob = probabilities[false_id].item()
    prediction = "true" if true_prob > false_prob else "false"
    return prediction, true_prob, false_prob

correct = 0
total = 0

for sample in dataset.select(range(100)):
    prompt = construct_prompt(sample)
    prediction, prob_true, prob_false = predict_from_logits(prompt)
    gold = str(sample['answer']).strip().lower()
    
    if prediction == gold:
        correct += 1
    total += 1

    print(f"Prediction: {prediction} | Gold: {gold}")

accuracy = correct / total if total > 0 else 0
print(f"\nAccuracy over {total} examples: {accuracy * 100:.2f}%")
