from datasets import load_dataset, concatenate_datasets


culture_bank_dataset = load_dataset("SALT-NLP/CultureBank", split="reddit")
culture_bank_dataset_2 = load_dataset("SALT-NLP/CultureBank", split="tiktok")

culture_bank_dataset = concatenate_datasets([culture_bank_dataset, culture_bank_dataset_2])

def lowercase_tf(example):
    example["output"] = example["output"].replace("True", "true").replace("False", "false")
    return example

culture_bank_dataset = (
    culture_bank_dataset.filter(lambda x: x["cultural group"] != "American")
    .rename_column("eval_question", "instruction")
    .rename_column("eval_persona", "input")
    .rename_column("eval_whole_desc", "output")
    .select_columns(["instruction", "input", "output"])
    .map(lowercase_tf)
)


alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")

data = concatenate_datasets([culture_bank_dataset, alpaca_dataset]).shuffle(seed=42)



data.to_json("culture_shuffle.jsonl", lines=True)
