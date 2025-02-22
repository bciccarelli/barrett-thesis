from datasets import load_dataset, Value

dataset = load_dataset("kellycyy/CulturalBench", split="test")

dataset = dataset.cast_column("answer", Value("string"))

dataset.to_json("culturalbench-hard-preprocessed.jsonl", lines=True)
