from datasets import load_dataset
import pandas as pd

def formatting_prompts_func(example):
    question = example["Question"]
    response = example["Response"]
    
    # Gemma instruction format
    return {
        "text": f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    }

print("Loading and formatting dataset...")
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train", streaming=True)

# Take a small sample for initial training
sample_size = 1000
dataset_sample = dataset.take(sample_size)

processed_data = [formatting_prompts_func(example) for example in dataset_sample]

# Convert to a Hugging Face Dataset object
from datasets import Dataset
processed_dataset = Dataset.from_pandas(pd.DataFrame(data=processed_data))

print(f"Saving {len(processed_dataset)} processed examples to disk...")
processed_dataset.save_to_disk("processed_dataset_medical_sft")

print("Dataset preparation complete.")

