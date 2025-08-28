# Finetune Gemma-3-270m

This repository contains code and instructions for finetuning the `gemma-3-270m` model on a medical dataset. The finetuned model is available on [Hugging Face](hashvibe007/gemma3-270m-med-reasoning).

## Datasets

- FreedomIntelligence/medical-o1-reasoning-SFT

## Usage

1. Clone the repository.
2. uv init
3. uv add -r requirements.txt
4. run finetune.ipynb

## Model

- **Base model:** gemma-3-270m
- **Finetuned for:** FreedomIntelligence/medical-o1-reasoning-SFT"

## Training:
- Used unsloth for faster training.
- Used hugginface datasets library to load data
- Converted dataset into Gemma expected chat format
```
# We now use convert_to_chatml to try converting datasets to the correct format for finetuning purposes!
def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "user", "content": example["Question"]},
            {"role": "assistant", "content": example["Complex_CoT"] + "\n" + example["Response"]}
        ]
    }


dataset = dataset.map(
    convert_to_chatml
)
```
- trl for training
```
# train the model
from trl import SFTTrainer
```
- Saved model after training
```
model.save_pretrained("gemma3-270m-med")  # Local saving
tokenizer.save_pretrained("gemma3-270m-med")
```
- Pushed to huggingface

