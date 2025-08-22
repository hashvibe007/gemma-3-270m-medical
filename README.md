# Finetune Gemma

This repository contains code and instructions for finetuning the `gemma-3-270m` model on a medical ECG dataset. The finetuned model is available on [Hugging Face](https://huggingface.co/hashvibe007/gemma-3-270m-medical-sft-lora).

## Features

- FreedomIntelligence/medical-o1-reasoning-SFT"
- Model checkpoints and logs
- Instructions for training and evaluation

## Usage

1. Clone the repository.
2. uv init
3. uv add -r requirements.txt
3. uv prepare_new_data.py
5. uv finetune_new.py
6. uv run inference.py

## Model

- **Base model:** gemma-3-270m
- **Finetuned for:** FreedomIntelligence/medical-o1-reasoning-SFT"
