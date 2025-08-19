import torch
from unsloth import FastLanguageModel
from getpass import getpass

# 1. Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs_medical_sft", # Load from our saved directory
)

# 2. Get Hugging Face credentials
hf_username = input("Enter your Hugging Face username: ")
hf_token = getpass("Enter your Hugging Face token (will be hidden): ")
model_name = "gemma-3-270m-medical-sft"

# 3. Push LoRA adapters to the Hub
print("Pushing LoRA model to the Hub...")
model.push_to_hub(f"{hf_username}/{model_name}-lora", token=hf_token)
tokenizer.push_to_hub(f"{hf_username}/{model_name}-lora", token=hf_token)
print("LoRA model pushed successfully.")

# 4. Merge adapters and push the full model
print("\nMerging adapters and pushing the full model to the Hub...")
model.save_pretrained_merged(f"{model_name}", tokenizer, save_method = "merged_16bit",)
model.push_to_hub_merged(f"{hf_username}/{model_name}", tokenizer, save_method = "merged_16bit", token = hf_token)
print("Full model pushed successfully.")

# 4.5. Reload merged model for GGUF export
from unsloth import FastLanguageModel
merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(model_name)

# 5. Save and push GGUF version
print("\nSaving and pushing GGUF version to the Hub...")
model.save_pretrained_gguf(f"{model_name}", tokenizer)
model.push_to_hub_gguf(f"{hf_username}/{model_name}", tokenizer, token = hf_token)
print("GGUF model pushed successfully.")

print("\nAll models have been pushed to the Hugging Face Hub!")
