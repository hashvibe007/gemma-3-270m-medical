
import torch
import logging
from unsloth import FastLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="inference.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

logger.info("Starting inference script")

# 1. Load Model and Tokenizer from the fine-tuned directory
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

logger.info("Loading model and tokenizer from 'outputs_medical_sft'")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs_medical_sft", # YOUR MODEL YOU SAVED
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
logger.info("Model and tokenizer loaded")

# 2. Prepare a sample prompt for inference
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a highly skilled medical professional. Your task is to analyze the provided ECG data and generate a concise, accurate medical report.

### Input:
A 55-year-old male patient presents with intermittent chest pain and shortness of breath. The ECG shows ST-segment elevation in leads V1-V4.

### Response:
"""
logger.info(f"Using prompt:\n{prompt}")

# 3. Tokenize the prompt and generate a response
logger.info("Tokenizing prompt and generating response")
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1048, use_cache=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.info("Response generated")

# 4. Print the response
print(response)
logger.info(f"Generated response:\n{response}")
logger.info("Inference script finished")
