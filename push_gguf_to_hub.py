
import logging
from unsloth import FastLanguageModel
from peft import PeftModel
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="push_gguf_to_hub.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

# User configuration
BASE_MODEL = "unsloth/gemma-3-270m-it"  # base model repo or path
LORA_DIR = "outputs_medical_sft"         # LoRA adapter directory
MERGED_DIR = "outputs_medical_sft_merged" # Where to save merged model
GGUF_DIR = "outputs_medical_sft_gguf"     # Where to save GGUF model (optional)
HF_REPO = None  # e.g. "your-username/your-model-name" if pushing to hub
HF_TOKEN = None # Hugging Face token if needed

# 1. Load base model
logger.info("Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
logger.info("Base model loaded.")

# 2. Load LoRA adapter into base model using PEFT
logger.info("Loading LoRA adapter into base model using PEFT...")
lora_model = PeftModel.from_pretrained(base_model, LORA_DIR)
logger.info("LoRA adapter loaded.")

# 3. Merge and save merged model
logger.info("Merging and saving merged model...")
lora_model.save_pretrained_merged(MERGED_DIR, base_tokenizer, save_method="merged_16bit")
logger.info(f"Merged model saved to '{MERGED_DIR}'")

# 4. (Optional) Export GGUF
try:
    logger.info("Exporting GGUF model...")
    lora_model.save_pretrained_gguf(GGUF_DIR, base_tokenizer)
    logger.info(f"GGUF model saved to '{GGUF_DIR}'")
except Exception as e:
    logger.warning(f"GGUF export failed: {e}")

# 5. (Optional) Push to Hugging Face Hub
if HF_REPO:
    logger.info(f"Pushing merged model to Hugging Face Hub: {HF_REPO}")
    lora_model.push_to_hub_merged(HF_REPO, base_tokenizer, save_method="merged_16bit", token=HF_TOKEN)
    logger.info("Merged model pushed to Hub.")
    try:
        logger.info(f"Pushing GGUF model to Hugging Face Hub: {HF_REPO}")
        lora_model.push_to_hub_gguf(HF_REPO, base_tokenizer, token=HF_TOKEN)
        logger.info("GGUF model pushed to Hub.")
    except Exception as e:
        logger.warning(f"GGUF push failed: {e}")

logger.info("All done.")
print("All done. Check push_gguf_to_hub.log for details.")
