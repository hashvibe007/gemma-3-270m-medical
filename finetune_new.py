import torch
import logging
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_from_disk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="finetune.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

logger.info("Starting script")

# 1. Load Model and Tokenizer
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

logger.info("Loading model and tokenizer")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
logger.info("Model and tokenizer loaded")

# 2. Configure LoRA
logger.info("Configuring LoRA")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
logger.info("LoRA configured")

# 3. Load Processed Dataset
logger.info("Loading dataset")
dataset = load_from_disk("processed_dataset_medical_sft")
logger.info("Dataset loaded")

# 4. Set up Trainer
logger.info("Setting up Trainer")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 600, # We'll do a short training run for this sample
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        logging_dir = "logs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_medical_sft",
    ),
)
logger.info("Trainer set up")

# 5. Start Training
logger.info("Starting model training")
trainer.train()
logger.info("Training complete")

# 6. Save Model
logger.info("Saving model")
from peft import PeftModel
is_peft = isinstance(model, PeftModel)
logger.info(f"Is model a PEFT (LoRA) model: {is_peft}")
if not is_peft:
    logger.warning("Model is NOT a PEFT (LoRA) model. LoRA adapters will NOT be saved! Check your training logic.")
model.save_pretrained("outputs_medical_sft")
logger.info("Model saved to 'outputs_medical_sft' directory")
logger.info("Script finished")
