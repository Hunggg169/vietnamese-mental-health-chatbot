import os
import random
import logging
import time
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "tiiuae/falcon-rw-1b"
DATA_PATH = "./mental_health_vi_augmented_clean.json"
OUTPUT_DIR = "./finetuned-falcon-lora-final"

MAX_LEN = 768
EPOCHS = 3
PER_DEVICE_BATCH = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
SEED = 42
VALIDATION_SPLIT = 0.05
SAVE_TOTAL_LIMIT = 2
USE_FP16 = True
OPTIMIZER = "adamw_torch"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def extract_last_assistant_reply(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Trợ lý:"):
            prompt = "\n".join(lines[:i]) + "\nTrợ lý:"
            response = lines[i].replace("Trợ lý:", "").strip()
            return prompt, response
    return "Trợ lý:", text.strip()


def build_text_from_example(example):
    if "instruction" in example and "output" in example:
        instr = example.get("instruction", "").strip()
        out = example.get("output", "").strip()
        if out:
            return f"Người dùng: {instr}\nTrợ lý:", out
        return extract_last_assistant_reply(instr)
    return extract_last_assistant_reply(example.get("text", "").strip())


def tokenize_and_build(example, tokenizer):
    prompt, response = build_text_from_example(example)
    if not response:
        return None
    full_text = prompt + " " + response + tokenizer.eos_token
    enc = tokenizer(
        full_text, truncation=True, max_length=MAX_LEN, padding="max_length"
    )
    labels = enc["input_ids"].copy()
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    labels = [-100 if t == tokenizer.pad_token_id else t for t in labels]
    enc["labels"] = labels
    return enc


def process_dataset(ds, tokenizer):
    return Dataset.from_list(
        [x for x in (tokenize_and_build(ex, tokenizer) for ex in ds) if x]
    )


class TimeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        t = time.time() - self.epoch_start
        logger.info(f"Epoch {int(state.epoch)} time: {t/60:.2f} phút")

    def on_train_end(self, args, state, control, **kwargs):
        t = time.time() - self.train_start
        logger.info(f"Tổng thời gian train: {t/60:.2f} phút")


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    if VALIDATION_SPLIT > 0:
        ds = ds.train_test_split(test_size=VALIDATION_SPLIT, seed=SEED)
        train_ds, eval_ds = ds["train"], ds["test"]
    else:
        train_ds, eval_ds = ds, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)

    train_tok = process_dataset(train_ds, tokenizer)
    eval_tok = process_dataset(eval_ds, tokenizer) if eval_ds else None

    train_tok.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    if eval_tok:
        eval_tok.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,
        weight_decay=0.05,
        fp16=USE_FP16,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_tok else "no",
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        optim=OPTIMIZER,
        gradient_checkpointing=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[TimeCallback()],
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
