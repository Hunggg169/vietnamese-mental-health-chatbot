from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "tiiuae/falcon-rw-1b"
lora_path = "./finetuned-falcon-lora-final"
merged_path = "./falcon-lora-merged"

print("Merge LoRA vào base model...")
tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype="auto", device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, lora_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merge hoàn tất: {merged_path}")
