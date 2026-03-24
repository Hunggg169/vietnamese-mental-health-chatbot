# -*- coding: utf-8 -*-
"""
test_lora_clean.py
Mục đích: Chat và kiểm thử model LoRA đã fine-tune (base 4-bit + adapter).
Model được nạp ở chế độ 4-bit để tiết kiệm VRAM, nhưng vẫn có thể trả lời mượt.
Câu trả lời được rút gọn và làm sạch tự động.
"""

import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Bảo đảm in ra tiếng Việt đúng font
sys.stdout.reconfigure(encoding="utf-8")

# Cấu hình cơ bản
MODEL_NAME = "tiiuae/falcon-rw-1b"  # Base model
LORA_ADAPTER_PATH = "./finetuned-falcon-lora-final"  # Đường dẫn adapter LoRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 120  # Giới hạn độ dài câu trả lời

# Cấu hình nạp model 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Đang tải tokenizer và base model (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(
    LORA_ADAPTER_PATH if os.path.isdir(LORA_ADAPTER_PATH) else MODEL_NAME,
    use_fast=True,
)

# Thêm token PAD nếu thiếu
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Tải base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Resize embedding nếu số lượng token khác nhau
if base_model.get_input_embeddings().weight.shape[0] != len(tokenizer):
    old_size = base_model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)
    print(f"Resize embedding từ {old_size} → {new_size}")
    base_model.resize_token_embeddings(new_size)

# Nạp adapter LoRA
print("Nạp LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, LORA_ADAPTER_PATH, device_map="auto"
)

print("\nModel sẵn sàng để test.\nGõ 'exit' để thoát.\n")


# Hàm sinh phản hồi (chat)
def chat(prompt: str) -> str:
    """
    Sinh phản hồi từ model theo phong cách hội thoại.
    Làm sạch kết quả đầu ra, chỉ giữ tối đa 2 câu mạch lạc.
    """
    # Chuẩn hóa prompt giống dữ liệu huấn luyện
    full_prompt = f"Người dùng: {prompt}\nTrợ lý:"
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    ).to(DEVICE)

    # Sinh văn bản
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.55,
        top_p=0.92,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Giải mã kết quả
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Lấy phần sau "Trợ lý:"
    if "Trợ lý:" in text:
        text = text.split("Trợ lý:")[-1]

    # Làm sạch ký tự đặc biệt, xuống dòng, khoảng trắng
    text = text.replace("\n", " ").replace("  ", " ").strip()
    text = re.sub(
        r"[^\w\s,.?!àáạãảâầấậẫẩăằắặẵẳèéẹẽẻêềếệễểìíịĩỉòóọõỏôồốộỗổơờớợỡởùúụũủưừứựữửỳýỵỹỷđA-Za-z0-9]",
        "",
        text,
    )
    text = text.strip()

    # Cắt chỉ giữ tối đa 2 câu
    sentences = re.split(r"[.!?]", text)
    text = ". ".join(s.strip() for s in sentences if s.strip())[:300]
    text = ". ".join(sentences[:2]).strip() + "."

    return text


# Chạy chế độ chat
if __name__ == "__main__":
    try:
        while True:
            user_input = input("Người dùng: ").strip()
            if user_input.lower() in ("exit", "quit", "bye"):
                print(
                    "Trợ lý: Cảm ơn bạn. Chúc bạn một ngày thật tốt lành nhé!"
                )
                break

            reply = chat(user_input)
            print(f"\nTrợ lý: {reply}\n{'-' * 60}")

    except KeyboardInterrupt:
        print("\nĐã dừng chương trình bằng bàn phím.")
