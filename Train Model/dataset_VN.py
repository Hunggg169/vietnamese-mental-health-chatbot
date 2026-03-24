from datasets import load_dataset
import re
import json
from deep_translator import GoogleTranslator

# 1. Load dataset gốc
dataset = load_dataset("heliosbrahma/mental_health_chatbot_dataset")


# 2. Hàm tách HUMAN / ASSISTANT
def split_text(example):
    text = example["text"]

    # Regex để tách <HUMAN> và <ASSISTANT>
    match = re.match(r"<HUMAN>:(.*)\n<ASSISTANT>:(.*)", text, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        question, answer = text, ""  # fallback nếu không tách được

    return {"question": question, "answer": answer}


dataset = dataset.map(split_text)

# 3. Hàm dịch sang tiếng Việt
translator = GoogleTranslator(source="en", target="vi")


def translate_batch(batch):
    questions = batch["question"]
    answers = batch["answer"]

    questions_vi = [translator.translate(q) if q else "" for q in questions]
    answers_vi = [translator.translate(a) if a else "" for a in answers]

    return {
        "instruction": questions_vi,  # đầu vào cho model
        "input": ["" for _ in questions_vi],  # có thể để trống
        "output": answers_vi,  # câu trả lời
    }


dataset_vi = dataset["train"].map(translate_batch, batched=True, batch_size=16)

# 4. Xuất ra file JSON để train
output_file = "mental_health_vi.json"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in dataset_vi:
        json.dump(
            {
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

print(f"Dataset đã dịch và lưu vào {output_file}")
