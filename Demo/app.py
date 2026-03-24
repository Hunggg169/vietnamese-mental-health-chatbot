# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import torch
import re
import sys
import unicodedata
import os
import logging
import uuid
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from pymongo import MongoClient, errors
from dotenv import load_dotenv

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
load_dotenv()
app.secret_key = "secret_key"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_NAME = "HungHz/falcon-lora-merged"

DEVICE = "cpu"
MAX_NEW_TOKENS = 200
sys.stdout.reconfigure(encoding="utf-8")

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    MONGO_URI = "mongodb+srv://<username>:<password>@cluster0.mongodb.net/"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["chatbot_psychology"]
    chats = db["chat_history"]
except errors.ServerSelectionTimeoutError:
    chats = None

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32
).to(DEVICE)

model.eval()


def clean_text(text):
    if not isinstance(text, str):
        return ""

    if "Trợ lý:" in text:
        text = text.split("Trợ lý:")[-1]

    text = unicodedata.normalize("NFC", text)

    text = "".join(ch for ch in text if ch.isprintable())

    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s,.!?]", " ", text)

    text = re.sub(r"\s{2,}", " ", text).strip()

    sentences = re.split(r"[.!?]", text)

    text = ". ".join(s.strip() for s in sentences[:3] if s.strip())

    if text and not text.endswith("."):
        text += "."

    return text


def chat(prompt):
    try:
        full_prompt = f"Người dùng: {prompt}\nTrợ lý:"

        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(DEVICE)

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        return clean_text(text)

    except Exception:
        return "Xin lỗi, tôi đang gặp sự cố khi xử lý phản hồi."


@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())


@app.route("/")
def home():
    return send_from_directory(".", "chatbot.html")


@app.route("/chat_stream", methods=["POST"])
def chat_stream():

    try:

        data = request.get_json(force=True)

        user_input = data.get("message", "").strip()

        user_id = data.get("user_id", "anonymous")

        session_id = session.get("session_id")

        if not user_input:
            return jsonify({"reply": "Xin hãy nhập tin nhắn."})

        if user_input.lower() in (
            "bye",
            "tạm biệt",
            "exit",
            "quit",
            "hẹn gặp lại",
        ):

            reply = "Cảm ơn bạn đã chia sẻ. Hãy nhớ rằng bạn xứng đáng được yêu thương và nghỉ ngơi. Hẹn gặp lại nhé!"

        else:

            reply = chat(user_input)

        if chats is not None:

            chats.insert_one(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "user_message": user_input,
                    "bot_reply": reply,
                    "timestamp": datetime.now(),
                }
            )

        return jsonify({"reply": reply})

    except Exception:

        return jsonify({"reply": "Đã xảy ra lỗi khi xử lý yêu cầu."})


@app.route("/clear", methods=["POST"])
def clear_chat():

    try:

        session_id = session.get("session_id")

        if chats is not None:
            chats.delete_many({"session_id": session_id})

        return jsonify({"status": "cleared"})

    except Exception:

        return jsonify({"status": "error"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
