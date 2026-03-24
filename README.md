Vietnamese Mental Health Chatbot (LLM Fine-tuning)
Overview

This project focuses on fine-tuning a Large Language Model (LLM) to build a Vietnamese mental health chatbot. The system is designed to generate supportive, context-aware responses using domain-specific data and can be integrated into real-time web applications.

Model
Base Model: Falcon
Fine-tuning Method: LoRA (Low-Rank Adaptation)
Framework: PyTorch, Hugging Face Transformers

Pretrained Model:
https://huggingface.co/HungHz/falcon-lora-merged

Features
Vietnamese conversational chatbot
Domain-specific responses for mental health support
Lightweight fine-tuned model using LoRA
Real-time interaction via web integration
Training Pipeline
Data collection and preprocessing
Instruction-format dataset preparation
Fine-tuning using LoRA
Merging LoRA weights into base model
Evaluation and testing
Tech Stack
Language: Python
Frameworks: PyTorch, Transformers
Techniques: LoRA fine-tuning
Deployment: Flask / Node.js
Database: MongoDB

Limitations
Limited dataset size
May produce inaccurate responses in complex scenarios
Not intended to replace professional mental health services
Future Improvements
Expand dataset size and quality
Improve evaluation metrics
Optimize multi-turn conversations
Enhance deployment performance and scalability
Author
Name: Quản Trọng Hùng
