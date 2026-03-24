Vietnamese Mental Health Chatbot (LLM Fine-tuning)
Overview

This project focuses on fine-tuning a large language model to build a Vietnamese mental health chatbot. The system is designed to provide supportive and context-aware responses using domain-specific data.

Model
Base Model: Falcon
Fine-tuning Method: LoRA (Low-Rank Adaptation)
Framework: PyTorch, Hugging Face Transformers

Model: https://huggingface.co/HungHz/falcon-lora-merged

Features
Vietnamese conversational chatbot
Domain-specific responses (mental health support)
Lightweight fine-tuned model using LoRA
Deployable for web-based interaction
Training Pipeline
Data collection and preprocessing
Formatting dataset for instruction tuning
Fine-tuning using LoRA
Merging LoRA weights into base model
Evaluation and testing
Tech Stack
Language: Python
Frameworks: PyTorch, Transformers
Techniques: LoRA fine-tuning
Deployment: Web-based chatbot (Node.js / Flask)
Database: MongoDB
Project Structure
.
├── data/
├── training/
├── inference/
├── app/
├── README.md
Limitations
Limited dataset size
Model may generate inaccurate responses in complex cases
Not a replacement for professional mental health support
Future Improvements
Larger and higher-quality dataset
Better evaluation metrics
Multi-turn conversation optimization
Deployment optimization (latency & scaling)
Author
Name: Quản Trọng Hùng
