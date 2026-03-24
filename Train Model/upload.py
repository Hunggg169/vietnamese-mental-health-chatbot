from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="./falcon-lora-merged",
    repo_id="HungHz/falcon-lora-merged",
    repo_type="model",
)
