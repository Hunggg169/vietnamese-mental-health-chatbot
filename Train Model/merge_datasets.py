import json
from tqdm import tqdm

# CẤU HÌNH
DATA_FILES = [
    "./mental_health_vi.json",  # file gốc 10k
    "./mental_health_vi_augmented_clean.json",  # file mới 6.7k
]
OUTPUT_FILE = "./mental_health_vi_merged.json"


# HÀM GỘP
def normalize_text(text):
    return " ".join(text.lower().strip().split())


def main():
    all_data = []
    seen = set()

    # Đọc từng file
    for path in DATA_FILES:
        print(f"Đang đọc: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"  → {len(data):,} mẫu")
            for ex in tqdm(data, desc=f"Đang xử lý {path}"):
                instr = normalize_text(ex["instruction"])
                out = normalize_text(ex["output"])
                key = instr + " || " + out
                if key not in seen:
                    seen.add(key)
                    all_data.append(ex)

    print(f"\nTổng số mẫu sau khi gộp & loại trùng: {len(all_data):,}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu dataset gộp tại: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
