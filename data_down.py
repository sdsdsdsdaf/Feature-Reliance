from datasets import load_dataset
from huggingface_hub import whoami

print(whoami())
split = "validation"
dataset_val = load_dataset(
    "ILSVRC/imagenet-1k", 
    data_files={'validation': 'data/val-*'}, 
    split='validation',
    token=True,                  # 핵심! Gated 데이터셋 접근 권한 증명
    download_mode="force_redownload" # 꼬인 캐시 무시하고 강제 다운로드
)

print(f"로드 성공: {len(dataset_val)}개 샘플")

dataset_val.save_to_disk(f"/home/user/projects/Feature-Reliance/Data/ILSVRC2012/{split}")
