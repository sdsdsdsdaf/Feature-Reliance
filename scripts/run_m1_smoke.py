"""M1 in-domain 검증 smoke: FrozenSAE의 FVU/L0/cosine/alive-latent를 실측하고
outputs/experiments/M1/result.json으로 저장한다 (docs/plans/contracts/README.md
공통 헤더 규약). CPU에서 완주 가능하도록 imagenette val 소규모 subset만 쓴다."""

import datetime
import json
import subprocess
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from torchvision.datasets import Imagenette

from Model.sae_runtime import FrozenSAE
from Utils.SAE_utils import TransformDataset, collect_tokens_with_hook

CHECKPOINT_PATH = "outputs/reservoir_sae/vit_b_sae.pt"
TASK_ID = "M1"
SEED = 0
NUM_IMAGES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def git_commit_hash() -> str:
    """현재 HEAD 커밋 해시(40자)를 반환한다. 실패 시 'unknown'."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main():
    """imagenette val 소규모 subset으로 M1 재구성 품질을 실측하고 result.json을 쓴다."""
    torch.manual_seed(SEED)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    frozen = FrozenSAE.from_checkpoint(CHECKPOINT_PATH, device=DEVICE)

    model = timm.create_model(frozen.meta["model_name"], pretrained=True)
    model.eval().to(DEVICE)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    base_dataset = Imagenette(root="data", split="val", download=False)
    dataset = TransformDataset(base_dataset, transform=transform)
    subset = torch.utils.data.Subset(dataset, list(range(min(NUM_IMAGES, len(dataset)))))
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

    tokens = collect_tokens_with_hook(
        model,
        loader,
        max_tokens=None,
        target_block=frozen.meta["target_block"],
        token_scope=frozen.meta["token_scope"],
        device=DEVICE,
        cache_dtype=torch.float32,
    )

    # collect_tokens_with_hook은 대용량 덤프를 대비해 CPU에 캐시한다.
    # FrozenSAE의 buffer는 DEVICE에 있으므로 여기서 맞춰준다.
    tokens = tokens.to(DEVICE)

    fvu = frozen.fvu(tokens)
    x = frozen.normalize(tokens)
    z = frozen.encode(x, normalized=True)
    l0 = frozen.l0(z)
    xh = frozen.decode(z)
    cosine = torch.nn.functional.cosine_similarity(x, xh, dim=1).mean().item()
    alive_latents = int((z > frozen.active_threshold).any(dim=0).sum().item())

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # phaseM.md 실측 참고치와의 정합성 판정 (order-of-magnitude 정합만 확인;
    # M1은 자체 수치 게이트를 정의하지 않고 T1.1이 이 값을 baseline으로 쓴다)
    expected_fvu_ref = 4e-4
    expected_l0_ref = 497
    fvu_ok = fvu < expected_fvu_ref * 10  # 한 자릿수 이내
    l0_ok = expected_l0_ref * 0.3 < l0 < expected_l0_ref * 3
    if fvu_ok and l0_ok:
        verdict = "pass"
        verdict_reason = "FVU/L0가 phaseM.md 실측 참고치(FVU~4e-4, L0~497)와 같은 자릿수/범위 내 일치"
    else:
        verdict = "fail"
        verdict_reason = (
            f"FVU/L0가 참고치에서 벗어남: fvu={fvu:.6g}(ref~{expected_fvu_ref:g}), "
            f"l0={l0:.4g}(ref~{expected_l0_ref})"
        )

    result = {
        "task_id": TASK_ID,
        "git_commit": git_commit_hash(),
        "seed": SEED,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": {
            "checkpoint": CHECKPOINT_PATH,
            "num_images": NUM_IMAGES,
            "dataset": "imagenette2/val",
            # 어느 장치에서 잰 수치인지 남긴다 — T1.1이 이 값을 기준선으로 쓴다.
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "model_name": frozen.meta["model_name"],
            "target_block": frozen.meta["target_block"],
            "token_scope": frozen.meta["token_scope"],
            "active_threshold": frozen.active_threshold,
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "result": {
            "fvu": fvu,
            "mean_l0": l0,
            "cosine": cosine,
            "alive_latents": alive_latents,
            "hidden_dim": frozen.hidden_dim,
            "input_dim": frozen.input_dim,
            "num_tokens": int(tokens.shape[0]),
            "checkpoint_fields_verified": {
                "input_dim": frozen.input_dim,
                "hidden_dim": frozen.hidden_dim,
                "token_mean_shape": list(frozen.token_mean.shape),
                "token_std_shape": list(frozen.token_std.shape),
                "active_threshold": frozen.active_threshold,
            },
            "reference_from_phaseM_md": {
                "l0_ref": expected_l0_ref,
                "fvu_ref": expected_fvu_ref,
                "measured_on": "imagenette 200 images / 39200 patch tokens (earlier session)",
            },
        },
        "unknown_fields": [],
    }

    out_dir = Path("outputs/experiments/M1")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
