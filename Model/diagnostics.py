"""M6 — `c_k` 진단 모듈.

`c_k = acc(gain ≡ 1) − acc(gain with gain_k = 0)`. `gain_k = 0`은 residual 주입에서
latent k의 기여만 빼는 것과 정확히 같다. `GainIntervention`의 `gain ≡ 1` baseline이
무개입 forward와 비트 동일하므로(M2 불변식), 이 baseline과의 차이는 재구성 오차가
섞이지 않은 latent k 하나의 causal 기여만 반영한다.

비용이 핵심 제약이다: naive하게 `n_candidates × n_images`번 전체 forward를 돌리면
끝나지 않는다. 이 모듈은 두 가지로 비용을 줄인다.

1. 이미지마다 `h`(block10 출력)와 `code`를 한 번만 계산해 캐시하고, latent별
   ablation은 tail(blocks[hook_block+1:] + norm + head)만 재실행한다 — 전체
   네트워크의 2/12만 다시 돈다.
2. latent 하나를 지우는 delta는 `code`의 열 하나에만 의존한다는 선형성을 이용해,
   `basis.delta`를 "probe" 코드(후보별 one-hot)에 한 번만 호출해 후보 전체의
   딕셔너리 행을 `[latent_chunk, D]` 행렬로 뽑아낸다. 이후 각 배치·각 latent의
   delta는 `code[:, k] * dict_row[k]`라는 outer product 하나로 끝난다(`basis`의
   내부 표현이 무엇이든 `delta`가 code에 대해 선형이기만 하면 성립하는 일반적인
   방법 — `GainBasis` 구현체 내부(`W_dec`/`R` 등)를 직접 만지지 않는다).

`GainIntervention.forward`를 그대로 호출하지 않고 backbone/basis를 직접 조립하는
이유는, forward는 `intervention.gain` 하나만 쓰는데 이 함수는 후보마다 다른 "그
latent만 끈" 가상의 gain을 대량으로 병렬 평가해야 하기 때문이다. 그 결과
`intervention.gain` 자체는 이 함수 실행 중 전혀 건드리지 않는다 — 그래도 방어적으로
snapshot/restore를 둔다(예외가 나도 상태가 새지 않게).
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_FIRING_RATE_FLOOR = 0.005  # 발화율 하한(0.5%) — phaseM.md M6 절 예시값


def _iterate_capped(loader, device, max_images):
    """loader를 순회하되 누적 이미지 수가 max_images를 넘지 않게 마지막 배치를 자른다.
    max_images가 None이면 loader를 있는 그대로 끝까지 순회한다."""
    seen = 0
    for x, y in loader:
        if max_images is not None:
            if seen >= max_images:
                break
            if seen + x.shape[0] > max_images:
                take = max_images - seen
                x, y = x[:take], y[:take]
        seen += x.shape[0]
        yield x.to(device), y.to(device)


def _upstream(intervention, x: Tensor):
    """GainIntervention.forward의 상류(blocks[:hook_block+1])를 그대로 재현해
    (prefix, patch_shape, flat, code)를 얻는다. gain은 전혀 쓰지 않는다 — 이 시점의
    활성·code는 gain과 무관한 상수이기 때문이다."""
    backbone = intervention.backbone
    basis = intervention.basis
    with torch.no_grad():
        h = backbone.patch_embed(x)
        h = backbone._pos_embed(h)
        h = backbone.patch_drop(h)
        h = backbone.norm_pre(h)
        for blk in backbone.blocks[: intervention.hook_block + 1]:
            h = blk(h)
        prefix = h[:, : intervention.num_prefix_tokens]
        patch = h[:, intervention.num_prefix_tokens :]
        flat = patch.reshape(-1, patch.shape[-1])  # [B*T, D]
        code = basis.encode(flat)  # [B*T, gain_dim], 상수
    return prefix, patch.shape, flat, code


def _tail(intervention, prefix: Tensor, patch_shape, patch_plus_flat: Tensor) -> Tensor:
    """단일 변형(하나의 gain 세팅)에 대해 tail(blocks[hook_block+1:] + norm + head)만
    돌려 logits를 낸다. baseline 정확도 산출에 쓴다."""
    backbone = intervention.backbone
    with torch.no_grad():
        patch_plus = patch_plus_flat.view(patch_shape)
        h_plus = torch.cat([prefix, patch_plus], dim=1)
        for blk in backbone.blocks[intervention.hook_block + 1 :]:
            h_plus = blk(h_plus)
        h_plus = backbone.norm(h_plus)
        feat = backbone.forward_head(h_plus, pre_logits=True)
        logits = backbone.head(feat)
    return logits


def _tail_chunk(intervention, prefix: Tensor, patch_shape, patch_plus_flat_expanded: Tensor, c: int) -> Tensor:
    """latent_chunk개의 ablation 변형을 배치 축으로 이어붙여 tail을 **한 번만** 돌린다.
    patch_plus_flat_expanded: [B*T, c, D]. 반환: logits [c, B, num_classes]."""
    backbone = intervention.backbone
    b, t, d = patch_shape
    patch_mega = patch_plus_flat_expanded.view(b, t, c, d).permute(2, 0, 1, 3).reshape(c * b, t, d)
    num_prefix = prefix.shape[1]
    prefix_mega = prefix.unsqueeze(0).expand(c, -1, -1, -1).reshape(c * b, num_prefix, d)
    with torch.no_grad():
        h_plus = torch.cat([prefix_mega, patch_mega], dim=1)
        for blk in backbone.blocks[intervention.hook_block + 1 :]:
            h_plus = blk(h_plus)
        h_plus = backbone.norm(h_plus)
        feat = backbone.forward_head(h_plus, pre_logits=True)
        logits = backbone.head(feat)
    return logits.view(c, b, -1)


def _firing_threshold(basis) -> float:
    """"발화"의 판정 임계값. SAE 기반 basis(latent/random_dict)는 FrozenSAE.l0()과
    같은 정의(active_threshold, 보통 0.2)를 써야 M6의 후보 선정이 M1의 L0 측정과
    같은 척도를 공유한다 — SAE 코드는 ReLU라 대부분의 latent가 항상 미세한 양수를
    갖고, 정의를 `code > 0`으로 두면 사실상 전부가 "발화"로 잡혀 후보 축소가
    무의미해진다. basis에 sae가 없으면(예: ChannelGainBasis) 0.0으로 fallback한다."""
    sae = getattr(basis, "sae", None)
    if sae is not None and hasattr(sae, "active_threshold"):
        return float(sae.active_threshold)
    return 0.0


def _latent_firing_rate(intervention, loader, device, max_images) -> Tensor:
    """전체 K개 latent 각각의 발화율(코드가 _firing_threshold를 넘는 토큰 비율)을 측정한다.
    후보를 추리기 위한 사전 스캔이며, code 자체는 저장하지 않고 카운트만 누적한다."""
    gain_dim = intervention.basis.gain_dim
    threshold = _firing_threshold(intervention.basis)
    fire_counts = torch.zeros(gain_dim, device=device)
    total_tokens = 0
    for x, _y in _iterate_capped(loader, device, max_images):
        _, _, _, code = _upstream(intervention, x)
        fire_counts += (code > threshold).float().sum(dim=0)
        total_tokens += code.shape[0]
    if total_tokens == 0:
        return fire_counts  # 전부 0 — 아래에서 후보가 빈 채로 처리된다
    return fire_counts / total_tokens


def compute_c_k(
    intervention,
    loader,
    device,
    *,
    candidates: Tensor | None = None,
    latent_chunk: int = 64,
    max_images: int | None = None,
) -> Tensor:
    """latent별 causal 기여도 `c_k = acc(gain≡1) − acc(gain_k=0)`을 측정한다.

    candidates: 평가할 latent index(LongTensor, 1차원). None이면 발화율이
        `DEFAULT_FIRING_RATE_FLOOR` 이상인 latent만 자동 선정한다(전체 K 중 상당수는
        한 번도 발화하지 않으므로 전수 평가를 피한다).
    latent_chunk: 한 번의 tail 재실행에 함께 태울 latent 후보 수.
    max_images: 캐시할 최대 이미지 수. None이면 loader 전체를 쓴다(실전에서는
        비용 때문에 사실상 필수).

    반환: `[K]` 텐서. 후보에서 제외된 latent는 정확히 `0.0`. 부호가 있을 수 있다
    (제거했더니 오히려 정확도가 오르면 음수) — 클램프하지 않는다.
    """
    basis = intervention.basis
    gain_dim = basis.gain_dim
    original_gain = intervention.gain.detach().clone()

    try:
        if candidates is None:
            freq = _latent_firing_rate(intervention, loader, device, max_images)
            candidates = freq.ge(DEFAULT_FIRING_RATE_FLOOR).nonzero(as_tuple=True)[0]
        candidates = candidates.to(device=device, dtype=torch.long)

        c_k = torch.zeros(gain_dim, device=device)
        n_cand = candidates.numel()
        if n_cand == 0:
            return c_k.cpu()

        # 캐시 패스: h(block10 출력)와 code를 이미지마다 한 번만 계산해 저장.
        # code는 후보 열만 남겨 메모리를 아낀다(K 전체를 들고 있지 않는다).
        cached = []
        baseline_correct = 0
        total_images = 0
        for x, y in _iterate_capped(loader, device, max_images):
            prefix, patch_shape, flat, code = _upstream(intervention, x)
            logits = _tail(intervention, prefix, patch_shape, flat)
            baseline_correct += (logits.argmax(dim=1) == y).sum().item()
            total_images += x.shape[0]
            cached.append(
                {
                    "prefix": prefix,
                    "patch_shape": patch_shape,
                    "flat": flat,
                    "code_cand": code[:, candidates],
                    "y": y,
                }
            )

        if total_images == 0:
            return c_k.cpu()
        baseline_acc = baseline_correct / total_images

        gain_probe_zero = torch.zeros(gain_dim, device=device)
        for start in range(0, n_cand, latent_chunk):
            chunk_idx = candidates[start : start + latent_chunk]
            c = chunk_idx.numel()

            # probe: 후보별 one-hot code에 basis.delta를 한 번 호출해, 각 latent를
            # gain_k=0으로 껐을 때 h에 더해질 "단위 code당" 보정 벡터를 뽑는다.
            # delta가 code에 대해 선형이라는 사실만으로 성립하며, basis 내부의
            # 딕셔너리 표현(W_dec/R 등)을 직접 읽지 않는다.
            code_probe = torch.zeros(c, gain_dim, device=device)
            code_probe[torch.arange(c, device=device), chunk_idx] = 1.0
            dict_row = basis.delta(code_probe, gain_probe_zero)  # [c, D] == -Dict[k]*scale

            ablated_correct = torch.zeros(c, device=device)
            for item in cached:
                code_col = item["code_cand"][:, start : start + c]  # [N_tok, c]
                delta = code_col.unsqueeze(-1) * dict_row.unsqueeze(0)  # [N_tok, c, D]
                patch_plus_expanded = item["flat"].unsqueeze(1) + delta  # [N_tok, c, D]
                logits = _tail_chunk(intervention, item["prefix"], item["patch_shape"], patch_plus_expanded, c)
                correct = (logits.argmax(dim=-1) == item["y"].unsqueeze(0)).sum(dim=1)  # [c]
                ablated_correct += correct.float()

            ablated_acc = ablated_correct / total_images
            c_k[chunk_idx] = baseline_acc - ablated_acc

        return c_k.cpu()
    finally:
        intervention.gain.data.copy_(original_gain)
