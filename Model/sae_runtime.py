"""M1 — 학습된 SAE를 "추론 계기"로 감싼다.

정규화(`token_mean`/`token_std`) 왕복을 캡슐화해 호출자가 그 통계를 잊고 raw
활성을 그대로 SAE에 넣는 사고를 원천 차단한다. `FrozenSAE`는 SAE·정규화 통계를
전부 frozen buffer/submodule로만 보관하며 학습 파라미터를 만들지 않는다
(gain은 M2에서 별도 모듈이 갖는다).
"""

import torch
from torch import nn

from Model.SAE import VanillaL1SAE


class FrozenSAE(nn.Module):
    def __init__(self, sae: VanillaL1SAE, token_mean: torch.Tensor, token_std: torch.Tensor,
                 active_threshold: float, meta: dict):
        """학습된 SAE와 그 정규화 통계를 한 객체로 묶는다.
        보관: sae(파라미터 전부 requires_grad=False, eval), token_mean/token_std/
              active_threshold는 buffer로, meta는 ckpt의 model_name·target_block·token_scope.
        직접 부르는 일은 드물고 보통 from_checkpoint를 쓴다."""
        super().__init__()
        if token_mean.shape != (1, sae.input_dim):
            raise ValueError(f"token_mean.shape must be (1, {sae.input_dim}), got {tuple(token_mean.shape)}")
        if token_std.shape != (1, sae.input_dim):
            raise ValueError(f"token_std.shape must be (1, {sae.input_dim}), got {tuple(token_std.shape)}")

        sae = sae.eval()
        for p in sae.parameters():
            p.requires_grad_(False)
        self.sae = sae

        self.register_buffer("token_mean", token_mean.detach().float())
        self.register_buffer("token_std", token_std.detach().float())
        self.register_buffer("active_threshold_buf", torch.tensor(float(active_threshold)))
        self.meta = dict(meta)

    @property
    def input_dim(self) -> int:
        """SAE 입력(=백본 활성) 차원. 보통 768."""
        return self.sae.input_dim

    @property
    def hidden_dim(self) -> int:
        """SAE 코드(=개념 딕셔너리) 차원. 보통 12288 (expansion 16)."""
        return self.sae.hidden_dim

    @property
    def active_threshold(self) -> float:
        """l0()가 기본으로 쓰는 발화 판정 임계값(float)."""
        return float(self.active_threshold_buf.item())

    @classmethod
    def from_checkpoint(cls, path: str, device) -> "FrozenSAE":
        """ckpt 파일 하나에서 SAE와 정규화 통계를 전부 복원해 조립한다."""
        ck = torch.load(path, map_location="cpu")
        sae = VanillaL1SAE(
            ck["input_dim"],
            ck["hidden_dim"],
            b_dec_init=ck["b_dec_init"],
            dec_bias_mode="geom",
        )
        sae.load_state_dict(ck["sae_state_dict"])
        sae = sae.eval()
        for p in sae.parameters():
            p.requires_grad_(False)

        meta = {
            "model_name": ck.get("model_name"),
            "target_block": ck.get("target_block"),
            "token_scope": ck.get("token_scope"),
        }
        frozen = cls(
            sae,
            ck["token_mean"],
            ck["token_std"],
            ck["active_threshold"],
            meta,
        )
        frozen = frozen.to(device)

        # 검증: unit-norm 유지 확인
        row_norms = frozen.sae.W_dec.norm(dim=1)
        if not torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-2):
            raise ValueError(
                f"W_dec rows are not unit-norm after loading checkpoint (min={row_norms.min().item():.4f}, "
                f"max={row_norms.max().item():.4f})"
            )
        return frozen

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        """백본 활성(raw) -> SAE 입력 공간.  (h - token_mean) / token_std.
        SAE는 정규화된 토큰으로 학습됐으므로 raw h를 그냥 넣으면 안 된다."""
        return (h - self.token_mean) / self.token_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """normalize의 역변환. SAE 출력(정규화 공간)을 백본이 이해하는 raw 공간으로 되돌린다."""
        return x * self.token_std + self.token_mean

    @torch.no_grad()
    def encode(self, h: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
        """활성 -> 개념 코드. [N,768] -> [N,12288] 비음수 희소 벡터.
        normalized=False면 내부에서 normalize를 먼저 부른다. 항상 no_grad."""
        x = h if normalized else self.normalize(h)
        return self.sae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """개념 코드 -> 활성. encode의 역방향, 단 정규화 공간 출력이다."""
        return self.sae.decode(z)

    def decode_delta(self, code_delta: torch.Tensor) -> torch.Tensor:
        """코드의 '변화량'을 raw 활성 공간의 변화량으로 되돌린다. [N,K] -> [N,768].
        decode()는 절대량이라 b_dec·token_mean이 붙지만, 이건 차분이라 그 덧셈 항들이
        상쇄되고 곱셈 항 token_std만 남는다:  code_delta @ W_dec * token_std
        gain 개입이 쓰는 유일한 경로 — 덕분에 호출자가 W_dec·token_std를 직접 만질 일이 없다."""
        return (code_delta @ self.sae.W_dec) * self.token_std

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        """코드를 raw 활성 공간의 '절대량'으로 되돌린다. denormalize(decode(z)).
        decode_delta가 차분이라면 이건 전체 값 — residual=False(h 통째 대체) 경로가 쓴다."""
        return self.denormalize(self.decode(z))

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """encode -> decode_raw 왕복. SAE를 통과시킨 h를 raw 공간으로 돌려준다.
        '이 SAE가 표현을 얼마나 보존하나'를 눈으로 볼 때 쓴다. = decode_raw(encode(h))"""
        return self.decode_raw(self.encode(h))

    @torch.no_grad()
    def fvu(self, h: torch.Tensor) -> float:
        """재구성 실패율. 0이면 완벽, 클수록 SAE가 이 입력을 못 담고 있다는 뜻.
        실험 1의 게이트 임계와 M7의 런타임 판정이 이 값을 공유한다.
        Utils.SAE_utils.evaluate_sae_tokens의 normalized_mse와 동일 정의."""
        x = self.normalize(h)
        xh = self.decode(self.encode(x, normalized=True))
        mse = ((x - xh) ** 2).mean()
        # clamp_min으로 하한을 건다. 파이썬 max()에 CPU 리터럴 텐서를 섞으면
        # h가 CUDA일 때 device 불일치로 터진다.
        var = x.var(unbiased=False).clamp_min(1e-12)
        return (mse / var).item()

    def l0(self, z: torch.Tensor, threshold: float | None = None) -> float:
        """토큰 하나당 평균 몇 개의 개념이 켜졌나. 희소성 지표.
        M2의 RandomDictGainBasis가 sparsity를 맞출 때 이 측정값을 받아 쓴다."""
        thr = self.active_threshold if threshold is None else threshold
        return (z > thr).sum(-1).float().mean().item()
