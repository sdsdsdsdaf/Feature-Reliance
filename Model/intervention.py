"""M2 — gain 주입 모듈.

실험 4A의 세 arm(latent/channel/random_dict)이 같은 루프에 꽂히도록 gain의
주소지정만 basis(`Model/gain_basis.py`)로 분리하고, 이 모듈은 "block10 출력에
gain을 주입하고 tail만 학습 가능한 그래프로 남긴다"는 배관만 책임진다.
학습 파라미터는 `gain` 벡터 하나뿐이다(전역 불변식).
"""

import torch
from torch import nn

from Model.gain_basis import GainBasis


class GainIntervention(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        basis: GainBasis,
        hook_block: int = 10,
        num_prefix_tokens: int = 1,
        residual: bool = True,
    ):
        """백본과 basis를 묶고 이 모듈의 유일한 학습 파라미터를 만든다.
        - backbone 전체를 requires_grad_(False)로 얼리고 eval 모드로 고정한다
        - self.gain = nn.Parameter(torch.ones(basis.gain_dim))  ← 유일한 학습 대상
        - hook_block: 몇 번째 블록 출력에 주입할지 (기본 10 = SAE가 학습된 지점)
        - num_prefix_tokens: cls 토큰 수. 패치 토큰만 개입 대상이라 앞에서 잘라낸다
        - residual: True면 h + delta(원본 보존), False면 absolute로 h 통째 대체.
          False인데 basis가 absolute를 구현 안 했으면(supports_absolute=False)
          __init__ 시점에 ValueError로 즉시 실패시킨다(스트림 한참 돌다가 터지면 안 된다)."""
        super().__init__()
        if not residual and not getattr(basis, "supports_absolute", True):
            raise ValueError(
                f"residual=False인데 basis '{getattr(basis, 'name', basis)}'는 absolute()를 지원하지 않는다"
            )

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone

        self.basis = basis
        self.hook_block = hook_block
        self.num_prefix_tokens = num_prefix_tokens
        self.residual = residual
        self.gain = nn.Parameter(torch.ones(basis.gain_dim))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """optimizer에 넘길 파라미터. 항상 [self.gain] 하나뿐이어야 한다(전역 불변식)."""
        return [self.gain]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """이미지 배치를 받아 (logits, h') 반환.
        h'는 대조학습에서 거리를 재는 backbone feature(분류기 직전 표현)다.
        timm 내부 API를 직접 호출해 상류(blocks[:hook_block+1])를 no_grad로 묶는다.
        훅+detach 방식은 forward 시 상류 그래프를 그대로 쌓아 메모리를 낭비하므로 쓰지 않는다."""
        backbone = self.backbone
        with torch.no_grad():
            h = backbone.patch_embed(x)
            h = backbone._pos_embed(h)
            h = backbone.patch_drop(h)
            h = backbone.norm_pre(h)
            for blk in backbone.blocks[: self.hook_block + 1]:
                h = blk(h)
            h = h.detach()
            prefix = h[:, : self.num_prefix_tokens]
            patch = h[:, self.num_prefix_tokens :]
            flat = patch.reshape(-1, patch.shape[-1])  # [B*T, dim]
            code = self.basis.encode(flat)  # 상수

        if self.residual:  # grad 경로 시작(양쪽 모두)
            patch_plus = (flat + self.basis.delta(code, self.gain)).view_as(patch)
        else:
            patch_plus = self.basis.absolute(code, self.gain).view_as(patch)  # h를 통째로 대체
        h_plus = torch.cat([prefix, patch_plus], dim=1)

        for blk in backbone.blocks[self.hook_block + 1 :]:  # grad 통과(weight는 frozen)
            h_plus = blk(h_plus)
        h_plus = backbone.norm(h_plus)

        feat = backbone.forward_head(h_plus, pre_logits=True)  # 대조학습 거리를 재는 h'
        logits = backbone.head(feat)
        return logits, feat

    def reset(self) -> None:
        """gain을 1로 되돌려 무개입 상태로 만든다. 도메인 전환·시드 반복 사이에 부른다."""
        with torch.no_grad():
            self.gain.fill_(1.0)
