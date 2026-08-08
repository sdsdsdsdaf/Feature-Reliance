"""활성 정규화 규약(per_dim / scalar) 테스트.

왜 스칼라 분기를 뒀나: 차원별 나눗셈은 축마다 배율이 달라 raw 공간의 **방향**을
뒤튼다. 실측(2026-08-07, block10 patch 토큰 20만개)에서 차원별 std 가 1.38~12.16 로
8.8배 벌어져 있어 왜곡이 작지 않다. SAE 가 찾으려는 게 활성 공간의 방향(개념)이므로
축을 제각기 늘리면 그 방향이 보존되지 않는다.

반면 차원별 **평균 빼기**는 평행이동이라 거리·방향을 보존하므로 두 모드 모두 유지한다.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Utils.SAE_utils import fit_token_normalizer_streaming, normalize_tokens_chunked

DIM = 16


class _Block(nn.Module):
    def forward(self, x):
        return x


class _Backbone(nn.Module):
    """iter_token_batches_with_hook 이 기대하는 최소 인터페이스만 흉내낸다.

    커서로 앞으로 나아가야 한다 — 매번 앞부분만 돌려주면 통계가 그 몇 개로만 잡혀
    테스트가 조용히 무의미해진다(실제로 한 번 그렇게 짰다가 걸렸다)."""

    def __init__(self, tokens):
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])
        self._tokens = tokens
        self._pos = 0

    def forward(self, images):
        n = int(images.shape[0])
        chunk = self._tokens[self._pos : self._pos + n]
        self._pos += n
        return self.blocks[0](chunk)


def _loader(tokens, batch_images=4):
    n = tokens.shape[0]
    for start in range(0, n, batch_images):
        yield torch.zeros(min(batch_images, n - start), 3, 4, 4), torch.zeros(1)


def _fit(tokens, mode):
    """축별 분산이 크게 다른 토큰으로 통계를 적합한다."""
    model = _Backbone(tokens.unsqueeze(1))  # [N, 1, DIM] — token_scope="all" 로 그대로 흐르게
    return fit_token_normalizer_streaming(
        model, _loader(tokens), max_tokens=None, target_block=0,
        token_scope="all", device="cpu", norm_mode=mode,
    )[0]


@pytest.fixture
def tokens():
    torch.manual_seed(0)
    scale = torch.linspace(0.5, 8.0, DIM)          # 축마다 분산을 크게 다르게
    return torch.randn(512, DIM) * scale + torch.randn(DIM) * 3


def test_rejects_unknown_mode(tokens):
    with pytest.raises(ValueError, match="norm_mode"):
        _fit(tokens, "whitening")


def test_both_modes_keep_the_1xD_shape(tokens):
    """FrozenSAE 가 token_std.shape == (1, input_dim) 을 검증한다. 형태를 바꾸면 안 된다."""
    for mode in ("per_dim", "scalar"):
        s = _fit(tokens, mode)
        assert s["mean"].shape == (1, DIM), mode
        assert s["std"].shape == (1, DIM), mode


def test_scalar_mode_has_one_distinct_scale(tokens):
    s = _fit(tokens, "scalar")
    assert torch.allclose(s["std"], s["std"][0, 0].expand_as(s["std"])), "scalar 인데 원소가 다르다"
    p = _fit(tokens, "per_dim")
    assert p["std"].max() / p["std"].min() > 5, "per_dim 이 축별로 안 다르면 테스트가 무의미하다"


def test_scalar_mode_targets_unit_expected_norm(tokens):
    """std = sqrt(mean_d(var_d)) 이므로 정규화 후 E[||x||2] = sqrt(D) 가 되어야 한다.

    SAELens 의 'expected_average_only_in' 과 같은 규약이다."""
    s = _fit(tokens, "scalar")
    z = normalize_tokens_chunked(tokens.clone(), s, chunk_size=128)
    assert z.norm(dim=1).mean().item() == pytest.approx(DIM**0.5, rel=0.05)


def test_scalar_mode_preserves_direction_per_dim_does_not(tokens):
    """핵심 계약. 스칼라 배율은 방향을 보존하고 차원별 배율은 뒤튼다."""
    s = _fit(tokens, "scalar")
    p = _fit(tokens, "per_dim")
    centered = tokens - s["mean"]                       # 중심화는 두 모드 공통

    zs = normalize_tokens_chunked(tokens.clone(), s, chunk_size=128)
    zp = normalize_tokens_chunked(tokens.clone(), p, chunk_size=128)

    cos = torch.nn.functional.cosine_similarity
    # scalar: 중심화된 원본과 방향이 완전히 같다(양수 상수배)
    assert cos(centered, zs, dim=1).min().item() == pytest.approx(1.0, abs=1e-5)
    # per_dim: 방향이 틀어진다
    assert cos(centered, zp, dim=1).mean().item() < 0.99


def test_both_modes_center_per_dim(tokens):
    """평균 빼기는 평행이동이라 기하를 보존하므로 두 모드 모두 차원별로 한다."""
    for mode in ("per_dim", "scalar"):
        s = _fit(tokens, mode)
        z = normalize_tokens_chunked(tokens.clone(), s, chunk_size=128)
        assert z.mean(dim=0).abs().max().item() < 1e-4, mode


def test_default_is_per_dim_so_nothing_changes_silently():
    """dataclass 기본값은 기존 동작이어야 한다. 규약 전환은 SAE_validation.py 에서 명시적으로 한다."""
    from Utils.Config import SAETokenConfig

    assert SAETokenConfig().norm_mode == "per_dim"
