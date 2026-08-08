"""개입 진단의 캐시+꼬리 경로가 구 훅 경로와 같은 값을 내는지 검증한다.

바꾼 이유: 예전 구조는 배치마다 clean 1회 + 개입 105회 = 106번의 **전체** ViT forward 를
돌았고, 그중 blocks[0..target_block] 은 105번이 전부 같은 값을 다시 계산하는 낭비였다.
개입은 target_block 출력을 갈아끼우므로 앞부분은 한 번만 계산하면 된다.

리팩터링의 위험은 "빨라졌는데 값이 달라지는" 것이라, 구 구현을 기준으로 고정한다.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Model.SAE import VanillaL1SAE
from Utils.SAE_plot_utils import (
    cache_block_activations,
    evaluate_latent_intervention_curve,
    run_model_with_latent_intervention,
    run_vit_tail,
)

DIM, HIDDEN, TOKENS, N_IMG = 8, 32, 5, 12
TARGET_BLOCK = 1


class _Block(nn.Module):
    """토큰을 섞는 선형 블록. 항등이면 꼬리를 건너뛰어도 통과해버려 테스트가 무의미해진다."""

    def __init__(self, dim, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(dim, dim)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(dim, dim, generator=g) * 0.3)
            self.lin.bias.copy_(torch.randn(dim, generator=g) * 0.1)

    def forward(self, x):
        return torch.tanh(self.lin(x)) + x


class _ViT(nn.Module):
    """timm ViT 중 개입 경로가 실제로 건드리는 부분만 흉내낸다:
    blocks / norm / forward_head(pre_logits) / head."""

    def __init__(self, n_blocks=4, n_classes=6):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(DIM, seed=i) for i in range(n_blocks)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, n_classes)
        g = torch.Generator().manual_seed(99)
        self.embed = nn.Parameter(torch.randn(N_IMG, TOKENS, DIM, generator=g), requires_grad=False)

    def forward_head(self, x, pre_logits=False):
        feat = x[:, 0]  # CLS pooling
        return feat if pre_logits else self.head(feat)

    def forward(self, images):
        # images 는 인덱스 텐서로 쓴다 — 이미지 디코딩은 이 테스트의 관심사가 아니다
        x = self.embed[images.long().reshape(-1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(self.forward_head(x, pre_logits=True))


def _loader(batch=4):
    idx = torch.arange(N_IMG)
    for s in range(0, N_IMG, batch):
        chunk = idx[s : s + batch]
        yield chunk, chunk % 6


def _sae():
    torch.manual_seed(0)
    return VanillaL1SAE(DIM, HIDDEN, b_dec_init=torch.zeros(DIM), dec_bias_mode="zero")


@pytest.fixture
def setup():
    torch.manual_seed(1)
    model, sae = _ViT(), _sae()
    model.eval()
    stats = {"mean": torch.randn(1, DIM) * 0.5, "std": torch.rand(1, DIM) + 0.5}
    return model, sae, stats


def test_cached_tail_reproduces_a_full_forward(setup):
    """캐시한 활성에 꼬리만 태운 logits 가 전체 forward 와 같아야 한다."""
    model, _sae_, _stats = setup
    images = torch.arange(N_IMG)
    full = model(images)

    h, _ = cache_block_activations(model, _loader(), TARGET_BLOCK, "cpu", cache_dtype=torch.float32)
    tail = run_vit_tail(model, h, TARGET_BLOCK)
    assert torch.allclose(full, tail, atol=1e-5)


def test_cache_shape_and_labels(setup):
    model, _sae_, _stats = setup
    h, labels = cache_block_activations(model, _loader(), TARGET_BLOCK, "cpu", cache_dtype=torch.float32)
    assert h.shape == (N_IMG, TOKENS, DIM)
    assert labels.tolist() == (torch.arange(N_IMG) % 6).tolist()


def test_cache_respects_max_batches(setup):
    model, _sae_, _stats = setup
    h, labels = cache_block_activations(model, _loader(batch=4), TARGET_BLOCK, "cpu", max_batches=2, cache_dtype=torch.float32)
    assert h.shape[0] == 8 and labels.shape[0] == 8


def test_empty_loader_is_an_error_not_a_silent_empty_result(setup):
    model, _sae_, _stats = setup
    with pytest.raises(RuntimeError, match="활성이 하나도"):
        cache_block_activations(model, iter([]), TARGET_BLOCK, "cpu")


@pytest.mark.parametrize("scope", ["all", "patch", "cls"])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_cached_path_matches_the_hook_path(setup, scope, alpha, monkeypatch):
    """핵심 계약. 개입 결과가 구 훅 경로와 같아야 한다."""
    import Utils.SAE_plot_utils as P

    model, sae, stats = setup
    monkeypatch.setattr(P, "INTERVENTION_TOKEN_SCOPE", scope)
    latent_ids = [1, 5, 9]
    images = torch.arange(N_IMG)

    ref = run_model_with_latent_intervention(
        images, latent_ids, alpha, model, sae, stats, TARGET_BLOCK, scope, "cpu"
    )

    h, _ = cache_block_activations(model, _loader(), TARGET_BLOCK, "cpu", cache_dtype=torch.float32)
    edited = P._edit_block_output(h, sae, stats, latent_ids, alpha, scope)
    got = run_vit_tail(model, edited, TARGET_BLOCK)

    assert torch.allclose(ref, got, atol=1e-5), f"scope={scope} alpha={alpha}"


def test_curve_records_match_the_old_per_batch_computation(setup, monkeypatch):
    """집계까지 포함해 record 값이 구 구현과 일치해야 한다."""
    import Utils.SAE_plot_utils as P

    model, sae, stats = setup
    monkeypatch.setattr(P, "INTERVENTION_TOKEN_SCOPE", "all")
    specs = {("fam", "cue"): [1, 5, 9], ("fam", "rand"): [2, 3]}
    alphas = [0.0, 1.0]

    records = evaluate_latent_intervention_curve(
        model, sae, stats, specs, list(_loader()), TARGET_BLOCK, "cpu",
        alphas=alphas, label_to_imagenet=None, chunk_images=5,
    )
    assert len(records) == len(specs) * len(alphas)

    # 구 구현과 동일한 방식으로 기준값을 직접 계산한다
    images, labels = torch.arange(N_IMG), torch.arange(N_IMG) % 6
    clean = model(images)
    for rec in records:
        ids = specs[(rec["family"], rec["baseline"])]
        ref = run_model_with_latent_intervention(
            images, ids, rec["alpha"], model, sae, stats, TARGET_BLOCK, "all", "cpu"
        )
        assert rec["n"] == N_IMG
        assert rec["clean_acc"] == pytest.approx((clean.argmax(1) == labels).float().mean().item(), abs=1e-6)
        assert rec["intervention_acc"] == pytest.approx((ref.argmax(1) == labels).float().mean().item(), abs=1e-6)
        assert rec["logit_l1"] == pytest.approx((ref - clean).abs().mean(dim=1).mean().item(), rel=1e-4)


def test_alpha_one_is_not_assumed_to_be_a_no_op(setup, monkeypatch):
    """진단 경로는 residual 이 아니라 '통째 대체'라 alpha=1 이어도 재구성 오차가 남는다.

    이걸 no-op 으로 착각하면 재구성 바닥을 0 으로 잘못 읽게 된다."""
    import Utils.SAE_plot_utils as P

    model, sae, stats = setup
    monkeypatch.setattr(P, "INTERVENTION_TOKEN_SCOPE", "all")
    images = torch.arange(N_IMG)
    clean = model(images)
    at_one = run_model_with_latent_intervention(images, [1, 2], 1.0, model, sae, stats, TARGET_BLOCK, "all", "cpu")
    assert not torch.allclose(clean, at_one, atol=1e-4)


# ---------------------------------------------------------------------------
# token_scope 배선 (2026-08-07)
#
# 예전엔 개입 범위가 모듈 상수 "all" 로 박혀 있어서, SAE 를 patch 로 학습해놓고
# 진단에서는 CLS 까지 재구성으로 갈아끼웠다. 그 토큰이 분류 head 가 pooling 해서 쓰는
# 자리라, 개입과 무관한 재구성 대가가 얹혀 cue 효과의 SNR 을 나쁘게 보이게 했다.
# ---------------------------------------------------------------------------


def test_token_scope_argument_overrides_the_module_fallback(setup, monkeypatch):
    import Utils.SAE_plot_utils as P

    model, sae, stats = setup
    monkeypatch.setattr(P, "INTERVENTION_TOKEN_SCOPE", "all")   # 폴백은 all 로 둔 채
    specs = {("fam", "cue"): [1, 5]}

    as_patch = evaluate_latent_intervention_curve(
        model, sae, stats, specs, list(_loader()), TARGET_BLOCK, "cpu",
        alphas=[0.0], label_to_imagenet=None, chunk_images=6, token_scope="patch",
    )
    as_all = evaluate_latent_intervention_curve(
        model, sae, stats, specs, list(_loader()), TARGET_BLOCK, "cpu",
        alphas=[0.0], label_to_imagenet=None, chunk_images=6, token_scope="all",
    )
    # CLS 를 건드리느냐 마느냐는 실제로 값을 바꿔야 한다 — 안 바뀌면 인자가 안 먹은 것이다
    assert as_patch[0]["js_divergence"] != pytest.approx(as_all[0]["js_divergence"], rel=1e-6)


def test_none_falls_back_to_the_module_constant(setup, monkeypatch):
    import Utils.SAE_plot_utils as P

    model, sae, stats = setup
    specs = {("fam", "cue"): [1, 5]}

    def run(scope_arg, fallback):
        monkeypatch.setattr(P, "INTERVENTION_TOKEN_SCOPE", fallback)
        return evaluate_latent_intervention_curve(
            model, sae, stats, specs, list(_loader()), TARGET_BLOCK, "cpu",
            alphas=[0.0], label_to_imagenet=None, chunk_images=6, token_scope=scope_arg,
        )[0]["js_divergence"]

    assert run(None, "patch") == pytest.approx(run("patch", "all"), rel=1e-9)


def test_save_trial_plots_passes_the_config_scope():
    """배선이 빠지면 그 경로만 조용히 폴백('all')로 돌아간다."""
    import ast
    import inspect

    import Utils.SAE_plot_utils as P

    tree = ast.parse(inspect.getsource(P.save_trial_plots))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "evaluate_latent_intervention_curve"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert "token_scope" in kw, "save_trial_plots 가 token_scope 를 안 넘긴다"
    assert ast.unparse(kw["token_scope"]) == "config.hook.token_scope"
