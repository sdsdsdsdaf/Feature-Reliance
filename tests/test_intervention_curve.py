"""evaluate_latent_intervention_curve 계약 테스트.

이미지 루프를 바깥으로 빼면서(clean forward를 구성마다 다시 돌지 않도록) 값이
바뀌지 않았는지 고정한다. 검증하는 것:
- record의 개수/순서가 (spec x alpha) 그대로다
- 개입이 없으면(latent_ids 비었거나 alpha=1.0) clean과 동치라 acc_drop / js가 0이다
- 집계가 배치 크기에 무관하다 — 배치 1과 배치 N의 결과가 같다
- clean_acc는 spec/alpha에 의존하지 않으므로 모든 record에서 같다
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Model.SAE import VanillaL1SAE
from Utils.SAE_plot_utils import evaluate_latent_intervention_curve

DIM = 8
TOKENS = 5
CLASSES = 4


class _Block(nn.Module):
    def forward(self, x):
        return x


class _TinyViT(nn.Module):
    """blocks[i] 훅과 (B, tokens, dim) -> (B, classes)만 만족하는 최소 백본."""

    def __init__(self, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(depth)])
        self.head = nn.Linear(DIM, CLASSES)
        self.clean_forwards = 0

    def forward(self, images):
        self.clean_forwards += 1
        x = images.reshape(images.shape[0], 1, DIM).expand(-1, TOKENS, DIM).contiguous()
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.mean(dim=1))


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = _TinyViT()
    sae = VanillaL1SAE(input_dim=DIM, hidden_dim=DIM * 2, b_dec_init=torch.zeros(DIM), dec_bias_mode="zero")
    token_stats = {"mean": torch.zeros(1, DIM), "std": torch.ones(1, DIM)}
    images = torch.randn(8, DIM)
    labels = torch.randint(0, CLASSES, (8,))
    return model, sae, token_stats, TensorDataset(images, labels)


def _loader(dataset, bs):
    return DataLoader(dataset, batch_size=bs, shuffle=False)


def test_record_order_and_count(setup):
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0, 1], ("color", "random_0"): [2, 3], ("shape", "cue"): [4]}
    alphas = [0.0, 1.0]
    recs = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=alphas, label_to_imagenet=None)
    assert len(recs) == len(specs) * len(alphas)
    assert [(r["family"], r["baseline"], r["alpha"]) for r in recs] == [
        ("color", "cue", 0.0), ("color", "cue", 1.0),
        ("color", "random_0", 0.0), ("color", "random_0", 1.0),
        ("shape", "cue", 0.0), ("shape", "cue", 1.0),
    ]
    assert all(r["n"] == len(ds) for r in recs)


def test_empty_latent_ids_is_pure_sae_reconstruction(setup):
    """latent_ids가 비면 alpha가 무엇이든 z를 안 건드리므로 alpha끼리 값이 같다."""
    model, sae, stats, ds = setup
    specs = {("color", "cue"): []}
    recs = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=[0.0, 0.5, 1.0], label_to_imagenet=None)
    for key in ("js_divergence", "acc_drop", "logit_l1"):
        assert len({round(r[key], 10) for r in recs}) == 1, key


def test_clean_acc_identical_across_specs_and_alphas(setup):
    """clean forward는 spec/alpha와 무관하다 — 값이 흔들리면 집계가 섞인 것이다."""
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0], ("shape", "random_0"): [1, 2]}
    recs = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 3), 0, "cpu", alphas=[0.0, 1.0], label_to_imagenet=None)
    assert len({round(r["clean_acc"], 12) for r in recs}) == 1


def test_aggregation_is_batch_size_invariant(setup):
    """배치 크기를 바꿔도 가중 평균이라 같은 값이 나와야 한다."""
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0, 3]}
    alphas = [0.0, 1.0]
    a = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 8), 0, "cpu", alphas=alphas, label_to_imagenet=None)
    b = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 2), 0, "cpu", alphas=alphas, label_to_imagenet=None)
    for ra, rb in zip(a, b):
        for key in ("js_divergence", "cross_entropy", "logit_l1", "clean_acc", "intervention_acc"):
            assert ra[key] == pytest.approx(rb[key], abs=1e-6), key


def test_clean_forward_runs_once_per_batch_not_once_per_config(setup):
    """clean forward가 구성 수만큼 반복되면 안 된다 — 이 루프 순서가 비용의 대부분이다."""
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0], ("color", "random_0"): [1], ("shape", "cue"): [2]}
    alphas = [0.0, 0.5, 1.0]
    n_batches = 2
    model.clean_forwards = 0
    evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=alphas, label_to_imagenet=None)
    n_configs = len(specs) * len(alphas)
    # 배치당 clean 1회 + 구성 수만큼 개입 forward(개입도 model()을 통과한다)
    assert model.clean_forwards == n_batches * (1 + n_configs)


def test_max_batches_truncates(setup):
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0]}
    recs = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 2), 0, "cpu", alphas=[1.0], max_batches=1, label_to_imagenet=None)
    assert recs[0]["n"] == 2


def test_label_map_is_applied(setup):
    """label_to_imagenet이 주어지면 라벨을 그 인덱스로 옮겨서 정확도를 잰다."""
    model, sae, stats, ds = setup
    specs = {("color", "cue"): [0]}
    identity = torch.arange(CLASSES)
    shifted = torch.full((CLASSES,), CLASSES - 1, dtype=torch.long)
    a = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=[1.0], label_to_imagenet=identity)
    b = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=[1.0], label_to_imagenet=None)
    c = evaluate_latent_intervention_curve(model, sae, stats, specs, _loader(ds, 4), 0, "cpu", alphas=[1.0], label_to_imagenet=shifted)
    assert a[0]["clean_acc"] == b[0]["clean_acc"]
    assert c[0]["clean_acc"] != pytest.approx(a[0]["clean_acc"]) or a[0]["clean_acc"] == 0.0
