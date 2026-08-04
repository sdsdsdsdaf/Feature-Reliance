"""M4 계약 검증: weak/strong 증강, TwoCropTransform, timm 정규화 상수 일치."""

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from Utils.tta_transforms import TwoCropTransform, strong_transform, weak_transform


def _toy_image(size=256, seed=0):
    """재현 가능한 무작위 RGB PIL 이미지를 하나 만든다(빠른 unit test용)."""
    gen = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size, size, 3, generator=gen) * 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


class TestWeakStrongTransform:
    def test_weak_transform_returns_expected_shape_tensor(self):
        """weak_transform은 [3, size, size] 텐서를 반환한다."""
        img = _toy_image()
        out = weak_transform(size=224)(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_strong_transform_returns_expected_shape_tensor(self):
        """strong_transform은 [3, size, size] 텐서를 반환한다."""
        img = _toy_image()
        out = strong_transform(size=224)(img)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_weak_and_strong_views_differ_across_draws(self):
        """weak과 strong은 확률적이지만, 여러 draw에 걸쳐 평균적으로 서로 달라야 한다."""
        img = _toy_image()
        weak_fn = weak_transform(size=224)
        strong_fn = strong_transform(size=224)
        diffs = []
        for i in range(5):
            torch.manual_seed(i)
            w = weak_fn(img)
            torch.manual_seed(i)
            s = strong_fn(img)
            diffs.append((w - s).abs().sum().item())
        assert any(d > 1e-3 for d in diffs)


class TestNormalizationConstants:
    def test_normalization_matches_timm_resolved_config(self):
        """weak/strong의 정규화 상수는 timm resolve_model_data_config가 보고하는 값과 같아야 한다."""
        import timm
        from timm.data import resolve_model_data_config

        model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
        data_cfg = resolve_model_data_config(model)
        expected_mean = torch.tensor(data_cfg["mean"]).view(3, 1, 1)
        expected_std = torch.tensor(data_cfg["std"]).view(3, 1, 1)

        # 순백(255,255,255) 이미지는 normalize 후 (1 - mean) / std가 되어
        # 정규화 상수를 직접 역산할 수 있다.
        white = Image.new("RGB", (256, 256), color=(255, 255, 255))
        out = weak_transform(size=224)(white)
        recovered_mean = 1.0 - out.mean(dim=(1, 2)) * expected_std.view(3)
        torch.testing.assert_close(
            recovered_mean, expected_mean.view(3), rtol=0, atol=1e-4
        )


class TestTwoCropTransform:
    def test_returns_two_tensors_with_expected_shape(self):
        """TwoCropTransform.__call__은 (weak, strong) 두 텐서를 반환한다."""
        img = _toy_image()
        two_crop = TwoCropTransform(weak_transform(size=224), strong_transform(size=224))
        weak, strong = two_crop(img)
        assert isinstance(weak, torch.Tensor)
        assert isinstance(strong, torch.Tensor)
        assert weak.shape == (3, 224, 224)
        assert strong.shape == (3, 224, 224)

    def test_two_views_differ_across_draws(self):
        """TwoCropTransform이 만드는 두 뷰는 여러 draw에 걸쳐 서로 달라야 한다."""
        img = _toy_image()
        two_crop = TwoCropTransform(weak_transform(size=224), strong_transform(size=224))
        diffs = []
        for i in range(5):
            torch.manual_seed(i)
            weak, strong = two_crop(img)
            diffs.append((weak - strong).abs().sum().item())
        assert any(d > 1e-3 for d in diffs)


class _ToyDataset(Dataset):
    """DataLoader 통합 테스트용 최소 Dataset: 고정된 개수의 무작위 이미지를 반환."""

    def __init__(self, n=6, transform=None):
        """n개의 서로 다른 시드로 생성한 toy 이미지를 보관하고, transform을 적용해 반환한다."""
        self.images = [_toy_image(seed=i) for i in range(n)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return self.transform(img) if self.transform else img


class TestDataLoaderIntegration:
    def test_dataloader_yields_batched_pairs_of_expected_shape(self):
        """TwoCropTransform을 쓰는 DataLoader는 [(B,3,H,W), (B,3,H,W)] 형태의 배치 쌍을 낸다."""
        two_crop = TwoCropTransform(weak_transform(size=224), strong_transform(size=224))
        dataset = _ToyDataset(n=6, transform=two_crop)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        batch_weak, batch_strong = next(iter(loader))
        assert batch_weak.shape == (3, 3, 224, 224)
        assert batch_strong.shape == (3, 3, 224, 224)
