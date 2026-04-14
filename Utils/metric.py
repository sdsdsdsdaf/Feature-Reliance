import numpy as np
import cv2 
import numpy
import torch.nn.functional as F
import torch
from torch import Tensor

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# TODO Simtorch로 계산 결과 동일하게 나오는지 검증

def linear_cka(X:Tensor, Y:Tensor, eps=1e-12) -> Tensor:
    X = X.reshape(X.size(0), -1)
    Y = Y.reshape(Y.size(0), -1)

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X.T @ Y).pow(2).sum()
    hsic_xx = (X.T @ X).pow(2).sum()
    hsic_yy = (Y.T @ Y).pow(2).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps))

def relative_accuracy(
    acc_sup: float,
    acc_original: float,
    acc_chance: float = None,
    num_class: int = None,
    eps: float = 1e-8
):
    if acc_chance is None and num_class is None:
        raise ValueError("Provide either acc_chance or num_class.")

    if acc_chance is None:
        acc_chance = 1.0 / num_class

    denom = acc_original - acc_chance

    if abs(denom) < eps:
        return 0.0

    rel_acc = (acc_sup - acc_chance) / denom

    return rel_acc

def kl_divergence(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """
    Compute KL(P || Q)

    Args:
        logits_p: reference distribution logits
        logits_q: comparison distribution logits
    """
    p = F.softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean")


def js_divergence(logits_p: Tensor, logits_q: Tensor) -> Tensor:
    """
    Compute JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    m = 0.5 * (p + q)

    log_m = torch.log(m.clamp_min(1e-12))
    kl_pm = F.kl_div(log_m, p, reduction="batchmean")
    kl_qm = F.kl_div(log_m, q, reduction="batchmean")

    return 0.5 * (kl_pm + kl_qm)


# Feature Suppression 측정 함수

def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 grayscale in [0, 1].
    Accepts HxW or HxWxC.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    gray = gray.astype(np.float32)

    # Normalize to [0, 1] if input looks like uint8 range
    if gray.max() > 1.0:
        gray = gray / 255.0

    return gray


def compute_local_variance(gray: np.ndarray, window_size: int = 11) -> float:
    """
    Mean local variance over non-overlapping windows.
    Higher means more local texture/detail.
    """
    h, w = gray.shape
    vals = []

    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            patch = gray[y:y + window_size, x:x + window_size]
            vals.append(float(np.var(patch)))

    if not vals:
        return 0.0
    return float(np.mean(vals))


def local_variance_ratio(
    original: np.ndarray,
    transformed: np.ndarray,
    window_size: int = 11,
) -> float:
    """
    LV ratio from the paper style:
    min(1, LV(transformed) / LV(original))
    Lower means stronger texture suppression.
    """
    x = to_grayscale_float(original)
    y = to_grayscale_float(transformed)

    lv_x = compute_local_variance(x, window_size=window_size)
    lv_y = compute_local_variance(y, window_size=window_size)

    eps = 1e-8
    return float(min(1.0, lv_y / (lv_x + eps)))


def compute_high_frequency_energy(gray: np.ndarray, radius: int = 11) -> float:
    """
    High-frequency energy fraction using 2D FFT.
    Higher means more fine texture / high-frequency content.
    """
    h, w = gray.shape
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    high_freq_mask = dist > radius

    total_energy = float(power.sum())
    hf_energy = float(power[high_freq_mask].sum())

    eps = 1e-8
    return hf_energy / (total_energy + eps)


def high_frequency_energy_ratio(
    original: np.ndarray,
    transformed: np.ndarray,
    radius: int = 11,
) -> float:
    """
    HFE ratio from the paper style:
    min(1, HFE(transformed) / HFE(original))
    Lower means stronger texture suppression.
    """
    x = to_grayscale_float(original)
    y = to_grayscale_float(transformed)

    hfe_x = compute_high_frequency_energy(x, radius=radius)
    hfe_y = compute_high_frequency_energy(y, radius=radius)

    eps = 1e-8
    return float(min(1.0, hfe_y / (hfe_x + eps)))


def compute_sobel_magnitude(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel gradient magnitude for ESSIM.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag


def edge_ssim(
    original: np.ndarray,
    transformed: np.ndarray,
    sobel_ksize: int = 3,
) -> float:
    """
    ESSIM = SSIM on Sobel magnitude maps.
    Higher means better shape / edge preservation.
    """
    x = to_grayscale_float(original)
    y = to_grayscale_float(transformed)

    sx = compute_sobel_magnitude(x, ksize=sobel_ksize)
    sy = compute_sobel_magnitude(y, ksize=sobel_ksize)

    data_range = max(float(sx.max() - sx.min()), float(sy.max() - sy.min()), 1e-8)
    return float(ssim(sx, sy, data_range=data_range))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    Safe Pearson correlation for flattened arrays.
    Returns 0 if variance is too small.
    """
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    a_std = float(a.std())
    b_std = float(b.std())

    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0

    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def gradient_correlation(
    original: np.ndarray,
    transformed: np.ndarray,
    ksize: int = 3,
) -> float:
    """
    GC = 0.5 * (corr(gx_x, gx_y) + corr(gy_x, gy_y))
    Higher means better shape preservation.
    """
    x = to_grayscale_float(original)
    y = to_grayscale_float(transformed)

    gx_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=ksize)
    gy_x = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=ksize)

    gx_y = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=ksize)
    gy_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=ksize)

    corr_x = _safe_corr(gx_x, gx_y)
    corr_y = _safe_corr(gy_x, gy_y)

    return float(0.5 * (corr_x + corr_y))


def evaluate_feature_metrics(
    original: np.ndarray,
    transformed: np.ndarray,
    window_size: int = 11,
    radius: int = 11,
    sobel_ksize: int = 3,
) -> dict:
    """
    Returns all four metrics and aggregated texture/shape scores.

    Interpretation:
    - LV_ratio, HFE_ratio: lower => stronger texture suppression
    - ESSIM, GC: higher => better shape preservation
    - texture_score = mean(LV_ratio, HFE_ratio)
    - shape_score = mean(ESSIM, GC)
    """
    lv = local_variance_ratio(original, transformed, window_size=window_size)
    hfe = high_frequency_energy_ratio(original, transformed, radius=radius)
    essim = edge_ssim(original, transformed, sobel_ksize=sobel_ksize)
    gc = gradient_correlation(original, transformed, ksize=sobel_ksize)

    texture_score = 0.5 * (lv + hfe)
    shape_score = 0.5 * (essim + gc)

    return {
        "LV_ratio": lv,
        "HFE_ratio": hfe,
        "ESSIM": essim,
        "GC": gc,
        "texture_score": texture_score,
        "shape_score": shape_score,
    }
    
def compute_dataset_metrics(dataset, base_transform=None, transform=None, max_samples=None, desc=None):
    results = {
        "LV_ratio": [],
        "HFE_ratio": [],
        "ESSIM": [],
        "GC": [],
        "texture_score": [],
        "shape_score": [],
    }
    if max_samples is None:
        max_samples = 1e9
    
    for i in tqdm(range(min(len(dataset), max_samples)), desc=desc):
        img, _ = dataset[i]
        original = img.copy()
        # 공통 preprocessing
        if base_transform is not None:
            original = base_transform(img.copy())            # HWC
        transformed = transform(original.copy())             # HWC

        # metric 계산
        metrics = evaluate_feature_metrics(original, transformed)

        for k in results:
            results[k].append(metrics[k])

    # 평균
    summary = {k: float(np.mean(v)) for k, v in results.items()}

    return summary

if __name__ == "__main__":

    logits_p = torch.randn(32, 10)
    logits_q = torch.randn(32, 10)

    # 1. KL identity test: KL(P || P) == 0
    kl_same = kl_divergence(logits_p, logits_p)
    print("KL(P || P):", kl_same.item())

    # 2. JS identity test: JS(P, P) == 0
    js_same = js_divergence(logits_p, logits_p)
    print("JS(P, P):", js_same.item())

    # 3. KL asymmetry test: KL(P || Q) != KL(Q || P)
    kl_pq = kl_divergence(logits_p, logits_q)
    kl_qp = kl_divergence(logits_q, logits_p)
    print("KL(P || Q):", kl_pq.item())
    print("KL(Q || P):", kl_qp.item())

    # 4. JS symmetry test: JS(P, Q) == JS(Q, P)
    js_pq = js_divergence(logits_p, logits_q)
    js_qp = js_divergence(logits_q, logits_p)
    print("JS(P, Q):", js_pq.item())
    print("JS(Q, P):", js_qp.item())

    # 5. Extreme case test
    logits_a = torch.tensor([[10.0, -10.0, -10.0]])
    logits_b = torch.tensor([[-10.0, 10.0, -10.0]])

    print("KL extreme:", kl_divergence(logits_a, logits_b).item())
    print("JS extreme:", js_divergence(logits_a, logits_b).item())
    print()

    # CKA 검증
    # TODO 후에 Simtorch와의 결과 비교
    
    X = torch.randn(1024, 128)
    Y_rand = torch.randn(1024, 128)

    cka_rand_xy = linear_cka(X, Y_rand)
    cka_rand_yx = linear_cka(Y_rand, X)

    A = torch.randn(128, 128)
    Y_lin = X @ A
    cka_lin = linear_cka(X, Y_lin)

    perm = torch.randperm(X.size(0))
    Y_perm = Y_lin[perm]
    cka_perm = linear_cka(X, Y_perm)

    print("CKA(X, X):", linear_cka(X, X))
    print("CKA random X,Y:", cka_rand_xy)
    print("CKA random Y,X:", cka_rand_yx)
    print("Random symmetry diff:", abs(cka_rand_xy - cka_rand_yx))
    print("CKA linear transformed:", cka_lin)
    print("CKA permuted:", cka_perm)