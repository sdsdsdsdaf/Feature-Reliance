import numpy as np
import cv2 
import numpy
import torch.nn.functional as F
import torch
from torch import Tensor

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