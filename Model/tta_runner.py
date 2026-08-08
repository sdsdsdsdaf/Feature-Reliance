"""M8 — online TTA runner.

M2(`GainIntervention`)·M3(`AdaContrastLoss`+`MemoryBank`)·M4(`TwoCropTransform`)를
하나의 스트리밍 적응 루프로 배선한다. 학습 파라미터는 여전히 `gain` 하나뿐이다
(`intervention.trainable_parameters()`가 optimizer에 넘기는 전부).

평가 프로토콜은 두 가지를 병기한다:
- `online_acc`: 배치마다 그 배치 자신의 gain 갱신이 반영되기 '전' 예측을 누적한
  것 — Tent 이래의 표준. 이 배치의 갱신 결과로 이 배치를 채점하면 정보 누설이다.
- `final_acc`: 스트림(들)이 끝난 뒤, 그 시점의 gain으로 스트림을 다시 훑어
  재평가한 것(adapt-then-eval). 추가 갱신은 하지 않는다.

`distance_space`(h/z/logit)는 memory bank의 dim과 enqueue하는 텐서까지 같이
따라가야 한다 — 그러지 않으면 "z"/"logit" 조건이 껍데기만 남는다(phaseM.md M8
절 "⚠️ distance_space는 memory bank까지 따라가야 한다" 참조).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from Model.adacontrast import AdaContrastLoss, MemoryBank
from Utils.progress import pbar
from Model.intervention import GainIntervention

_VALID_DISTANCE_SPACES = ("h", "z", "logit")


@dataclass
class TTAConfig:
    """OnlineTTARunner 조립에 필요한 모든 축을 config 하나로 모은다.

    실험 4A·4B·5·6·7 전부 이 config만 바꿔 돌 수 있어야 한다(코드 분기 금지, M8 불변식).
    """

    lr: float = 1e-3
    batch_size: int = 64
    steps_per_batch: int = 1
    buffer: int | str = 16384
    """0=in-batch(실험 6 축) · 양의 정수=고정 capacity · "full"=`full_buffer_capacity` 사용."""
    full_buffer_capacity: Optional[int] = None
    """buffer="full"일 때만 쓰인다 — 스트림이 담을 총 샘플 수(호출자가 미리 계산해 넣는다).
    __init__ 시점에 preallocate하는 MemoryBank 구조상, "동적으로 계속 자란다"는 못 흉내내고
    이 값을 상한 capacity로 써서 실질적으로 안 밀려나게 만드는 방식으로 "full"을 구현한다."""
    passes: int = 1
    """1=순수 single-pass online(실험 6 축). >1이면 스트림 전체를 이 횟수만큼 반복한다."""
    knn_k: int = 10
    tau: float = 0.07
    anchor_mode: str = "off"  # off | l2 | ck — 실제 값 계산은 주입된 anchor 콜러블이 한다(M5)
    anchor_lam: float = 0.0
    distance_space: str = "h"  # h | z | logit — 실험 4B 축
    use_diversity: bool = False
    fvu_threshold: Optional[float] = None
    seed: int = 0

    def __post_init__(self) -> None:
        """distance_space·buffer 조합의 명백한 오설정을 생성 시점에 즉시 걸러낸다."""
        if self.distance_space not in _VALID_DISTANCE_SPACES:
            raise ValueError(f"distance_space는 {_VALID_DISTANCE_SPACES} 중 하나여야 한다: {self.distance_space!r}")
        if self.buffer == "full":
            if self.full_buffer_capacity is None:
                raise ValueError('buffer="full"이면 full_buffer_capacity를 함께 지정해야 한다')
        elif isinstance(self.buffer, int):
            if self.buffer < 0:
                raise ValueError(f"buffer는 0 이상이어야 한다: {self.buffer}")
        else:
            raise ValueError(f'buffer는 0 이상의 정수 또는 "full"이어야 한다: {self.buffer!r}')


def _pool_code_to_image(code: Tensor, batch_size: int) -> Tensor:
    """패치 토큰 단위 code `[B*T_patch, K]`를 이미지 단위로 mean-pool해 `[B, K]`로 만든다.

    distance_space="z"에서만 쓴다. z를 이미지를 "어떤 개념이 얼마나 켜졌나"의 가방으로
    보는 M8 계약의 결정(패치 축 mean-pool)을 그대로 구현한다."""
    total, dim = code.shape
    if total % batch_size != 0:
        raise ValueError(f"code의 행 수({total})가 batch_size({batch_size})로 나누어떨어지지 않는다")
    t = total // batch_size
    return code.view(batch_size, t, dim).mean(dim=1)


class OnlineTTARunner:
    def __init__(
        self,
        intervention: GainIntervention,
        loss_fn: AdaContrastLoss,
        anchor: Optional[Callable[[Tensor], Tensor]],
        config: TTAConfig,
        gate=None,
    ):
        """부품을 조립하고 optimizer·memory bank를 만든다.
        - optimizer = Adam(intervention.trainable_parameters(), lr=config.lr)
          ← trainable_parameters()가 [gain] 하나만 주므로 다른 건 절대 안 움직인다
        - bank = MemoryBank(dim=..., capacity=..., num_classes=loss_fn.num_classes).
          dim은 config.distance_space에 맞춰 h'(backbone.num_features) / z(basis.gain_dim)
          / logit(num_classes) 중 하나로 정한다. config.distance_space와
          loss_fn.distance_space가 다르면 즉시 ValueError(런타임 shape 오류로 늦게
          죽는 것보다 조립 시점에 잡는다).
        - anchor: gain을 1로 당기는 콜러블(M5, `GainAnchor.__call__`과 같은 서명)이거나
          None(M5가 아직 없거나 anchor_mode="off"일 때) — None이면 페널티는 0으로 취급한다.
        - gate=None이면 FVU 게이트 없이 항상 갱신한다. gate!=None은 M7이 T1.1로
          dropped돼 미구현이다(NotImplementedError) — 시그니처만 열어 둔다.
        - config.seed로 torch/random 난수를 고정한다."""
        if config.distance_space != loss_fn.distance_space:
            raise ValueError(
                f"config.distance_space({config.distance_space!r})와 "
                f"loss_fn.distance_space({loss_fn.distance_space!r})가 다르다 — "
                "memory bank가 query와 다른 공간의 negative를 들고 있게 된다"
            )
        if gate is not None:
            raise NotImplementedError(
                "FVU gate(M7)는 T1.1이 전제를 반증해 dropped됐다 — gate=None만 지원한다"
            )

        self.intervention = intervention
        self.loss_fn = loss_fn
        self.anchor = anchor
        self.config = config
        self.gate = gate

        torch.manual_seed(config.seed)
        random.seed(config.seed)

        device = intervention.gain.device
        feature_dim = getattr(intervention.backbone, "num_features", None)
        if feature_dim is None:
            raise ValueError("intervention.backbone에 num_features 속성이 없다 — h 공간 dim을 알 수 없다")
        basis_dim = intervention.basis.gain_dim
        num_classes = loss_fn.num_classes

        dim_by_space = {"h": feature_dim, "z": basis_dim, "logit": num_classes}
        bank_dim = dim_by_space[config.distance_space]

        capacity = config.full_buffer_capacity if config.buffer == "full" else int(config.buffer)

        self.bank = MemoryBank(dim=bank_dim, capacity=capacity, num_classes=num_classes, device=device)
        self.optimizer = torch.optim.Adam(intervention.trainable_parameters(), lr=config.lr)

    def _forward_for_space(self, x: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """distance_space에 맞춰 (logits, feat, key)를 반환한다.
        key는 contrastive 거리·bank enqueue에 쓰는 벡터 — "h"면 feat, "logit"이면 logits,
        "z"면 패치 코드를 이미지 단위로 mean-pool한 것(M2의 return_code=True 경로로 얻는다)."""
        if self.config.distance_space == "z":
            logits, feat, code = self.intervention(x, return_code=True)
            key = _pool_code_to_image(code, batch_size)
        else:
            logits, feat = self.intervention(x)
            key = feat if self.config.distance_space == "h" else logits
        return logits, feat, key

    def _anchor_penalty(self) -> Tensor:
        """anchor가 없으면(=off) 0 스칼라, 있으면 anchor(gain)의 결과를 그대로 반환한다."""
        if self.anchor is None:
            return torch.zeros((), device=self.intervention.gain.device)
        return self.anchor(self.intervention.gain)

    def _adapt_one_batch(self, x_weak: Tensor, x_strong: Tensor, y: Tensor):
        """한 배치를 (1) 적응 전 예측으로 채점 -> (2) gain 갱신 -> (3) bank enqueue 순서로 처리한다.
        평가가 갱신보다 먼저 일어나야 정보 누설이 없다(M8 계약의 핵심 불변식).
        반환: (pre_correct, batch_size, 마지막 step의 항별 손실 dict)."""
        batch_size = y.shape[0]

        # 1) 적응 전 예측 — 이 배치의 갱신이 반영되기 전 gain으로 채점한다.
        with torch.no_grad():
            logits_pre, _ = self.intervention(x_weak)
            pre_correct = int((logits_pre.argmax(dim=1) == y).sum().item())

        # 2) gain 갱신
        loss_log = None
        logits_w = key_w = None
        for _ in range(self.config.steps_per_batch):
            logits_w, feat_w, key_w = self._forward_for_space(x_weak, batch_size)
            logits_s, feat_s, key_s = self._forward_for_space(x_strong, batch_size)

            loss_kwargs = {}
            if self.config.distance_space == "z":
                loss_kwargs["z_weak"] = key_w
                loss_kwargs["z_strong"] = key_s

            losses = self.loss_fn(feat_w, logits_w, feat_s, logits_s, self.bank, **loss_kwargs)
            total = losses["total"] + self._anchor_penalty()

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            loss_log = {k: v.detach().item() for k, v in losses.items()}

        # 3) bank에 이번 배치의 weak-side key를 밀어넣는다(같은 distance_space).
        with torch.no_grad():
            probs_w = torch.softmax(logits_w.detach(), dim=1)
            self.bank.enqueue(key_w.detach(), probs_w)

        return pre_correct, batch_size, loss_log

    def run(self, stream) -> dict:
        """스트림을 처음부터 끝까지 config.passes회 흘리며 적응시킨다.
        `stream`은 (x_weak, x_strong, y) 3-튜플 배치를 내는 반복 가능한 객체여야 하고,
        passes>1 또는 final_acc 재평가를 위해 여러 번 순회할 수 있어야 한다
        (예: torch.utils.data.DataLoader, 또는 리스트).
        배치마다 '적응 전 예측'을 먼저 채점하고 그 다음에 gain을 갱신한다.
        반환: 정확도·궤적·게이트 통계가 담긴 결과 dict."""
        device = self.intervention.gain.device
        online_correct = 0
        n_seen = 0
        acc_vs_time: list[float] = []
        gain_trace: list[float] = []
        loss_trace: list[Optional[dict]] = []

        for _pass_idx in range(self.config.passes):
            for x_weak, x_strong, y in pbar(
                stream, desc=f"TTA adapt (pass {_pass_idx + 1}/{self.config.passes})", unit="batch", leave=False
            ):
                x_weak = x_weak.to(device)
                x_strong = x_strong.to(device)
                y = y.to(device)

                pre_correct, batch_size, loss_log = self._adapt_one_batch(x_weak, x_strong, y)

                online_correct += pre_correct
                n_seen += batch_size
                acc_vs_time.append(online_correct / n_seen)
                gain_trace.append((self.intervention.gain.detach() - 1.0).norm().item())
                loss_trace.append(loss_log)

        online_acc = online_correct / n_seen if n_seen > 0 else float("nan")

        # adapt-then-eval: 스트림 종료 후 gain으로 다시 훑어 재평가(추가 갱신 없음).
        final_correct = 0
        final_seen = 0
        with torch.no_grad():
            for x_weak, _x_strong, y in pbar(stream, desc="final eval", unit="batch", leave=False):
                x_weak = x_weak.to(device)
                y = y.to(device)
                logits, _feat = self.intervention(x_weak)
                final_correct += int((logits.argmax(dim=1) == y).sum().item())
                final_seen += y.shape[0]
        final_acc = final_correct / final_seen if final_seen > 0 else float("nan")

        return {
            "online_acc": online_acc,
            "final_acc": final_acc,
            "acc_vs_time": acc_vs_time,
            "gain_trace": gain_trace,
            "loss_trace": loss_trace,
            "gate_skipped": 0,  # gate=None만 지원하므로 항상 0(M7 dropped)
            "n_seen": n_seen,
        }
