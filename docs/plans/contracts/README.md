# Contract Briefs

[Plans.md](../Plans.md)의 각 Phase task가 **무엇을 만들어야 하는지**를 고정하는 문서 모음.

| 문서 | 담당 task | 계약 형태 |
|---|---|---|
| [phase0.md](phase0.md) | T0.1 데이터 수급, T0.2 lint baseline | loader 시그니처 · 디렉토리 레이아웃 · 설정 파일 |
| [phaseM.md](phaseM.md) | M1~M8 방법 구현 | class/function 시그니처 · 동작 · 불변식 |
| [phase1.md](phase1.md) | T1.1~T1.4b 실험 1~4B (MVP) | 진입점 · 절차 · 산출 아티팩트 스키마 · 판정 규칙 |
| [phase2.md](phase2.md) | T2.1~T2.3 baseline·실험 5·6 | 동일 |
| [phase3.md](phase3.md) | T3.1~T3.2 실험 7·8 | 동일 |

## 문서 위계

```
experiment_plan.md   product contract (SSOT) — 무엇이 정답인가
        ↓
plan.md              설계 근거 — 왜 그렇게 하는가
        ↓
Plans.md             task ledger — 무엇을 언제 하는가 (DoD/Depends/Status)
        ↓
contracts/*.md       구현 계약 — 무엇을 만드는가 (시그니처/절차/산출물)
```

계약이 spec과 충돌하면 **spec이 이긴다.** 계약을 고치고 그 사유를 Plans.md `Spec delta`에 남긴다.

## 공통 규약

**시그니처 작성 규칙** — 시그니처만 나열하지 않는다. 블록만 읽고도 무엇을 하는 함수인지 알 수 있어야 한다.

- 모든 함수/메서드에 **무엇을 하는지 한 줄**을 붙인다. 타입만으로 자명해 보여도 붙인다(`normalize(h: Tensor) -> Tensor`는 무엇을 정규화하는지 알려주지 않는다).
- 클래스는 **`__init__`이 무엇을 받아 무엇을 들고 있게 되는지**를 쓴다. 보관하는 상태·기본값의 의미·학습 파라미터를 만드는지 여부까지.
- 인자 중 동작을 가르는 것(mode 스위치, 0이면 의미가 바뀌는 capacity 등)은 개별로 설명한다.
- 반환값이 dict/tuple이면 키·순서를 적는다.

```python
# 나쁨 — 무엇을 하는지 알 수 없다
def normalize(self, h: Tensor) -> Tensor: ...

# 좋음
def normalize(self, h: Tensor) -> Tensor:
    """백본 활성(raw) -> SAE 입력 공간. (h - token_mean) / token_std.
    SAE는 정규화된 토큰으로 학습됐으므로 raw h를 그냥 넣으면 안 된다."""
```

**산출물 경로**: `outputs/experiments/<task-id>/` 아래에 모은다.

```
outputs/experiments/T1.1/
    result.json          # 기계 판독용 — 아래 각 절의 스키마를 따른다
    result.md            # 사람이 읽는 표
    figures/*.png
    config.json          # 재현용: 커밋 해시·시드·전체 config 덤프
```

**모든 `result.json`의 공통 헤더**

```json
{
  "task_id": "T1.1",
  "git_commit": "<40자 해시>",
  "seed": 0,
  "started_at": "<RFC3339>",
  "finished_at": "<RFC3339>",
  "config": { },
  "verdict": "pass | fail | inconclusive",
  "verdict_reason": "<한 줄>",
  "result": { }
}
```

**`verdict`는 사람이 나중에 붙이는 게 아니라 스크립트가 판정 규칙에 따라 자동으로 채운다.** 각 절의 "판정 규칙"이 그 규칙이다. `inconclusive`는 데이터 부족·실행 실패처럼 판단 자체가 불가능할 때만 쓴다.

**unknown data contract**: 측정하지 못한 값은 `null`로 두고 `"unknown_fields"` 배열에 사유를 적는다. 0이나 기본값으로 채우지 않는다 (`not_observed != absent`).

**시드**: 모든 실험은 `seed ∈ {0, 1, 2}` 3회 반복이 기본이다. 단일 시드 결과는 `verdict`를 붙이지 않는다(`inconclusive`). 예외는 결정론적인 T0.1·T1.1·T1.2 항등성 검사.
