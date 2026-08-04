# Contract Brief — Phase 0 (Prep)

> [Plans.md](../Plans.md) T0.1 · T0.2. 공통 규약은 [README.md](README.md).

---

## T0.1 — 데이터 수급 · loader

**목적**: 실험 1~8이 쓰는 전 데이터셋을 **하나의 팩토리**로 열 수 있게 한다. 실험 코드가 데이터셋별 디렉토리 구조를 아는 일이 없어야 한다.

### 진입점

```python
# Utils/datasets.py
def build_dataset(name: str, split: str = "test", *,
                  root: str = "data",
                  transform=None,
                  corruption: str | None = None,
                  severity: int | None = None) -> tuple[Dataset, DatasetMeta]:
    """이름 하나로 아무 데이터셋이나 열어주는 단일 창구.
    실험 코드가 데이터셋별 디렉토리 구조·라벨 파일 형식을 알 필요가 없게 만든다.
    - name: 아래 목록 중 하나
    - split: "train"/"val"/"test", ImageNet-C는 "hp"/"report", ImageNet-9는 variant 이름
    - corruption/severity: ImageNet-C 전용. 다른 데이터셋에 주면 ValueError
    반환: (Dataset, 그 데이터셋이 뭔지 설명하는 meta)"""

@dataclass
class DatasetMeta:
    """열린 데이터셋의 정체를 담은 카드. 실험 코드는 이걸 보고 분기한다."""
    name: str
    num_classes: int
    num_samples: int
    has_group_labels: bool      # True면 __getitem__이 (img, y, group) 3-튜플을 준다
    num_groups: int | None      # worst-group acc를 몇 개 그룹으로 나눠 재는지
    group_names: list[str] | None
    class_to_idx: dict[str, int]
    source: str                 # 실제로 읽은 경로 — 어느 사본을 썼는지 결과에 남기기 위함
```

`name ∈ {"imagenet", "imagenet-c", "waterbirds", "colored-mnist", "imagenet-r", "imagenet-a", "imagenet-sketch", "imagenet-9"}`

**Dataset의 `__getitem__` 반환 계약**: `(image, target)` 또는 group label이 있으면 `(image, target, group)`. `has_group_labels`가 결정한다. 실험 코드는 이 플래그로 분기한다.

### 데이터셋별 계약

#### `waterbirds` — 보유 (검증됨)

```
data/waterbirds/waterbird_complete95_forest2water2/
    metadata.csv
    001.Black_footed_Albatross/*.jpg   ... (200 class dirs)
```

`metadata.csv` 컬럼: `img_id, img_filename, y, split, place, place_filename` (11,788행, 검증됨)

- `y`: 0=landbird, 1=waterbird — **이게 분류 라벨** (200종 CUB 클래스가 아니라 **2-class 이진 문제**)
- `place`: 0=land, 1=water — 배경 (spurious 속성)
- `split`: 0=train(4,795) / 1=val(1,199) / 2=test(5,794)
- **group = `2*y + place`** → 4개

실측 그룹 분포:

| group | (y, place) | 의미 | n |
|---|---|---|---|
| 0 | (0,0) | landbird on land | 6,220 |
| 1 | (0,1) | landbird on water | 2,905 |
| 2 | (1,0) | **waterbird on land** | **831** |
| 3 | (1,1) | waterbird on water | 1,832 |

**group 2(831장)가 최소·최대상충 그룹이다.** worst-group accuracy는 사실상 이 그룹의 정확도이고, 실험 4A의 주 지표다. `img_filename`이 200개 CUB 종 디렉토리를 가리키지만 **종 라벨은 쓰지 않는다.**

**T1.3이 요구하는 것**: 실험 3은 `y`와 `place`를 **각각** 층화 변수로 쓴다. `group = 2*y + place`에서 `y = group // 2`, `place = group % 2`로 복원되므로 `(image, target, group)` 계약으로 충분하지만, **loader가 `split` 필터를 지원해야 한다** — ①정답 측정은 `split=2`(test, 그룹 상대적으로 균형), `c_k^train`은 `split=0`(train, 95% 편향)에서 잰다. `build_dataset("waterbirds", split="train"|"val"|"test")`로 노출한다.

#### `imagenet-c` — 보유

```
data/imagenet-c/<corruption>/<severity>/<wnid>/*.JPEG
```
`corruption` 15 test + 4 extra(`gaussian_blur`·`saturate`·`spatter`·`speckle_noise`), `severity ∈ 1..5`.
**HP 튜닝은 4 extra에서만, 보고는 15 test에서** — spec의 ImageNet-C 규약. loader가 `split="hp"` / `"report"`로 이 구분을 강제한다.

#### `colored-mnist` — 생성

```
data/colored-mnist/{train1,train2,test}.pt
```
각 `.pt` = `{"images": [N,2,14,14] float in [0,1], "labels": [N,1] float, "e": float}`.
`e`: train1=0.2, train2=0.1, test=0.9. 2-class. group = `2*label + color`(색 채널 index).
[scripts/download_shift_datasets.py](../../../scripts/download_shift_datasets.py)가 생성하며 IRM 구성이 검증돼 있다(색-라벨 일치율 0.795 / 0.898 / 0.100).

#### `imagenet-r` / `imagenet-a` / `imagenet-sketch` — 미보유

```
data/imagenet-{r,a,sketch}/<wnid>/*.jpg
```
`ImageFolder` + **wnid → ImageNet-1k 인덱스 매핑**이 필수. R/A는 200 클래스 부분집합이라 1000-way head 출력을 200개 열로 **마스킹**해야 정확도가 맞는다. 이 매핑을 `Utils/imagenet_subsets.py`에 상수로 둔다.

#### `imagenet-9` — 미보유

```
data/imagenet-9/<variant>/val/<class>/*.JPEG
```
variant = `original`·`mixed_rand`·`mixed_same`·`only_fg`·`no_fg` 등. 9-class coarse.
`build_dataset("imagenet-9", split="mixed_rand")` 식으로 variant를 split 인자로 받는다.

### DoD 검증 스크립트

```python
# tests/test_datasets.py
for name in AVAILABLE:                      # 미보유는 skip 표시
    ds, meta = build_dataset(name)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch[0].shape[0] == 4
    log(f"{name}: n={meta.num_samples} classes={meta.num_classes} "
        f"groups={meta.num_groups} src={meta.source}")
```

**미보유 데이터셋은 실패가 아니라 `skipped`로 기록한다.** `result.json`의 `unknown_fields`에 사유를 남긴다.

### 산출물

```json
{"result": {"datasets": {"waterbirds": {"status": "ok", "n": 11788, "classes": 2,
                                        "groups": 4, "group_counts": [6220,2905,831,1832]},
                         "imagenet-r": {"status": "missing", "reason": "egress blocked"}}}}
```

**판정 규칙**: `verdict="pass"` ⟺ Waterbirds·ImageNet-C·ColoredMNIST가 `ok`. 나머지는 미보유여도 MVP(실험 1~4A)가 돌아가므로 통과.

---

## T0.2 — lint/format baseline

**목적**: Phase M 신규 코드에서 실행 시점에야 터지는 버그(`F821` 미정의 이름 등)를 미리 잡는다.

### 범위 결정 (축소)

- **`black` 도입하지 않는다.** 기존 14,755 LOC 전면 재포맷 diff가 병렬 세션 작업과 충돌하고 `git blame`을 무의미하게 만든다. 논문 코드에서 스타일 통일의 실익이 대가보다 작다.
- **`ruff`만, `select = ["F"]`(pyflakes)로 좁게.** 스타일 규칙(E·W·I)은 켜지 않는다.
- **기존 코드는 건드리지 않는다.** 신규 경로만 대상.

### 산출물

```toml
# pyproject.toml (신규)
[tool.ruff]
line-length = 120
target-version = "py311"
extend-exclude = ["third_party", "outputs", "data", "*.ipynb"]

[tool.ruff.lint]
select = ["F"]        # pyflakes만: 미정의 이름·미사용 import·미사용 변수
ignore = ["F401"]     # __init__.py re-export 대비. 필요시 파일별 noqa
```

**검증**: `ruff check Model/ Utils/ scripts/ tests/`가 통과.

### 선행 조건

`pip install ruff`가 **PyPI 접속을 필요로 한다** — 현재 egress 차단으로 이 세션에서는 불가. T0.1의 데이터 다운로드와 같은 벽이다.

**판정 규칙**: `verdict="pass"` ⟺ `pyproject.toml`이 커밋되고 `ruff check`가 신규 경로에서 0 error.
아무 task도 T0.2에 의존하지 않으므로(Recommended) 막히면 건너뛴다.
