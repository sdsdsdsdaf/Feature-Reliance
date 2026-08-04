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

#### `colored-mnist` — 보유 (생성 완료, 실측 검증됨)

```
data/colored_mnist/{train1,train2,test}.pt        ← 디렉토리는 언더스코어
```

**정본 생성기는 [scripts/data/make_colored_mnist.py](../../../scripts/data/make_colored_mnist.py)다.** (`scripts/download_shift_datasets.py`에도 ColoredMNIST 생성 경로가 있지만 규격이 다르다 — 디스크의 데이터는 전자가 만든 것이므로 **전자를 따른다**. 후자의 ColoredMNIST 분기는 쓰지 않는다.)

실측 구조 (`torch.load`로 확인):

```python
{"images": uint8[N, 3, 28, 28], "labels": int64[N]}     # "e" 키 없음
# N: train1=25000, train2=25000, test=10000
```

- **3채널 28×28 uint8.** IRM 원본의 `2×14×14`가 아니다 — ViT-B/16(224×224×3) 파이프라인에 리사이즈로 바로 붙는 쪽을 택한 것.
- **`e`(환경) 키가 파일에 없다.** 환경은 파일명(`train1`/`train2`/`test`)이 곧 식별자다.
- **color는 채널에서 복원한다**: `R` 채널에 픽셀이 있으면 `color=0`, `G` 채널이면 `color=1` (`B`는 항상 0).
  `color = (images[:, 1].sum((1,2)) > 0).long()` → `group = 2*label + color` (4 그룹).
- 생성 파라미터: 라벨 노이즈 0.25 (전 환경 공통), color flip = train1 **0.10** / train2 **0.20** / test **0.90**.
  → 색-라벨 일치율 ≈ **0.90 / 0.80 / 0.10**. test에서 상관이 뒤집히므로 색에 의존한 모델이 무너진다.
- **caveat(생성기 docstring에도 있음)**: 28×28 합성 숫자는 자연 이미지가 아니라 ImageNet-ViT+SAE 스택과 맞지 않는다. **별도 소형 트랙**으로 다루고 메인 파이프라인 drop-in으로 취급하지 않는다.

#### `imagenet-r` / `imagenet-a` — 보유 (검증됨) / `imagenet-sketch` — 미보유

```
data/imagenet-r/imagenet-r/<wnid>/*.jpg        ← 한 겹 더 중첩돼 있다
data/imagenet-a/imagenet-a/<wnid>/*.jpg        ← 동일
data/imagenet-sketch/                          ← 비어 있음 (Google Drive 쿼터 실패)
```

**중첩 주의**: 배포 tarball이 자기 이름의 디렉토리를 품고 있어 `data/imagenet-r/imagenet-r/`가 된다. 루트를 하드코딩하지 말고 **`<root>/<name>/` 이 있으면 한 겹 내려가는 탐색**을 넣는다(sketch가 나중에 들어올 때 구조가 다를 수 있다).

각각 wnid 디렉토리 200개(+README 1). `ImageFolder` + **wnid → ImageNet-1k 인덱스 매핑**이 필수. R/A는 200 클래스 부분집합이라 1000-way head 출력을 200개 열로 **마스킹**해야 정확도가 맞는다. 이 매핑을 `Utils/imagenet_subsets.py`에 상수로 둔다 — **디렉토리명을 정렬한 순서가 아니라 실제 wnid 목록에서 유도**한다.

`imagenet-sketch`는 `status="missing"`으로 기록하고 넘어간다. 실험 5의 natural regime에서만 쓰이므로 MVP를 막지 않는다.

#### `imagenet-9` — 보유 (검증됨)

```
data/imagenet-9/bg_challenge/<variant>/val/<class>/*.JPEG    ← bg_challenge 한 겹 더
```

실측 variant: `original`, `mixed_same`, `mixed_rand`, `mixed_next`, `only_fg`, `no_fg`, `only_bg_b`, `only_bg_t`, `fg_mask`. 9-class coarse (`00_dog` … `08_...`).
`build_dataset("imagenet-9", split="mixed_rand")` 식으로 variant를 split 인자로 받는다. `fg_mask`는 이미지가 아니라 마스크라 분류용 split 목록에서 제외한다.

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

### 선행 조건 — 해소됨 (2026-08-05 실측)

`pip install ruff`는 **가능하다**(PyPI 도달 확인, `ruff-0.16.1` 설치 가능). 이전 판의 "egress 차단" 기술은 폐기한다.
현재 `ruff`·`black` 모두 **미설치**, `pytest 9.0.2` 설치됨, Python 3.11.14.

**판정 규칙**: `verdict="pass"` ⟺ `pyproject.toml`이 커밋되고 `ruff check`가 신규 경로에서 0 error.
아무 task도 T0.2에 의존하지 않으므로(Recommended) 막히면 건너뛴다.

> Plans.md의 T0.2 DoD는 `ruff check`·`black --check` 통과 **또는** 설정 파일 커밋으로 돼 있다. 위 축소 결정에 따라 **`black`은 도입하지 않고** "설정 파일 커밋" 갈래로 충족한다.
