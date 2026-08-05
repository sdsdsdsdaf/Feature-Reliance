"""오래 걸리는 루프에 진행 표시를 붙이는 공용 헬퍼.

기존 `Utils/SAE_utils.TQDM_KW`는 `disable=not sys.stdout.isatty()`라 **리다이렉트하면
바가 통째로 사라진다.** 이 프로젝트의 무거운 작업(실험 1의 75셀, `compute_c_k`,
TTA 스트림)은 대부분 백그라운드로 돌아 stdout이 파일이므로, 그 규약을 그대로 쓰면
정확히 진행을 보고 싶은 상황에서만 아무것도 안 보인다.

그래서 여기서는:
- **항상 표시한다**(tty 여부로 끄지 않는다).
- **stderr로 쓴다** — stdout은 결과 파싱용으로 비워 둔다.
- tty가 아니면 `mininterval`을 크게 잡아 `\\r` 갱신이 로그를 수천 줄로 채우지 않게 한다.

사용:
    from Utils.progress import pbar
    for batch in pbar(loader, desc="c_k 캐시"):
        ...
    with pbar(total=n, desc="셀") as bar:
        ...
        bar.update(1)
"""

from __future__ import annotations

import sys

from tqdm.auto import tqdm

# tty면 촘촘히, 리다이렉트면 드물게 — 로그 한 줄씩 남기는 용도.
_TTY = sys.stderr.isatty()
TQDM_KW = {
    "file": sys.stderr,
    "dynamic_ncols": True,
    "mininterval": 0.1 if _TTY else 30.0,
    "ascii": not _TTY,  # 로그 파일에 유니코드 블록이 깨져 남지 않게
}


def pbar(iterable=None, *, total=None, desc=None, unit="it", leave=True, **kw):
    """진행 표시가 붙은 tqdm을 만든다. 인자는 tqdm과 같고 기본값만 이 프로젝트에 맞춘다.

    - iterable을 주면 감싸서 반환하고, 없으면 수동 갱신용 tqdm 객체를 반환한다
      (`with pbar(total=...) as bar: bar.update(1)`).
    - tty가 아니어도 꺼지지 않는다 — 백그라운드 실행 로그에서 진행을 보기 위한 것이 이 모듈의 이유다.
    - desc는 무엇을 도는 중인지 한눈에 알 수 있게 반드시 준다."""
    kwargs = {**TQDM_KW, "total": total, "desc": desc, "unit": unit, "leave": leave, **kw}
    return tqdm(iterable, **kwargs)
