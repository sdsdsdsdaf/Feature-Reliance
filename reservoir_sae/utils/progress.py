from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


IS_TTY = sys.stdout.isatty()


def log(message: str) -> None:
    print(message, flush=True)


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    log(f"[start] {name}")
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        log(f"[failed] {name} ({elapsed:.1f}s)")
        raise
    else:
        elapsed = time.perf_counter() - start
        log(f"[done] {name} ({elapsed:.1f}s)")


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, log_every: int = 10) -> Iterator[T]:
    if IS_TTY:
        yield from tqdm(iterable, desc=desc, total=total, file=sys.stdout, dynamic_ncols=True)
        return

    for idx, item in enumerate(iterable, start=1):
        yield item
        if total is not None and (idx == 1 or idx == total or idx % max(1, log_every) == 0):
            log(f"[progress] {desc}: {idx}/{total}")
