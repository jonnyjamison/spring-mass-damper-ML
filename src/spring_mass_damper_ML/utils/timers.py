import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        dur = time.perf_counter() - start
        print(f"[TIMER] {name}: {dur:.3f}s")