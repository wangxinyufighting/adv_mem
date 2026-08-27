import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def map_rewards(score: Callable[..., Any], *columns: Iterable[Any]) -> list[Any]:
    """Evaluate rollout rewards concurrently while preserving input order."""
    with ThreadPoolExecutor(
        max_workers=int(os.getenv("REWARD_MAX_WORKERS", "4"))
    ) as pool:
        return list(pool.map(lambda values: score(*values), zip(*columns, strict=True)))
