from threading import Lock
from typing import Any

from defender.models import MemoryBuilderRewardContext
from defender.reward import MemoryBuilderReward
from training.reward_pool import map_rewards


_REWARD: MemoryBuilderReward | None = None
_REWARD_LOCK = Lock()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point for Memory Builder GRPO."""
    global _REWARD
    if _REWARD is None:
        with _REWARD_LOCK:
            if _REWARD is None:
                _REWARD = MemoryBuilderReward.from_env()
    context = MemoryBuilderRewardContext.from_dict(extra_info)
    return _REWARD.evaluate(solution_str, context)


def compute_scores(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """verl batch reward entry point."""
    return map_rewards(
        compute_score,
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
    )
