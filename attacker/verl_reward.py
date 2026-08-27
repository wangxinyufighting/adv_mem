from threading import Lock
from typing import Any

from attacker.models import AttackerRewardContext
from attacker.reward import AttackerReward
from training.reward_pool import map_rewards


_REWARD: AttackerReward | None = None
_REWARD_LOCK = Lock()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point."""
    global _REWARD
    if _REWARD is None:
        with _REWARD_LOCK:
            if _REWARD is None:
                _REWARD = AttackerReward.from_env()
    context = AttackerRewardContext.from_dict(extra_info)
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
