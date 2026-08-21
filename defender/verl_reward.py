from typing import Any

from defender.models import MemoryBuilderRewardContext
from defender.reward import MemoryBuilderReward


_REWARD: MemoryBuilderReward | None = None


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
        _REWARD = MemoryBuilderReward.from_env()
    context = MemoryBuilderRewardContext.from_dict(extra_info)
    return _REWARD.evaluate(solution_str, context)
