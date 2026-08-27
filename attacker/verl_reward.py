from typing import Any

from attacker.models import AttackerRewardContext
from attacker.reward import AttackerReward


_REWARD: AttackerReward | None = None


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
        _REWARD = AttackerReward.from_env()
    context = AttackerRewardContext.from_dict(extra_info)
    return _REWARD.evaluate(solution_str, context)
