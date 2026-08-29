from typing import Any

from attacker.models import AttackerRewardContext
from attacker.reward import AttackerReward


_REWARD: AttackerReward | None = None


def _reward() -> AttackerReward:
    global _REWARD
    if _REWARD is None:
        _REWARD = AttackerReward.from_env()
    return _REWARD


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point."""
    context = AttackerRewardContext.from_dict(extra_info)
    return _reward().evaluate(solution_str, context)


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    group_ids: list[str],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Score rollout groups while sharing one Oracle call per prompt."""
    contexts = [AttackerRewardContext.from_dict(item) for item in extra_infos]
    return _reward().evaluate_batch(solution_strs, contexts, list(group_ids))
