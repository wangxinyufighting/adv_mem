from typing import Any

from attacker.models import RouteSelectorRewardContext
from attacker.reward import RouteSelectorReward


_REWARD: RouteSelectorReward | None = None


def _reward() -> RouteSelectorReward:
    global _REWARD
    if _REWARD is None:
        _REWARD = RouteSelectorReward.from_env()
    return _REWARD


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom reward entry point."""
    context = RouteSelectorRewardContext.from_dict(extra_info)
    return _reward().evaluate(solution_str, context)


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    group_ids: list[str],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Score route choices while caching each probe outcome per memory version."""
    contexts = [RouteSelectorRewardContext.from_dict(item) for item in extra_infos]
    return _reward().evaluate_batch(solution_strs, contexts, list(group_ids))
