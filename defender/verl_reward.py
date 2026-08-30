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


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    group_ids: list[str],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """Score a batch once per unique response in each prompt group."""
    cache = {}
    results = []
    for response, context, group_id in zip(
        solution_strs, extra_infos, group_ids, strict=True
    ):
        key = (str(group_id), response)
        if key not in cache:
            cache[key] = compute_score("", response, "", context)
        results.append(cache[key])
    print(
        "Memory Builder Reward: "
        f"samples={len(results)} unique={len(cache)} "
        f"unavailable={sum(not item['reward_available'] for item in results)}",
        flush=True,
    )
    return results
