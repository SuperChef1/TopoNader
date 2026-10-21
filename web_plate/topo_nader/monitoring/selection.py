from __future__ import annotations

from collections.abc import Mapping

from ..training.evaluator import TopoNaderEvaluation


def select_provisional_architecture(results: Mapping[str, TopoNaderEvaluation]) -> str:
    """Rank first, regret second, prediction error third; never select on train R2."""
    if not results:
        raise ValueError("no architecture evaluations supplied")
    return max(
        results,
        key=lambda key: (
            results[key].pairwise_accuracy,
            results[key].kendall_tau,
            -results[key].topk_regret,
            -results[key].rmse,
        ),
    )

