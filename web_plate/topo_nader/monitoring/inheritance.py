"""Conservative single-batch FE inheritance; training selection is unchanged."""
from dataclasses import asdict
import math

from ..config import TopoNaderInheritanceConfig
from ..training.evaluator import TopoNaderEvaluation
from .selection import select_provisional_architecture


def decide_inheritance(parent: TopoNaderEvaluation, child: TopoNaderEvaluation,
                       config: TopoNaderInheritanceConfig, distinct_child: bool = True) -> dict:
    reasons = []
    finite = all(math.isfinite(getattr(m, key)) for m in (parent, child)
                 for key in ('pairwise_accuracy', 'kendall_tau', 'topk_regret', 'rmse'))
    gain = child.pairwise_accuracy - parent.pairwise_accuracy if finite else None
    regret_increase = child.topk_regret - parent.topk_regret if finite else None
    if not distinct_child:
        reasons.append('no_distinct_child')
    elif not finite:
        reasons.append('nonfinite_metrics')
    else:
        winner = select_provisional_architecture({'parent': parent, 'child': child})
        if winner != 'child':
            reasons.append('parent_preferred_by_original_ranking')
        if config.enabled:
            if min(parent.sample_count, child.sample_count) < config.min_feedback_samples:
                reasons.append('insufficient_valid_feedback')
            if gain < config.min_pairwise_improvement - 1e-12:
                reasons.append('pairwise_gain_too_small')
            if regret_increase > config.max_topk_regret_increase + 1e-12:
                reasons.append('topk_regret_degraded')
            if config.min_pairwise_accuracy is not None and child.pairwise_accuracy < config.min_pairwise_accuracy:
                reasons.append('child_quality_below_minimum')
    return {'policy_version': 'single_batch_guard_v1', 'accepted': not reasons,
            'reasons': reasons or ['ranking_gain_and_topk_protection_passed' if config.enabled
                                   else 'original_ranking_passed'],
            'pairwise_improvement': gain, 'topk_regret_increase': regret_increase,
            'thresholds': asdict(config),
            'parent_metrics': parent.to_dict(), 'child_metrics': child.to_dict()}
