from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import TopoNaderTriggerConfig
from ..training.evaluator import TopoNaderEvaluation


@dataclass
class TopoNaderTriggerState:
    architecture_age: int = 0
    consecutive_degraded_rounds: int = 0
    last_audit_iteration: int | None = None


@dataclass
class TopoNaderTriggerDecision:
    audit_required: bool
    reasons: tuple[str, ...]
    state: TopoNaderTriggerState

    def to_dict(self) -> dict:
        return {"audit_required": self.audit_required, "reasons": list(self.reasons), "state": asdict(self.state)}


class TopoNaderTrigger:
    def __init__(self, config: TopoNaderTriggerConfig):
        self.config = config

    def evaluate(
        self,
        iteration: int,
        metrics: TopoNaderEvaluation,
        state: TopoNaderTriggerState,
    ) -> TopoNaderTriggerDecision:
        if not self.config.enabled:
            return TopoNaderTriggerDecision(False, ("trigger_disabled",), state)
        new_state = TopoNaderTriggerState(
            architecture_age=state.architecture_age + 1,
            consecutive_degraded_rounds=state.consecutive_degraded_rounds,
            last_audit_iteration=state.last_audit_iteration,
        )
        reasons: list[str] = []
        enough_feedback = metrics.sample_count >= self.config.min_feedback_samples
        degraded = enough_feedback and (
            metrics.pairwise_accuracy < self.config.min_pairwise_accuracy
            or metrics.kendall_tau < self.config.min_kendall_tau
            or metrics.topk_regret > self.config.max_topk_regret
        )
        new_state.consecutive_degraded_rounds = new_state.consecutive_degraded_rounds + 1 if degraded else 0
        if new_state.consecutive_degraded_rounds >= self.config.patience:
            reasons.append("persistent_ranking_degradation")
        if new_state.architecture_age >= self.config.max_architecture_age:
            reasons.append("maximum_architecture_age")
        audit_required = bool(reasons)
        if audit_required:
            new_state.last_audit_iteration = iteration
        return TopoNaderTriggerDecision(audit_required, tuple(reasons), new_state)

