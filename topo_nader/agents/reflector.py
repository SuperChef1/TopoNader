from __future__ import annotations

import json

from .base import TopoNaderBaseAgent


class TopoNaderReflector(TopoNaderBaseAgent):
    def diagnose(self, architecture: dict, metrics: dict, recent_experiences: list[dict]):
        system = (
            f"You are TopoNaderReflector. Diagnose a {int(architecture.get('input_dim', 16))}-variable topology-optimization surrogate. "
            "Metrics are computed by deterministic code; do not invent measurements. Return JSON only with keys "
            "diagnosis, evidence, risks, recommended_focus. Ranking quality matters more than training R2."
        )
        user = json.dumps(
            {"current_architecture": architecture, "measured_metrics": metrics, "recent_experiences": recent_experiences},
            ensure_ascii=False,
        )
        return self.call(system, user)

    def reflect(self, audit_result: dict):
        system = (
            "You are TopoNaderReflector. Summarize a completed parent-child architecture audit. "
            "Use only supplied measurements. Return JSON only with keys outcome, effective_changes, failed_changes, lesson."
        )
        return self.call(system, json.dumps(audit_result, ensure_ascii=False))
