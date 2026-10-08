from __future__ import annotations

import json

from .base import TopoNaderBaseAgent


class TopoNaderProposer(TopoNaderBaseAgent):
    def propose(self, current_architecture: dict, diagnosis: dict, count: int):
        system = (
            "You are TopoNaderProposer. Propose bounded architecture changes for a 16-input scalar regression surrogate "
            "used to rank topology candidates. Do not modify PSO, finite-element settings, data splits, losses, or training budget. "
            "Return JSON only: {\"proposals\":[{\"title\":str,\"rationale\":str,\"changes\":object}]}."
        )
        user = json.dumps(
            {"current_architecture": current_architecture, "diagnosis": diagnosis, "requested_count": int(count)},
            ensure_ascii=False,
        )
        return self.call(system, user)

