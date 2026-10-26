from __future__ import annotations

import json

from .base import TopoNaderBaseAgent
from .change_scope import EDITABLE_FIELDS


class TopoNaderProposer(TopoNaderBaseAgent):
    def propose(self, current_architecture: dict, diagnosis: dict, count: int):
        input_dim = int(current_architecture.get("input_dim", 16))
        system = (
            f"You are TopoNaderProposer. Propose bounded architecture changes for a {input_dim}-input scalar regression surrogate "
            "Preserve input_dim exactly. "
            "used to rank topology candidates. Do not modify PSO, finite-element settings, data splits, losses, or training budget. "
            "Each proposal must have ONE explicit design intent. Declare the necessary architectural changes "
            "as a field-to-exact-target-value object; preserve unrelated structure. Multiple related fields "
            "may change together. Do not combine independent experiments in one proposal. "
            f"Editable fields: {','.join(sorted(EDITABLE_FIELDS))}. "
            "Return JSON only: {\"proposals\":[{\"title\":str,\"rationale\":str,"
            "\"design_intent\":str,\"changes\":object}]}."
        )
        user = json.dumps(
            {"current_architecture": current_architecture, "diagnosis": diagnosis, "requested_count": int(count)},
            ensure_ascii=False,
        )
        return self.call(system, user)
