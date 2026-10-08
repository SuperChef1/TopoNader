from __future__ import annotations

import json

from ..architecture.spec import TopoNaderArchitectureSpec
from .base import TopoNaderBaseAgent


class TopoNaderModifier(TopoNaderBaseAgent):
    def modify(self, current: TopoNaderArchitectureSpec, proposal: dict, feedback: list[dict] | None = None):
        system = (
            "You are TopoNaderModifier. Convert one proposal into one complete architecture JSON. "
            "Allowed encoder_type: mlp,residual_mlp; predictor_type: mlp,transformer; "
            "activation: relu,gelu,leaky_relu,silu; normalization: layer_norm,batch_norm,none. "
            "Preserve input_dim=16. Return only the complete JSON object with fields name,input_dim,encoder_type,"
            "hidden_widths,latent_dim,activation,normalization,dropout,predictor_type,predictor_depth,"
            "predictor_width,predictor_activation,attention_heads,metadata."
        )
        payload, record = self.call(
            system,
            json.dumps(
                {
                    "parent": current.to_dict(),
                    "proposal": proposal,
                    "previous_attempt_feedback": feedback or [],
                    "instruction": "Correct every reported error and return a distinct valid child architecture.",
                },
                ensure_ascii=False,
            ),
        )
        payload.setdefault("metadata", {})
        payload["metadata"]["proposal"] = proposal
        return TopoNaderArchitectureSpec.from_dict(payload), record
