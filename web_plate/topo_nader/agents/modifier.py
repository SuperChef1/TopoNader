from __future__ import annotations

import json

from ..architecture.spec import TopoNaderArchitectureSpec
from .base import TopoNaderBaseAgent
from .change_scope import EDITABLE_FIELDS


class TopoNaderModifier(TopoNaderBaseAgent):
    def select_human_suggestions(self, current: TopoNaderArchitectureSpec, diagnosis: dict,
                                 available: list[dict], count: int, feedback: list[dict]):
        """Select only frozen library IDs; implementation is a separate checked call."""
        system = (
            "You are TopoNaderModifier. The separate Proposer is disabled. "
            "Use the supplied diagnosis/context and measured evidence to select the most appropriate "
            "human architectural suggestions for the current parent. Every suggestion is an alternative "
            "child of the same parent, not a sequential combined edit. "
            "Choose exactly requested_count distinct IDs from available_suggestions, in priority order. "
            "Do not invent suggestions, change target values, combine rules, or return an architecture. "
            "Each selected suggestion will later be resolved by program code and implemented separately. "
            "Return JSON only with keys suggestion_ids (list of strings) and rationale (nonempty string). "
            "Do not claim unmeasured performance improvements."
        )
        payload, record = self.call(system, json.dumps({
            'parent': current.to_dict(), 'diagnosis': diagnosis,
            'available_suggestions': available, 'requested_count': int(count),
            'previous_selection_feedback': feedback,
        }, ensure_ascii=False))
        return payload, record

    def modify_direct(self, current: TopoNaderArchitectureSpec, diagnosis: dict,
                      feedback: list[dict] | None = None):
        """Ablation: merge proposal planning and implementation in one LLM call."""
        system = (
            "You are TopoNaderModifier. No separate Proposer is available. "
            "Using the supplied diagnosis/context, propose ONE bounded architectural design intent "
            "and implement it in the same response. Preserve all unrelated parent fields. "
            "Do not modify PSO, FE settings, data splits, losses or training budget. "
            f"Preserve input_dim={current.input_dim} exactly. "
            f"Editable proposal fields: {','.join(sorted(EDITABLE_FIELDS))}. "
            "Allowed encoder_type: mlp,residual_mlp; predictor_type: mlp,transformer; "
            "activation: relu,gelu,leaky_relu,silu; normalization: layer_norm,batch_norm,none. "
            "Return JSON only with two keys: proposal and architecture. "
            "proposal must contain title, rationale, design_intent, and changes (a nonempty "
            "field-to-exact-target-value object). architecture must be the complete JSON with fields "
            "name,input_dim,encoder_type,hidden_widths,latent_dim,activation,normalization,dropout,"
            "predictor_type,predictor_depth,predictor_width,predictor_activation,attention_heads,metadata. "
            "Implement every declared target exactly. Necessary supporting changes must each be "
            "explained in architecture.metadata.supporting_changes as {field_name: nonempty reason}; "
            "otherwise use {}. Do not declare cosmetic name/metadata edits or copy old explanations. "
            "Latent dimension changes do not themselves require predictor_width changes. "
            "Do not combine independent experiments. Return a distinct valid child architecture."
        )
        payload, record = self.call(system, json.dumps({
            "parent": current.to_dict(), "diagnosis": diagnosis,
            "previous_attempt_feedback": feedback or [],
        }, ensure_ascii=False))
        if not isinstance(payload, dict) or not isinstance(payload.get("proposal"), dict) \
                or not isinstance(payload.get("architecture"), dict):
            raise ValueError("direct Modifier requires proposal and architecture objects")
        proposal, architecture = payload["proposal"], payload["architecture"]
        architecture.setdefault("metadata", {})
        architecture["metadata"]["proposal"] = proposal
        return proposal, TopoNaderArchitectureSpec.from_dict(architecture), record

    def modify(self, current: TopoNaderArchitectureSpec, proposal: dict, feedback: list[dict] | None = None,
               implementation_context: dict | None = None, strict_targets: bool = False):
        system = (
            "You are TopoNaderModifier. Convert one proposal into one complete architecture JSON. "
            "Allowed encoder_type: mlp,residual_mlp; predictor_type: mlp,transformer; "
            "activation: relu,gelu,leaky_relu,silu; normalization: layer_norm,batch_norm,none. "
            f"Preserve the parent's input_dim={current.input_dim} exactly. Return only the complete JSON object with fields name,input_dim,encoder_type,"
            "hidden_widths,latent_dim,activation,normalization,dropout,predictor_type,predictor_depth,"
            "predictor_width,predictor_activation,attention_heads,metadata."
            " Implement the exact target values in proposal.changes. Preserve every unrelated parent field. "
            "Necessary supporting changes are allowed, but declare EACH extra changed field in "
            "metadata.supporting_changes as {field_name: nonempty explanation of why this is necessary "
            "for the proposal or compatibility}. Use {} when no supporting changes are needed. "
            "Cosmetic name and metadata edits do not count as architecture changes; do not list them "
            "as supporting changes. They are ignored by structural scope checks, not by duplicate detection. "
            "Do not copy old supporting_changes declarations from parent metadata. "
            "The factory automatically rebuilds layer dimensions for latent_dim changes; changing latent_dim "
            "does not by itself require changing predictor_width. Do not add independent improvements."
        )
        user_payload = {
            "parent": current.to_dict(), "proposal": proposal,
            "previous_attempt_feedback": feedback or [],
            "instruction": "Correct every reported error and return a distinct valid child architecture.",
        }
        if strict_targets:
            system += (
                " This is a frozen human-suggestion experiment. Do not plan, select, replace or reinterpret "
                "the suggestion. Implement only proposal.changes. No extra structural changes are allowed, "
                "even with supporting explanations. Context may explain implementation constraints only."
            )
            user_payload['implementation_context'] = implementation_context or {}
        payload, record = self.call(
            system,
            json.dumps(user_payload, ensure_ascii=False),
        )
        payload.setdefault("metadata", {})
        payload["metadata"]["proposal"] = proposal
        return TopoNaderArchitectureSpec.from_dict(payload), record
