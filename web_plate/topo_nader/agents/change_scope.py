"""Audit declared proposal scope without pretending to verify scientific rationale."""
from __future__ import annotations

from ..architecture.spec import TopoNaderArchitectureSpec


POLICY_VERSION = "proposal_scope_v2"
COSMETIC_FIELDS = frozenset({"name", "metadata"})
EDITABLE_FIELDS = frozenset(
    TopoNaderArchitectureSpec().to_dict()
) - COSMETIC_FIELDS - {"input_dim"}


def proposal_errors(proposal: object) -> list[str]:
    if not isinstance(proposal, dict):
        return ["proposal_must_be_object"]
    errors = []
    if not isinstance(proposal.get("design_intent"), str) or not proposal["design_intent"].strip():
        errors.append("proposal_requires_one_explicit_design_intent")
    changes = proposal.get("changes")
    if not isinstance(changes, dict) or not changes:
        errors.append("proposal_changes_must_be_nonempty_field_value_object")
    else:
        for field in changes:
            if field not in EDITABLE_FIELDS:
                errors.append(f"proposal_field_not_editable: {field}")
    return errors


def check_change_scope(parent: TopoNaderArchitectureSpec,
                       child: TopoNaderArchitectureSpec, proposal: dict) -> dict:
    before, after = parent.to_dict(), child.to_dict()
    # Keep all raw architecture fields, even fields inactive for a particular predictor.
    # Cosmetic name/metadata changes do not count as architectural changes.
    actual = {key: {"before": before[key], "after": after[key]}
              for key in sorted(EDITABLE_FIELDS | {"input_dim"})
              if before[key] != after[key]}
    errors = proposal_errors(proposal)
    planned = proposal.get("changes", {})
    planned = planned if isinstance(planned, dict) else {}
    declarations = child.metadata.get("supporting_changes", {})
    if not isinstance(declarations, dict):
        errors.append("metadata.supporting_changes_must_be_field_reason_object")
        declarations = {}
    # These fields do not affect the architecture diff or ID. Keep their
    # declarations for audit, but do not treat them as structural scope errors.
    ignored_cosmetic = {key: value for key, value in declarations.items() if key in COSMETIC_FIELDS}
    declarations = {key: value for key, value in declarations.items() if key not in COSMETIC_FIELDS}
    if before["input_dim"] != after["input_dim"]:
        errors.append("input_dim_must_be_preserved")
    for key, value in planned.items():
        if key in EDITABLE_FIELDS and after[key] != value:
            errors.append(f"proposal_value_not_implemented: {key}")
    extras = sorted(set(actual) - set(planned) - {"input_dim"})
    for key in extras:
        reason = declarations.get(key)
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"undeclared_supporting_change: {key}; preserve parent value or explain necessity")
    for key, reason in declarations.items():
        if key not in EDITABLE_FIELDS:
            errors.append(f"supporting_field_not_editable: {key}")
        elif key not in extras:
            errors.append(f"supporting_declaration_not_extra_change: {key}")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"supporting_reason_required: {key}")
    return {
        "policy_version": POLICY_VERSION,
        "design_intent": proposal.get("design_intent"),
        "planned_changes": planned,
        "actual_changes": actual,
        "supporting_changes": declarations,
        "ignored_cosmetic_declarations": ignored_cosmetic,
        "supporting_rationale_semantically_verified": False,
        "valid": not errors,
        "errors": errors,
    }
