from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..architecture.spec import TopoNaderArchitectureSpec
from .client import TopoNaderLLMClient
from .modifier import TopoNaderModifier
from .proposer import TopoNaderProposer
from .reflector import TopoNaderReflector
from .change_scope import POLICY_VERSION, check_change_scope, proposal_errors


class TopoNaderAgentOrchestrator:
    """Reader-free TopoNader collaboration: Reflector -> Proposer -> Modifier."""

    def __init__(
        self,
        client: TopoNaderLLMClient,
        log_dir: str | Path,
        structured_output_retries: int = 2,
        reflector_enabled: bool = True,
        proposer_enabled: bool = True,
        human_proposals: list[dict] | None = None,
        human_policy: dict | None = None,
        modifier_selects_human: bool = False,
    ):
        self.reflector_enabled = reflector_enabled
        self.proposer_enabled = proposer_enabled
        if proposer_enabled and human_proposals is not None:
            raise ValueError('human proposals require the Proposer to be disabled')
        self.human_proposals = human_proposals
        self.human_policy = human_policy
        if modifier_selects_human and (proposer_enabled or human_proposals is None):
            raise ValueError('Modifier selection requires human proposals and a disabled Proposer')
        self.modifier_selects_human = modifier_selects_human
        agent_args = (client, log_dir, structured_output_retries)
        self.reflector = TopoNaderReflector("TopoNaderReflector", *agent_args)
        self.proposer = TopoNaderProposer("TopoNaderProposer", *agent_args)
        self.modifier = TopoNaderModifier("TopoNaderModifier", *agent_args)

    def generate_candidates(
        self,
        current: TopoNaderArchitectureSpec,
        metrics: dict,
        experiences: list[dict],
        count: int,
        validator: Callable[[TopoNaderArchitectureSpec], object] | None = None,
        retry_limit: int = 3,
    ) -> tuple[list[TopoNaderArchitectureSpec], dict]:
        if self.reflector_enabled:
            diagnosis, diagnosis_record = self.reflector.diagnose(current.to_dict(), metrics, experiences)
            usage = [diagnosis_record]
        else:
            # Keep measured information available, without an LLM interpretation.
            diagnosis = {"reflector_disabled": True, "measured_metrics": metrics,
                         "recent_experiences": experiences}
            usage = []
        candidates: list[TopoNaderArchitectureSpec] = []
        proposal_rounds = []
        attempts = []
        seen = {current.architecture_id}
        failure_context: list[dict] = []
        human_cursor = 0
        human_used_ids: set[str] = set()
        selection_attempts = []
        for proposal_round in range(max(1, retry_limit)):
            if len(candidates) >= count:
                break
            diagnosis_with_failures = dict(diagnosis)
            if failure_context:
                diagnosis_with_failures["previous_candidate_failures"] = failure_context[-10:]
                diagnosis_with_failures["avoid_architecture_ids"] = sorted(seen)
            if self.proposer_enabled:
                proposal_payload, proposal_record = self.proposer.propose(
                    current.to_dict(), diagnosis_with_failures, count - len(candidates)
                )
                usage.append(proposal_record)
                proposals = proposal_payload.get("proposals", [])
                proposal_rounds.append(proposal_payload)
            elif self.modifier_selects_human:
                available = [p for p in self.human_proposals if p['human_suggestion_id'] not in human_used_ids]
                requested = min(count - len(candidates), len(available))
                if not requested:
                    failure_context.append({'error': 'human_suggestion_pool_exhausted', 'round': proposal_round + 1})
                    break
                selection = {'round': proposal_round + 1,
                    'available_ids': [p['human_suggestion_id'] for p in available], 'requested_count': requested}
                try:
                    payload, record = self.modifier.select_human_suggestions(
                        current, diagnosis_with_failures, available, requested, failure_context[-10:])
                    usage.append(record)
                    selection['response'] = payload
                    if not isinstance(payload, dict) or set(payload) != {'suggestion_ids', 'rationale'}:
                        raise ValueError('selection_requires_only_suggestion_ids_and_rationale')
                    ids = payload['suggestion_ids']
                    by_id = {p['human_suggestion_id']: p for p in available}
                    if (not isinstance(ids, list) or len(ids) != requested
                            or not all(isinstance(x, str) for x in ids)):
                        raise ValueError('selection_requires_exact_count_of_string_ids')
                    if len(set(ids)) != len(ids):
                        raise ValueError('selection_contains_duplicate_ids')
                    if any(x not in by_id for x in ids):
                        raise ValueError('selection_contains_unknown_or_previously_used_ids')
                    if not isinstance(payload['rationale'], str) or not payload['rationale'].strip():
                        raise ValueError('selection_requires_rationale')
                    # Never accept LLM-authored changes: resolve authoritative targets by ID.
                    proposals = [by_id[x] for x in ids]
                    human_used_ids.update(ids)
                    selection.update(accepted=True, selected_ids=ids)
                    proposal_rounds.append({'source': 'modifier_selected_human_suggestions', 'proposals': proposals})
                except Exception as exc:
                    selection.update(accepted=False, error=str(exc))
                    selection_attempts.append(selection)
                    failure_context.append({'error': 'human_suggestion_selection_failed',
                                            'details': str(exc), 'round': proposal_round + 1})
                    continue
                selection_attempts.append(selection)
            elif self.human_proposals is not None:
                remaining = count - len(candidates)
                proposals = self.human_proposals[human_cursor:human_cursor + remaining]
                human_cursor += len(proposals)
                proposal_rounds.append({'source': 'human_suggestions', 'proposals': proposals})
            else:
                proposals = [None] * (count - len(candidates))
            if not proposals:
                failure_context.append({"error": ("human_suggestion_pool_exhausted"
                    if self.human_proposals is not None else "proposer_returned_no_proposals"),
                    "round": proposal_round + 1})
                continue
            for proposal in proposals:
                if len(candidates) >= count:
                    break
                scope_errors = proposal_errors(proposal) if self.proposer_enabled or self.human_proposals is not None else []
                if scope_errors:
                    failure_context.append({"proposal": proposal, "errors": scope_errors})
                    continue
                feedback: list[dict] = []
                accepted = False
                for attempt_index in range(max(1, retry_limit)):
                    attempt_record = {
                        "proposal": proposal,
                        "attempt": attempt_index + 1,
                    }
                    try:
                        if self.proposer_enabled:
                            candidate, record = self.modifier.modify(current, proposal, feedback)
                        elif self.human_proposals is not None:
                            candidate, record = self.modifier.modify(current, proposal, feedback,
                                implementation_context=diagnosis_with_failures, strict_targets=True)
                        else:
                            proposal, candidate, record = self.modifier.modify_direct(
                                current, diagnosis_with_failures, feedback
                            )
                            attempt_record["proposal"] = proposal
                        usage.append(record)
                        attempt_record["architecture_id"] = candidate.architecture_id
                        errors: list[str] = []
                        scope_report = check_change_scope(current, candidate, proposal)
                        attempt_record["change_scope"] = scope_report
                        errors.extend(scope_report["errors"])
                        if self.human_proposals is not None:
                            extras = set(scope_report['actual_changes']) - set(proposal['changes'])
                            scope_report['human_suggestion_exact_targets_only'] = not extras
                            if extras:
                                error = 'human_suggestion_extra_changes_forbidden: ' + ','.join(sorted(extras))
                                errors.append(error)
                                scope_report['errors'].append(error)
                                scope_report['valid'] = False
                            candidate.metadata['human_suggestion'] = {
                                'id': proposal['human_suggestion_id'], 'library_sha256': proposal['library_sha256']}
                        candidate.metadata["change_scope"] = scope_report
                        report_payload = None
                        if candidate.architecture_id in seen:
                            errors.append("duplicate_or_parent_architecture")
                        if validator is not None:
                            report = validator(candidate)
                            report_payload = report.to_dict() if hasattr(report, "to_dict") else report
                            if not bool(getattr(report, "valid", False)):
                                errors.extend(str(item) for item in getattr(report, "errors", ("validator_rejected",)))
                        attempt_record["validation"] = report_payload
                        if not errors:
                            candidates.append(candidate)
                            seen.add(candidate.architecture_id)
                            attempt_record["accepted"] = True
                            accepted = True
                            attempts.append(attempt_record)
                            break
                        feedback.append({"candidate": candidate.to_dict(), "errors": errors,
                                         "change_scope": scope_report})
                        attempt_record.update({"accepted": False, "errors": errors})
                    except Exception as exc:
                        feedback.append({"errors": [f"generation_or_parse_error: {exc}"]})
                        attempt_record.update({"accepted": False, "errors": feedback[-1]["errors"]})
                    attempts.append(attempt_record)
                if not accepted:
                    failure_context.append({"proposal": proposal, "feedback": feedback})
        trace = {
            "change_scope_policy": POLICY_VERSION,
            "agent_switches": {"reflector_enabled": self.reflector_enabled,
                               "proposer_enabled": self.proposer_enabled},
            "candidate_generation_mode": ("separate_proposer_modifier" if self.proposer_enabled
                else "human_suggestions_modifier_select" if self.modifier_selects_human
                else "human_suggestions_modifier" if self.human_proposals is not None else "modifier_joint_proposal"),
            "diagnosis": diagnosis,
            "proposal_rounds": proposal_rounds,
            "candidate_attempts": attempts,
            "requested_candidate_count": count,
            "accepted_candidate_count": len(candidates),
            "failures": failure_context,
            "usage": usage,
        }
        if self.human_proposals is not None:
            trace['human_suggestion_policy'] = self.human_policy
        if self.modifier_selects_human:
            trace['human_suggestion_policy'] = {
                **(self.human_policy or {}),
                'pool_ordering': (self.human_policy or {}).get('selection'),
                'selection': 'Modifier selects distinct applicable library IDs using diagnosis/context; no random fallback',
            }
            trace['human_suggestion_selection_attempts'] = selection_attempts
            trace['human_suggestion_selection_policy'] = 'modifier_library_id_selection_v1'
        return candidates, trace
