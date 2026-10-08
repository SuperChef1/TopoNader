from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..architecture.spec import TopoNaderArchitectureSpec
from .client import TopoNaderLLMClient
from .modifier import TopoNaderModifier
from .proposer import TopoNaderProposer
from .reflector import TopoNaderReflector


class TopoNaderAgentOrchestrator:
    """Reader-free TopoNader collaboration: Reflector -> Proposer -> Modifier."""

    def __init__(self, client: TopoNaderLLMClient, log_dir: str | Path):
        self.reflector = TopoNaderReflector("TopoNaderReflector", client, log_dir)
        self.proposer = TopoNaderProposer("TopoNaderProposer", client, log_dir)
        self.modifier = TopoNaderModifier("TopoNaderModifier", client, log_dir)

    def generate_candidates(
        self,
        current: TopoNaderArchitectureSpec,
        metrics: dict,
        experiences: list[dict],
        count: int,
        validator: Callable[[TopoNaderArchitectureSpec], object] | None = None,
        retry_limit: int = 3,
    ) -> tuple[list[TopoNaderArchitectureSpec], dict]:
        diagnosis, diagnosis_record = self.reflector.diagnose(current.to_dict(), metrics, experiences)
        candidates: list[TopoNaderArchitectureSpec] = []
        usage = [diagnosis_record]
        proposal_rounds = []
        attempts = []
        seen = {current.architecture_id}
        failure_context: list[dict] = []
        for proposal_round in range(max(1, retry_limit)):
            if len(candidates) >= count:
                break
            diagnosis_with_failures = dict(diagnosis)
            if failure_context:
                diagnosis_with_failures["previous_candidate_failures"] = failure_context[-10:]
                diagnosis_with_failures["avoid_architecture_ids"] = sorted(seen)
            proposal_payload, proposal_record = self.proposer.propose(
                current.to_dict(), diagnosis_with_failures, count - len(candidates)
            )
            usage.append(proposal_record)
            proposals = proposal_payload.get("proposals", [])
            proposal_rounds.append(proposal_payload)
            if not proposals:
                failure_context.append({"error": "proposer_returned_no_proposals", "round": proposal_round + 1})
                continue
            for proposal in proposals:
                if len(candidates) >= count:
                    break
                feedback: list[dict] = []
                accepted = False
                for attempt_index in range(max(1, retry_limit)):
                    attempt_record = {
                        "proposal": proposal,
                        "attempt": attempt_index + 1,
                    }
                    try:
                        candidate, record = self.modifier.modify(current, proposal, feedback)
                        usage.append(record)
                        attempt_record["architecture_id"] = candidate.architecture_id
                        errors: list[str] = []
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
                        feedback.append({"candidate": candidate.to_dict(), "errors": errors})
                        attempt_record.update({"accepted": False, "errors": errors})
                    except Exception as exc:
                        feedback.append({"errors": [f"generation_or_parse_error: {exc}"]})
                        attempt_record.update({"accepted": False, "errors": feedback[-1]["errors"]})
                    attempts.append(attempt_record)
                if not accepted:
                    failure_context.append({"proposal": proposal, "feedback": feedback})
        trace = {
            "diagnosis": diagnosis,
            "proposal_rounds": proposal_rounds,
            "candidate_attempts": attempts,
            "requested_candidate_count": count,
            "accepted_candidate_count": len(candidates),
            "failures": failure_context,
            "usage": usage,
        }
        return candidates, trace
