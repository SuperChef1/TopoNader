import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from topo_nader.agents.change_scope import check_change_scope, proposal_errors
from topo_nader.agents.orchestrator import TopoNaderAgentOrchestrator
from topo_nader.architecture.spec import TopoNaderArchitectureSpec


class ChangeScopeTests(unittest.TestCase):
    def setUp(self):
        self.parent = TopoNaderArchitectureSpec()
        self.proposal = {"title": "regularize", "design_intent": "reduce overfitting",
                         "rationale": "test", "changes": {"dropout": 0.2}}

    def test_planned_change_and_cosmetic_metadata_are_allowed(self):
        child = replace(self.parent, name="renamed", dropout=0.2, metadata={"note": "test"})
        report = check_change_scope(self.parent, child, self.proposal)
        self.assertTrue(report['valid'])
        self.assertEqual(set(report['actual_changes']), {'dropout'})

    def test_undeclared_extra_change_is_rejected(self):
        child = replace(self.parent, dropout=0.2, activation='silu')
        report = check_change_scope(self.parent, child, self.proposal)
        self.assertFalse(report['valid'])
        self.assertTrue(any('undeclared_supporting_change: activation' in x for x in report['errors']))

    def test_cosmetic_support_declarations_are_ignored_and_logged(self):
        for cosmetic in [{'name': 'Rename to reflect the proposal'},
                         {'metadata': ''}, {'name': None, 'metadata': {'note': 'test'}}]:
            with self.subTest(cosmetic=cosmetic):
                child = replace(self.parent, name='renamed', dropout=.2,
                                metadata={'supporting_changes': cosmetic})
                report = check_change_scope(self.parent, child, self.proposal)
                self.assertTrue(report['valid'])
                self.assertEqual(report['supporting_changes'], {})
                self.assertEqual(report['ignored_cosmetic_declarations'], cosmetic)
                self.assertEqual(set(report['actual_changes']), {'dropout'})
                self.assertEqual(report['policy_version'], 'proposal_scope_v2')

    def test_cosmetic_declaration_does_not_hide_unexplained_structural_change(self):
        child = replace(self.parent, dropout=.2, activation='silu',
                        metadata={'supporting_changes': {'name': 'Rename'}})
        report = check_change_scope(self.parent, child, self.proposal)
        self.assertFalse(report['valid'])
        self.assertTrue(any('undeclared_supporting_change: activation' in e for e in report['errors']))
        child = replace(child, metadata={'supporting_changes': {'name': 'Rename', 'learning_rate': 'test'}})
        self.assertIn('supporting_field_not_editable: learning_rate',
                      check_change_scope(self.parent, child, self.proposal)['errors'])

    def test_rename_does_not_create_a_distinct_architecture(self):
        child = replace(self.parent, name='renamed', metadata={'note': 'cosmetic'})
        self.assertEqual(child.architecture_id, self.parent.architecture_id)

    def test_declared_supporting_change_is_auditable_not_semantically_certified(self):
        proposal = {**self.proposal, 'changes': {'predictor_width': 130}}
        child = replace(self.parent, predictor_width=130, attention_heads=2,
                        metadata={'supporting_changes': {'attention_heads':
                                  'Use two heads so predictor_width=130 is divisible by heads.'}})
        report = check_change_scope(self.parent, child, proposal)
        self.assertTrue(report['valid'])
        self.assertFalse(report['supporting_rationale_semantically_verified'])

    def test_multiple_planned_fields_are_allowed(self):
        proposal = {**self.proposal, 'changes': {'hidden_widths': [128, 64], 'latent_dim': 16}}
        child = replace(self.parent, hidden_widths=(128, 64), latent_dim=16)
        self.assertTrue(check_change_scope(self.parent, child, proposal)['valid'])

    def test_declaring_reasons_does_not_override_exact_proposal_or_input_dimension(self):
        child = replace(self.parent, dropout=0.3, input_dim=24,
                        metadata={'supporting_changes': {'input_dim': 'claimed necessary'}})
        report = check_change_scope(self.parent, child, self.proposal)
        self.assertIn('proposal_value_not_implemented: dropout', report['errors'])
        self.assertIn('input_dim_must_be_preserved', report['errors'])

    def test_invalid_proposal_and_empty_or_stale_support_are_rejected(self):
        self.assertTrue(proposal_errors({'changes': {}}))
        self.assertTrue(proposal_errors({**self.proposal, 'changes': {'learning_rate': 0.1}}))
        for support in [{'activation': ' '}, {'normalization': 'not actually changed'}]:
            child = replace(self.parent, dropout=0.2, activation='silu',
                            metadata={'supporting_changes': support})
            self.assertFalse(check_change_scope(self.parent, child, self.proposal)['valid'])

    def test_orchestrator_returns_diff_to_modifier_and_accepts_corrected_attempt(self):
        test = self

        class Client:
            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    payload = {'diagnosis': 'test'}
                elif self.calls == 2:
                    payload = {'proposals': [test.proposal]}
                else:
                    if self.calls == 4:
                        feedback = json.loads(messages[-1]['content'])['previous_attempt_feedback']
                        test.assertIn('activation', feedback[0]['change_scope']['actual_changes'])
                    payload = replace(test.parent, dropout=0.2,
                                      activation='silu' if self.calls == 3 else 'relu').to_dict()
                return {'content': json.dumps(payload), 'model': 'stub'}

        with tempfile.TemporaryDirectory() as directory:
            candidates, trace = TopoNaderAgentOrchestrator(Client(), Path(directory)).generate_candidates(
                self.parent, {}, [], 1, retry_limit=2)
        self.assertEqual(len(candidates), 1)
        self.assertFalse(trace['candidate_attempts'][0]['accepted'])
        self.assertTrue(trace['candidate_attempts'][1]['accepted'])
        self.assertTrue(candidates[0].metadata['change_scope']['valid'])


if __name__ == '__main__':
    unittest.main()
