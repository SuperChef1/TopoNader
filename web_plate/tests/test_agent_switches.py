import json
import tempfile
import unittest
from pathlib import Path

from topo_nader.agents import TopoNaderAgentOrchestrator
from topo_nader.architecture import TopoNaderArchitectureSpec, TopoNaderArchitectureValidator
from topo_nader.config import TopoNaderConfig, TopoNaderArchitectureConfig


class RoleClient:
    def __init__(self, invalid_first=False):
        self.messages = []
        self.invalid_first = invalid_first
        self.modifications = 0

    def complete(self, messages):
        self.messages.append(messages)
        system, user = messages[0]['content'], json.loads(messages[1]['content'])
        proposal = {'title': 'regularize', 'rationale': 'test', 'design_intent': 'regularize',
                    'changes': {'dropout': 0.2}}
        if 'You are TopoNaderReflector' in system:
            payload = {'diagnosis': 'overfit', 'evidence': [], 'risks': [],
                       'recommended_focus': 'regularize'}
        elif 'You are TopoNaderProposer' in system:
            payload = {'proposals': [proposal]}
        else:
            self.modifications += 1
            child = dict(user['parent'], dropout=0.2, metadata={'supporting_changes': {}})
            if self.invalid_first and self.modifications == 1:
                child['input_dim'] += 1
            payload = ({'proposal': proposal, 'architecture': child}
                       if 'No separate Proposer' in system else child)
        return {'content': json.dumps(payload), 'model': 'stub', 'prompt_tokens': 1,
                'completion_tokens': 2}


class AgentSwitchTests(unittest.TestCase):
    def test_each_role_combination_preserves_generation_and_removes_only_disabled_calls(self):
        for reflector in (True, False):
            for proposer in (True, False):
                with self.subTest(reflector=reflector, proposer=proposer), tempfile.TemporaryDirectory() as d:
                    client = RoleClient()
                    agents = TopoNaderAgentOrchestrator(client, Path(d),
                        reflector_enabled=reflector, proposer_enabled=proposer)
                    parent = TopoNaderArchitectureSpec(input_dim=24)
                    candidates, trace = agents.generate_candidates(parent,
                        {'pairwise_accuracy': .4}, [{'iteration': 1}], 1,
                        validator=TopoNaderArchitectureValidator(TopoNaderArchitectureConfig(),24).validate)
                    systems = [m[0]['content'] for m in client.messages]
                    self.assertEqual(sum('You are TopoNaderReflector' in s for s in systems),int(reflector))
                    self.assertEqual(sum('You are TopoNaderProposer' in s for s in systems),int(proposer))
                    self.assertEqual(sum('You are TopoNaderModifier' in s for s in systems),1)
                    self.assertEqual(len(candidates),1)
                    self.assertEqual(candidates[0].input_dim,24)
                    self.assertTrue(trace['candidate_attempts'][0]['change_scope']['valid'])
                    self.assertEqual(trace['agent_switches'],
                        {'reflector_enabled':reflector,'proposer_enabled':proposer})
                    if not reflector:
                        context = json.loads(client.messages[0][1]['content'])['diagnosis']
                        self.assertEqual(context['measured_metrics'],{'pairwise_accuracy':.4})
                        self.assertEqual(context['recent_experiences'],[{'iteration':1}])

    def test_direct_modifier_keeps_scope_validation_and_retries_invalid_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            client = RoleClient(invalid_first=True)
            agents = TopoNaderAgentOrchestrator(client, Path(d), proposer_enabled=False)
            candidates, trace = agents.generate_candidates(TopoNaderArchitectureSpec(),{},[],1,
                validator=TopoNaderArchitectureValidator(TopoNaderArchitectureConfig()).validate,
                retry_limit=2)
        self.assertEqual(len(candidates),1)
        self.assertFalse(trace['candidate_attempts'][0]['accepted'])
        self.assertIn('input_dim_must_be_preserved',trace['candidate_attempts'][0]['errors'])
        self.assertTrue(trace['candidate_attempts'][1]['accepted'])
        self.assertTrue(json.loads(client.messages[-1][1]['content'])['previous_attempt_feedback'])

    def test_direct_modifier_cannot_bypass_scope_targets(self):
        class WrongTargetClient(RoleClient):
            def complete(self, messages):
                result = super().complete(messages)
                payload = json.loads(result['content'])
                if 'architecture' in payload:
                    payload['architecture']['dropout'] = .3
                result['content'] = json.dumps(payload)
                return result
        with tempfile.TemporaryDirectory() as d:
            agents = TopoNaderAgentOrchestrator(WrongTargetClient(),Path(d),
                reflector_enabled=False,proposer_enabled=False)
            candidates, trace = agents.generate_candidates(TopoNaderArchitectureSpec(),{},[],1,retry_limit=1)
        self.assertEqual(candidates,[])
        self.assertIn('proposal_value_not_implemented: dropout',trace['candidate_attempts'][0]['errors'])

    def test_direct_modifier_rejects_duplicate_parent(self):
        parent = TopoNaderArchitectureSpec(dropout=.2)
        with tempfile.TemporaryDirectory() as d:
            agents = TopoNaderAgentOrchestrator(RoleClient(),Path(d),proposer_enabled=False)
            candidates, trace = agents.generate_candidates(parent,{},[],1,retry_limit=1)
        self.assertEqual(candidates,[])
        self.assertIn('duplicate_or_parent_architecture',trace['candidate_attempts'][0]['errors'])

    def test_switches_default_on_and_reject_non_boolean_values(self):
        config = TopoNaderConfig()
        self.assertTrue(config.agents.reflector_enabled)
        self.assertTrue(config.agents.proposer_enabled)
        for name in ('reflector_enabled','proposer_enabled'):
            config = TopoNaderConfig()
            setattr(config.agents,name,'false')
            with self.assertRaisesRegex(ValueError,'boolean'):
                config.validate()
