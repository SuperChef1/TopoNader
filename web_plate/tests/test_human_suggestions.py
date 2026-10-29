import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from topo_nader.agents.human_suggestions import LIBRARY_PATH, generate_human_proposals, load_human_library
from topo_nader.agents.orchestrator import TopoNaderAgentOrchestrator
from topo_nader.architecture import TopoNaderArchitectureSpec, TopoNaderArchitectureValidator
from topo_nader.config import (TopoNaderConfig, TopoNaderAgentConfig, TopoNaderArchitectureConfig,
    TopoNaderTrainingConfig, TopoNaderMethodConfig, TopoNaderOptimizationConfig,
    TopoNaderCoverageConfig, TopoNaderUncertaintyConfig)
from topo_nader.pipeline import TopoNaderEngine

DIGEST = hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest()


class HumanClient:
    def __init__(self, extra=False):
        self.messages = []
        self.extra = extra

    def complete(self, messages):
        self.messages.append(messages)
        system = messages[0]['content']
        if 'TopoNaderReflector' in system:
            payload = {'diagnosis':'test','evidence':[],'risks':[],'recommended_focus':'implement'}
        else:
            if 'No separate Proposer' in system or 'TopoNaderProposer' in system:
                raise AssertionError('Human policy must never ask an LLM to plan a proposal')
            user = json.loads(messages[1]['content'])
            payload = {**user['parent'], **user['proposal']['changes'], 'metadata':{'supporting_changes':{}}}
            if self.extra:
                payload['normalization'] = 'batch_norm'
                payload['metadata']['supporting_changes'] = {'normalization':'claimed compatibility'}
        return {'content':json.dumps(payload),'model':'stub','prompt_tokens':1,'completion_tokens':1}


class TinyHumanEngine(TopoNaderEngine):
    @property
    def default_architecture(self):
        return TopoNaderArchitectureSpec(hidden_widths=(16,),latent_dim=8,
            predictor_type='mlp',predictor_width=16,predictor_depth=1,attention_heads=1)


class HumanSuggestionTests(unittest.TestCase):
    def test_inactive_mlp_attention_fields_are_resolved_by_program_not_llm(self):
        parent=TopoNaderArchitectureSpec(hidden_widths=(16,),predictor_type='mlp',
            predictor_width=16,predictor_depth=0,attention_heads=0)
        bounds=TopoNaderArchitectureConfig()
        proposals,_=generate_human_proposals(parent,bounds,42,DIGEST,
            TopoNaderArchitectureValidator(bounds).validate)
        p=next(p for p in proposals if p['human_suggestion_id']=='predictor_transformer')
        self.assertGreater(p['changes']['attention_heads'],0)
        self.assertEqual(p['changes']['predictor_depth'],1)

    def test_library_is_seeded_scoped_and_dimension_preserving_without_changing_rng(self):
        for dim in (16,24):
            parent = TopoNaderArchitectureSpec(input_dim=dim,hidden_widths=(32,16),
                predictor_type='mlp',predictor_width=16,predictor_depth=1,attention_heads=1)
            bounds = TopoNaderArchitectureConfig()
            validator = TopoNaderArchitectureValidator(bounds,dim).validate
            rng = torch.get_rng_state().clone()
            a,t = generate_human_proposals(parent,bounds,42,DIGEST,validator)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            b,u = generate_human_proposals(parent,bounds,42,DIGEST,validator)
            c,_ = generate_human_proposals(parent,bounds,123,DIGEST,validator)
            self.assertEqual(a,b)
            self.assertNotEqual([p['title'] for p in a],[p['title'] for p in c])
            self.assertGreaterEqual(len(a),5)
            self.assertEqual(len({p['human_suggestion_id'] for p in a}),len(a))
            self.assertEqual(t['library_sha256'],DIGEST)
            for p in a:
                self.assertNotIn('input_dim',p['changes'])
                self.assertTrue(p['source_urls'])

    def test_no_legal_rules_produces_documented_shortfall(self):
        bounds = TopoNaderArchitectureConfig(max_parameters=1)
        proposals,trace = generate_human_proposals(TopoNaderArchitectureSpec(hidden_widths=(16,)),
            bounds,42,DIGEST,TopoNaderArchitectureValidator(bounds).validate)
        self.assertEqual(proposals,[])
        self.assertEqual(trace['applicable_count'],0)
        self.assertTrue(trace['skipped'])

    def test_hash_and_mode_configuration_cannot_silently_change(self):
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
            load_human_library('0'*64)
        cfg = TopoNaderConfig()
        cfg.agents.proposer_disabled_mode='human_suggestions'
        with self.assertRaisesRegex(ValueError,'proposer_enabled=false'):
            cfg.validate()
        cfg.agents.proposer_enabled=False
        with self.assertRaisesRegex(ValueError,'pinned library'):
            cfg.validate()
        cfg.agents.human_suggestions_sha256=DIGEST
        cfg.validate()

    def test_modifier_implements_external_targets_with_and_without_reflector(self):
        parent = TopoNaderArchitectureSpec()
        bounds = TopoNaderArchitectureConfig()
        proposals,policy=generate_human_proposals(parent,bounds,42,DIGEST,
            TopoNaderArchitectureValidator(bounds).validate)
        for reflector in (True,False):
            client=HumanClient()
            with tempfile.TemporaryDirectory() as d:
                agent=TopoNaderAgentOrchestrator(client,Path(d),reflector_enabled=reflector,
                    proposer_enabled=False,human_proposals=proposals,human_policy=policy)
                children,trace=agent.generate_candidates(parent,{'measured':True},[],2,retry_limit=1)
            self.assertEqual(len(children),2)
            self.assertEqual(trace['candidate_generation_mode'],'human_suggestions_modifier')
            for child,proposal in zip(children,proposals):
                for key,value in proposal['changes'].items():
                    self.assertEqual(child.to_dict()[key],value)
                self.assertEqual(child.metadata['human_suggestion']['id'],proposal['human_suggestion_id'])
            systems=[m[0]['content'] for m in client.messages]
            self.assertEqual(sum('TopoNaderReflector' in x for x in systems),int(reflector))

    def test_even_explained_extra_changes_are_rejected(self):
        p={'human_suggestion_id':'fixed','library_sha256':DIGEST,'title':'fixed',
            'design_intent':'regularize','changes':{'dropout':.2}}
        with tempfile.TemporaryDirectory() as d:
            agent=TopoNaderAgentOrchestrator(HumanClient(extra=True),Path(d),reflector_enabled=False,
                proposer_enabled=False,human_proposals=[p])
            children,trace=agent.generate_candidates(TopoNaderArchitectureSpec(),{},[],1,retry_limit=1)
        self.assertEqual(children,[])
        attempt=trace['candidate_attempts'][0]
        self.assertTrue(any('human_suggestion_extra_changes_forbidden' in x for x in attempt['errors']))
        self.assertFalse(attempt['change_scope']['valid'])

    def test_cpu_pipeline_human_policy_and_legacy_direct_run_are_separate(self):
        client=HumanClient()
        root=Path(__file__).parents[1]
        cfg=TopoNaderConfig(
            agents=TopoNaderAgentConfig(reflector_enabled=False,proposer_enabled=False,
                proposer_disabled_mode='human_suggestions',human_suggestions_sha256=DIGEST),
            training=TopoNaderTrainingConfig(epochs=1,quick_epochs=1,patience=1,batch_size=8,device='cpu'),
            architecture=TopoNaderArchitectureConfig(candidate_count=1,fully_train_count=1,zero_cost_keep_count=1),
            method=TopoNaderMethodConfig(architecture_review_mode='every_round',uncertainty_sampling=False),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(particles=4,iterations=1,candidate_pool_size=8,
                fe_batch_size=2,exploitation_count=1,exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound=1.,upper_tolerance=0.)))
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);run=work/'human'
            engine=TinyHumanEngine(cfg,root,agent_client_factory=lambda _:client)
            data=work/'data.mat'
            designs=engine.geometry.denormalize(np.random.default_rng(42).random((12,4,4)))
            savemat(data,{'inputs':designs,'outputs':np.linspace(1.,2.,12)})
            engine.prepare_iteration(run,data,1)
            trace=json.loads((run/'iteration_0001/audit/trace.json').read_text(encoding='utf-8'))
            self.assertEqual(trace['accepted_candidate_count'],1)
            self.assertEqual(trace['human_suggestion_policy']['library_sha256'],DIGEST)
            result=work/'fe.mat';savemat(result,{'outputs':np.array([2.,3.])})
            calls=len(client.messages)
            engine.apply_feedback(run,data,1,result,work/'next.mat')
            self.assertEqual(len(client.messages),calls)
            self.assertEqual(json.loads((run/'state.json').read_text())['next_iteration'],2)
            legacy=replace(cfg,agents=replace(cfg.agents,proposer_disabled_mode='modifier_direct',
                human_suggestions_sha256=None))
            oldrun=work/'old'
            oldengine=TinyHumanEngine(legacy,root)
            oldengine.initialize_run(oldrun)
            path=oldrun/'config.snapshot.json'
            payload=json.loads(path.read_text(encoding='utf-8'))
            payload['agents'].pop('proposer_disabled_mode');payload['agents'].pop('human_suggestions_sha256')
            path.write_text(json.dumps(payload),encoding='utf-8');original=path.read_bytes()
            oldengine.initialize_run(oldrun)
            with self.assertRaisesRegex(ValueError,'different configuration'):
                engine.initialize_run(oldrun)
            self.assertEqual(path.read_bytes(),original)
