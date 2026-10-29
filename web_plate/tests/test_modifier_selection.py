import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import savemat

from topo_nader.agents.human_suggestions import LIBRARY_PATH, generate_human_proposals
from topo_nader.agents.orchestrator import TopoNaderAgentOrchestrator
from topo_nader.architecture import TopoNaderArchitectureSpec, TopoNaderArchitectureValidator
from topo_nader.config import (TopoNaderConfig, TopoNaderAgentConfig, TopoNaderArchitectureConfig,
    TopoNaderTrainingConfig, TopoNaderMethodConfig, TopoNaderOptimizationConfig,
    TopoNaderCoverageConfig, TopoNaderUncertaintyConfig)
from topo_nader.pipeline import TopoNaderEngine

DIGEST=hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest()


class SelectingClient:
    def __init__(self, focus='dropout_high', invalid=None, always_invalid=False, fail_first_rule=False):
        self.messages=[]
        self.focus=focus
        self.invalid=invalid
        self.always_invalid=always_invalid
        self.fail_first_rule=fail_first_rule
        self.select_calls=0
        self.implementation_calls=0

    def complete(self,messages):
        self.messages.append(messages)
        system=messages[0]['content'];user=json.loads(messages[1]['content'])
        if 'TopoNaderProposer' in system or 'No separate Proposer' in system:
            raise AssertionError('No Proposer or unconstrained joint planning call is allowed')
        if 'TopoNaderReflector' in system:
            payload={'diagnosis':'test','recommended_focus':self.focus,'evidence':[],'risks':[]}
        elif 'available_suggestions' in user:
            self.select_calls+=1
            ids=[p['human_suggestion_id'] for p in user['available_suggestions']]
            focus=user['diagnosis'].get('recommended_focus')
            if focus in ids:ids.remove(focus);ids.insert(0,focus)
            payload={'suggestion_ids':ids[:user['requested_count']],'rationale':'Use supplied diagnosis'}
            if self.invalid and (self.select_calls==1 or self.always_invalid):
                payload=self.invalid(payload)
        else:
            self.implementation_calls+=1
            payload={**user['parent'],**user['proposal']['changes'],'metadata':{'supporting_changes':{}}}
            if self.fail_first_rule and user['proposal']['human_suggestion_id']=='dropout_high':
                payload['normalization']='batch_norm'
                payload['metadata']['supporting_changes']={'normalization':'claimed implementation requirement'}
        return {'content':json.dumps(payload),'model':'stub','prompt_tokens':1,'completion_tokens':1}


class TinySelectionEngine(TopoNaderEngine):
    @property
    def default_architecture(self):
        return TopoNaderArchitectureSpec(hidden_widths=(16,),latent_dim=8,
            predictor_type='mlp',predictor_width=16,predictor_depth=1,attention_heads=1)


class ModifierSelectionTests(unittest.TestCase):
    def pool(self,parent=None):
        parent=parent or TopoNaderArchitectureSpec(hidden_widths=(16,),predictor_type='mlp',predictor_width=16)
        bounds=TopoNaderArchitectureConfig()
        proposals,policy=generate_human_proposals(parent,bounds,42,DIGEST,
            TopoNaderArchitectureValidator(bounds).validate)
        return parent,proposals,policy

    def test_diagnosis_reaches_modifier_and_changes_selected_library_id(self):
        parent,pool,policy=self.pool()
        for focus in ('dropout_high','encoder_gelu'):
            client=SelectingClient(focus)
            with tempfile.TemporaryDirectory() as d:
                agents=TopoNaderAgentOrchestrator(client,Path(d),proposer_enabled=False,
                    human_proposals=pool,human_policy=policy,modifier_selects_human=True)
                children,trace=agents.generate_candidates(parent,{'pairwise_accuracy':.4},[],1,retry_limit=1)
            self.assertEqual(len(children),1)
            selection=trace['human_suggestion_selection_attempts'][0]
            self.assertEqual(selection['selected_ids'],[focus])
            self.assertEqual(children[0].metadata['human_suggestion']['id'],focus)
            proposal=next(p for p in pool if p['human_suggestion_id']==focus)
            for key,value in proposal['changes'].items():self.assertEqual(children[0].to_dict()[key],value)
            self.assertEqual(trace['candidate_generation_mode'],'human_suggestions_modifier_select')
            self.assertEqual(client.select_calls,1)

    def test_invalid_selection_is_logged_and_retried_without_random_fallback(self):
        parent,pool,policy=self.pool()
        cases=[lambda p:{**p,'suggestion_ids':['invented','invented2']},
               lambda p:{**p,'suggestion_ids':[p['suggestion_ids'][0]]*2},
               lambda p:{**p,'suggestion_ids':p['suggestion_ids'][:1]},
               lambda p:{**p,'changes':{'dropout':.5}},
               lambda p:{**p,'rationale':''}]
        for invalid in cases:
            client=SelectingClient(invalid=invalid)
            with tempfile.TemporaryDirectory() as d:
                agents=TopoNaderAgentOrchestrator(client,Path(d),reflector_enabled=False,proposer_enabled=False,
                    human_proposals=pool,human_policy=policy,modifier_selects_human=True)
                children,trace=agents.generate_candidates(parent,{},[],2,retry_limit=2)
            self.assertEqual(len(children),2)
            self.assertFalse(trace['human_suggestion_selection_attempts'][0]['accepted'])
            self.assertTrue(trace['human_suggestion_selection_attempts'][1]['accepted'])
            requests=[json.loads(m[1]['content']) for m in client.messages if 'available_suggestions' in json.loads(m[1]['content'])]
            self.assertTrue(requests[1]['previous_selection_feedback'])

    def test_exhausted_selection_retries_keep_parent_instead_of_silently_sampling(self):
        parent,pool,policy=self.pool()
        client=SelectingClient(invalid=lambda p:{**p,'suggestion_ids':['missing']},always_invalid=True)
        with tempfile.TemporaryDirectory() as d:
            agents=TopoNaderAgentOrchestrator(client,Path(d),reflector_enabled=False,proposer_enabled=False,
                human_proposals=pool,human_policy=policy,modifier_selects_human=True)
            children,trace=agents.generate_candidates(parent,{},[],1,retry_limit=2)
        self.assertEqual(children,[])
        self.assertEqual(client.implementation_calls,0)
        self.assertEqual(len(trace['human_suggestion_selection_attempts']),2)

    def test_failed_suggestion_is_excluded_and_extra_changes_remain_forbidden(self):
        parent,pool,policy=self.pool()
        ids=['dropout_high','encoder_gelu'];by_id={p['human_suggestion_id']:p for p in pool}
        pool=[by_id[i] for i in ids]
        client=SelectingClient(fail_first_rule=True)
        with tempfile.TemporaryDirectory() as d:
            agents=TopoNaderAgentOrchestrator(client,Path(d),proposer_enabled=False,human_proposals=pool,
                human_policy=policy,modifier_selects_human=True)
            children,trace=agents.generate_candidates(parent,{},[],1,retry_limit=2)
        self.assertEqual(len(children),1)
        self.assertEqual(children[0].metadata['human_suggestion']['id'],'encoder_gelu')
        self.assertNotIn('dropout_high',trace['human_suggestion_selection_attempts'][1]['available_ids'])
        self.assertTrue(any('human_suggestion_extra_changes_forbidden' in e
            for e in trace['candidate_attempts'][0]['errors']))

    def test_no_reflector_variant_retains_raw_metrics_for_a_possible_matched_control(self):
        parent,pool,policy=self.pool();client=SelectingClient()
        with tempfile.TemporaryDirectory() as d:
            agents=TopoNaderAgentOrchestrator(client,Path(d),reflector_enabled=False,proposer_enabled=False,
                human_proposals=pool,human_policy=policy,modifier_selects_human=True)
            children,_=agents.generate_candidates(parent,{'measured':True},[],1,retry_limit=1)
        request=json.loads(client.messages[0][1]['content'])
        self.assertEqual(request['diagnosis']['measured_metrics'],{'measured':True})
        self.assertEqual(len(children),1)

    def test_cpu_pipeline_and_old_random_run_compatibility(self):
        client=SelectingClient();root=Path(__file__).parents[1]
        cfg=TopoNaderConfig(
            agents=TopoNaderAgentConfig(reflector_enabled=True,proposer_enabled=False,
                proposer_disabled_mode='human_suggestions_select',human_suggestions_sha256=DIGEST),
            training=TopoNaderTrainingConfig(epochs=1,quick_epochs=1,patience=1,batch_size=8,device='cpu'),
            architecture=TopoNaderArchitectureConfig(candidate_count=1,fully_train_count=1,zero_cost_keep_count=1),
            method=TopoNaderMethodConfig(architecture_review_mode='every_round',uncertainty_sampling=False),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(particles=4,iterations=1,candidate_pool_size=8,
                fe_batch_size=2,exploitation_count=1,exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound=1.,upper_tolerance=0.)))
        cfg.validate()
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);run=work/'new'
            engine=TinySelectionEngine(cfg,root,agent_client_factory=lambda _:client)
            data=work/'initial.mat'
            designs=engine.geometry.denormalize(np.random.default_rng(42).random((12,4,4)))
            savemat(data,{'inputs':designs,'outputs':np.linspace(1.,2.,12)})
            engine.prepare_iteration(run,data,1)
            trace=json.loads((run/'iteration_0001/audit/trace.json').read_text(encoding='utf-8'))
            self.assertEqual(trace['candidate_generation_mode'],'human_suggestions_modifier_select')
            self.assertEqual(trace['accepted_candidate_count'],1)
            result=work/'fe.mat';savemat(result,{'outputs':np.array([2.,3.])})
            engine.apply_feedback(run,data,1,result,work/'next.mat')
            feedback=json.loads((run/'iteration_0001/feedback.json').read_text(encoding='utf-8'))
            self.assertIsNotNone(feedback['post_reflection'])
            oldcfg=replace(cfg,agents=replace(cfg.agents,proposer_disabled_mode='human_suggestions'))
            oldrun=work/'old';oldengine=TinySelectionEngine(oldcfg,root)
            oldengine.initialize_run(oldrun)
            path=oldrun/'config.snapshot.json';original=path.read_bytes()
            oldengine.initialize_run(oldrun)
            with self.assertRaisesRegex(ValueError,'different configuration'):
                engine.initialize_run(oldrun)
            self.assertEqual(path.read_bytes(),original)
