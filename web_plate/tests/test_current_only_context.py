import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from topo_nader.architecture import TopoNaderArchitectureSpec
from topo_nader.config import (TopoNaderConfig, TopoNaderAgentConfig, TopoNaderArchitectureConfig,
    TopoNaderTrainingConfig, TopoNaderMethodConfig, TopoNaderOptimizationConfig,
    TopoNaderCoverageConfig, TopoNaderUncertaintyConfig, load_config)
from topo_nader.logging.experience import TopoNaderExperienceStore
from topo_nader.pipeline import TopoNaderEngine


class TinyContextEngine(TopoNaderEngine):
    @property
    def default_architecture(self):
        return TopoNaderArchitectureSpec(hidden_widths=(16,),latent_dim=8,predictor_type='mlp',
            predictor_width=16,predictor_depth=1,attention_heads=1)


class ContextClient:
    def __init__(self):
        self.messages=[]
        self.modifier_calls=0

    def complete(self,messages):
        self.messages.append(messages)
        system=messages[0]['content'];user=json.loads(messages[1]['content'])
        if 'TopoNaderReflector' in system:
            raise AssertionError('NoReflector must not call Reflector')
        if 'TopoNaderProposer' in system:
            payload={'proposals':[{'title':'regularize','design_intent':'regularize',
                'rationale':'test','changes':{'dropout':.2}}]}
        else:
            self.modifier_calls+=1
            payload={**user['parent'],'dropout':.2,'metadata':{'supporting_changes':{}}}
            if self.modifier_calls==1:payload['input_dim']+=1
        return {'content':json.dumps(payload),'model':'stub','prompt_tokens':1,'completion_tokens':1}


class CurrentOnlyContextTests(unittest.TestCase):
    def config(self,mode):
        return TopoNaderConfig(
            agents=TopoNaderAgentConfig(reflector_enabled=False,proposer_enabled=True,history_context=mode),
            training=TopoNaderTrainingConfig(epochs=1,quick_epochs=1,patience=1,batch_size=8,device='cpu'),
            architecture=TopoNaderArchitectureConfig(candidate_count=1,fully_train_count=1,zero_cost_keep_count=1),
            method=TopoNaderMethodConfig(architecture_review_mode='every_round',uncertainty_sampling=False),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(particles=4,iterations=1,candidate_pool_size=8,
                fe_batch_size=2,exploitation_count=1,exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound=1.,upper_tolerance=0.)))

    def test_current_only_removes_experiences_and_parent_metadata_but_keeps_current_feedback_and_retries(self):
        for mode in ('recent','current_only'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);run=root/'run';client=ContextClient()
                engine=TinyContextEngine(self.config(mode),Path(__file__).parents[1],agent_client_factory=lambda _:client)
                engine.initialize_run(run)
                parent_path=run/'state/current_architecture.json'
                parent=TopoNaderArchitectureSpec.load(parent_path)
                replace(parent,metadata={'old_proposal':'PARENT_HISTORY_SENTINEL'}).save(parent_path)
                original_parent=parent_path.read_bytes()
                TopoNaderExperienceStore(run/engine.config.paths.experience_file).append(
                    {'iteration':0,'reflection':'EXPERIENCE_HISTORY_SENTINEL'})
                (run/'state/last_metrics.json').write_text(json.dumps({'pairwise_accuracy':.44}),encoding='utf-8')
                dataset=root/'initial.mat'
                designs=engine.geometry.denormalize(np.random.default_rng(42).random((12,4,4)))
                savemat(dataset,{'inputs':designs,'outputs':np.linspace(1.,2.,12)})
                with patch.object(TopoNaderExperienceStore,'recent',autospec=True,
                    return_value=[{'iteration':0,'reflection':'EXPERIENCE_HISTORY_SENTINEL'}]) as history:
                    engine.prepare_iteration(run,dataset,1)
                self.assertEqual(history.call_count,0 if mode=='current_only' else 1)
                serialized=json.dumps(client.messages)
                for sentinel in ('PARENT_HISTORY_SENTINEL','EXPERIENCE_HISTORY_SENTINEL'):
                    self.assertEqual(sentinel in serialized,mode=='recent')
                proposer=json.loads(client.messages[0][1]['content'])
                diagnostic=proposer['diagnosis']
                self.assertEqual(diagnostic['measured_metrics']['previous_fe_feedback']['pairwise_accuracy'],.44)
                self.assertTrue(diagnostic['measured_metrics']['current_parent_training']['last_training_history'])
                self.assertEqual(bool(diagnostic['recent_experiences']),mode=='recent')
                modifier_requests=[json.loads(m[1]['content']) for m in client.messages
                    if 'TopoNaderModifier' in m[0]['content']]
                self.assertTrue(modifier_requests[1]['previous_attempt_feedback'])
                self.assertEqual(parent_path.read_bytes(),original_parent)
                trace=json.loads((run/'iteration_0001/audit/trace.json').read_text(encoding='utf-8'))
                if mode=='current_only':
                    self.assertEqual(trace['context_policy']['recent_experience_count'],0)
                    self.assertFalse(trace['context_policy']['parent_metadata_included'])

    def test_old_snapshot_keeps_recent_history_and_cannot_switch_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';cfg=self.config('recent')
            engine=TinyContextEngine(cfg,Path(__file__).parents[1]);engine.initialize_run(run)
            p=run/'config.snapshot.json';snapshot=json.loads(p.read_text(encoding='utf-8'))
            snapshot['agents'].pop('history_context');p.write_text(json.dumps(snapshot),encoding='utf-8')
            original=p.read_bytes();engine.initialize_run(run)
            changed=replace(cfg,agents=replace(cfg.agents,history_context='current_only'))
            with self.assertRaisesRegex(ValueError,'different configuration'):
                TinyContextEngine(changed,Path(__file__).parents[1]).initialize_run(run)
            self.assertEqual(p.read_bytes(),original)

    def test_config_only_changes_name_and_history_context(self):
        root=Path(__file__).parents[1]
        base=load_config(root/'configs/Ablation-NoReflector-4bars-30coverage-seed42.json').to_dict()
        new=load_config(root/'configs/Ablation-NoReflector-CurrentOnly-4bars-30coverage-seed42.json').to_dict()
        new['experiment_name']=base['experiment_name'];new['agents']['history_context']='recent'
        self.assertEqual(base,new)

    def test_invalid_history_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'history_context'):
            self.config('unknown').validate()
