import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from topo_nader.agents.random_proposals import generate_random_candidates
from topo_nader.architecture.spec import TopoNaderArchitectureSpec
from topo_nader.architecture.validator import TopoNaderArchitectureValidator
from topo_nader.config import (
    TopoNaderConfig, TopoNaderAgentConfig, TopoNaderArchitectureConfig,
    TopoNaderTrainingConfig, TopoNaderUncertaintyConfig, TopoNaderOptimizationConfig,
    TopoNaderCoverageConfig, TopoNaderMethodConfig, load_config,
)
from topo_nader.pipeline.engine import TopoNaderEngine


class RandomProposalTests(unittest.TestCase):
    def test_random_candidates_are_seeded_unique_bounded_and_scoped(self):
        for bars in [4, 6]:
            with self.subTest(bars=bars):
                parent = TopoNaderArchitectureSpec(input_dim=bars*4, hidden_widths=(32,16),
                    latent_dim=8, predictor_type='mlp', predictor_width=16)
                limits = TopoNaderArchitectureConfig(max_parameters=200000)
                validator = TopoNaderArchitectureValidator(limits, expected_input_dim=bars*4)
                a, trace = generate_random_candidates(parent, limits, 42, 5, validator.validate)
                b, _ = generate_random_candidates(parent, limits, 42, 5, validator.validate)
                c, _ = generate_random_candidates(parent, limits, 43, 5, validator.validate)
                self.assertEqual(len(a), 5)
                self.assertEqual([x.architecture_id for x in a], [x.architecture_id for x in b])
                self.assertNotEqual([x.architecture_id for x in a], [x.architecture_id for x in c])
                self.assertEqual(len({x.architecture_id for x in a}), 5)
                self.assertNotIn(parent.architecture_id, {x.architecture_id for x in a})
                self.assertTrue(all(x.input_dim == bars*4 and x.metadata['change_scope']['valid'] for x in a))
                self.assertEqual(trace['usage'], [])

    def test_random_audit_prepare_and_feedback_never_construct_llm_client(self):
        class TinyEngine(TopoNaderEngine):
            @property
            def default_architecture(self):
                return TopoNaderArchitectureSpec(hidden_widths=(16,), latent_dim=8,
                                                predictor_type='mlp', predictor_width=16)

        client_calls = []

        def forbidden_client(config):
            client_calls.append(config)
            raise AssertionError('Random ablation must not construct an LLM client')

        config = TopoNaderConfig(
            agents=TopoNaderAgentConfig(enabled=True, proposal_source='random'),
            method=TopoNaderMethodConfig(architecture_review_mode='every_round'),
            architecture=TopoNaderArchitectureConfig(candidate_count=1, fully_train_count=2,
                max_parameters=200000, zero_cost_enabled=False),
            training=TopoNaderTrainingConfig(epochs=1, quick_epochs=1, batch_size=8, patience=1, device='cpu'),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(particles=4, iterations=1,
                candidate_pool_size=8, fe_batch_size=2, exploitation_count=1, exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound_enabled=False)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = TinyEngine(config, Path(__file__).parents[1], forbidden_client)
            designs = engine.geometry.denormalize(np.random.default_rng(42).random((12,4,4)))
            initial = root/'initial.mat'
            savemat(initial, {'inputs': designs, 'outputs': np.linspace(1.,2.,12)})
            request = engine.prepare_iteration(root/'run', initial, 1)
            trace = json.loads((request.parent/'audit/trace.json').read_text())
            self.assertEqual(trace['proposal_source'], 'random')
            self.assertEqual(trace['accepted_candidate_count'], 1)
            self.assertIn('provisional_winner', trace)
            result = root/'result.mat'
            savemat(result, {'outputs': np.array([1.3,1.8])})
            engine.apply_feedback(root/'run', initial, 1, result, root/'next.mat')
            state = json.loads((root/'run/state.json').read_text())
            self.assertEqual(state['next_iteration'], 2)
            self.assertEqual(client_calls, [])
            self.assertFalse((request.parent/'agents').exists())
            self.assertEqual(loadmat(root/'next.mat')['outputsX'].size, 14)

    def test_ablation_configs_change_only_the_intended_settings(self):
        base = Path(__file__).parents[1]/'configs'
        b = load_config(base/'TopoNader-4bars-30coverage.json').to_dict()
        for group in ['C','D']:
            c = load_config(base/f'Ablation-{group}-4bars-30coverage-seed42.json').to_dict()
            c['experiment_name'] = b['experiment_name']
            if group == 'C':
                self.assertFalse(c['inheritance']['enabled'])
                c['inheritance']['enabled'] = b['inheritance']['enabled']
            else:
                self.assertEqual(c['agents']['proposal_source'], 'random')
                c['agents']['proposal_source'] = b['agents']['proposal_source']
            self.assertEqual(c,b)
            self.assertEqual(c['random_seed'],42)


if __name__ == '__main__':
    unittest.main()
