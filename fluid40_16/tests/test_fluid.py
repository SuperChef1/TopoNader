import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from topo_nader.config import load_config
from topo_nader.data.dataset import TopoNaderDataset
from topo_nader.optimization.geometry import TopoNaderGeometry
from topo_nader.optimization.acquisition import finalize_candidate_pool
from topo_nader.optimization.pso import TopoNaderPSO
from topo_nader.optimization.fluid_search import search_batch

ROOT = Path(__file__).resolve().parents[1]


class FluidTests(unittest.TestCase):
    def test_legacy_code_and_decisions_unchanged(self):
        original = ROOT / 'reference/utils.py'
        copied = ROOT / 'topo_nader/optimization/legacy_connectivity.py'
        self.assertEqual(original.read_bytes(), copied.read_bytes())
        spec = importlib.util.spec_from_file_location('reference_utils', original)
        ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
        geometry = TopoNaderGeometry(load_config(ROOT/'configs/smoke.json').optimization)
        rng = np.random.default_rng(73)
        grids = [(rng.random((40,16)) < density).astype(float)
                 for density in [0,.05,.2,.4,.7,1] for _ in range(15)]
        # Intentionally preserve original false rejection of a row-5-only path.
        row5 = np.ones((40,16)); row5[:,4] = 0
        grids.append(row5)
        for x in grids:
            self.assertEqual(geometry.valid(x), ref.isvalid(x.T))
        self.assertFalse(geometry.valid(row5))

    def test_initial_data_and_matlab_cell_numbering(self):
        config = load_config(ROOT/'configs/smoke.json')
        data = TopoNaderDataset.from_mat(ROOT/'data/initial_641.mat', config.data)
        self.assertEqual(data.inputs.shape,(641,40,16))
        self.assertTrue(np.isfinite(data.outputs).all())
        # MATLAB design(:) must visit x fastest, starting at top left.
        for x,y,sq in [(0,0,1),(39,0,40),(0,1,41),(39,15,640),(7,5,208)]:
            a=np.zeros((40,16));a[x,y]=1
            self.assertEqual(np.flatnonzero(a.ravel(order='F')).tolist(),[sq-1])
        with self.assertRaises(ValueError):
            TopoNaderDataset(np.zeros((3,640)),np.ones(3),config.data)

    def test_pool_filters_before_prediction(self):
        config=load_config(ROOT/'configs/smoke.json');g=TopoNaderGeometry(config.optimization)
        a=np.zeros((40,16));b=a.copy();b[10,5]=1
        def predictor(x):
            self.assertTrue(np.isin(x,[0,1]).all())
            return np.ones(len(x)),np.zeros(len(x))
        raw=np.stack([a,b,b,np.ones_like(a)]).reshape(-1,640)
        pool=finalize_candidate_pool(raw,g,predictor,excluded_designs=a[None])
        self.assertEqual(pool.designs.shape,(1,40,16))
        self.assertEqual(pool.rejection_counts,{'connectivity':1,'historical_duplicate':1,'batch_duplicate':1})

    def test_archive_scores_stay_aligned_across_stall_reset(self):
        objective=lambda x:(x>=.5).sum(axis=1).astype(float)
        result=TopoNaderPSO(8,5,stall_limit=1,seed=3).optimize(objective,dimensions=640,archive_size=12)
        self.assertLessEqual(len(result.trajectory_positions),12)
        np.testing.assert_equal(result.trajectory_scores,objective(result.trajectory_positions))

    def test_configs_and_architecture_are_fluid_specific(self):
        from topo_nader.pipeline.engine import TopoNaderEngine
        for name in ['default','baseline_fixed','smoke','smoke_audit']:
            cfg=load_config(ROOT/f'configs/{name}.json')
            engine=TopoNaderEngine(cfg,ROOT)
            self.assertEqual(cfg.data.design_shape,(40,16))
            self.assertEqual(engine.default_architecture.predictor_type,'mlp')
            cfg.optimization.coverage.upper_bound_enabled=True
            with self.assertRaises(ValueError):cfg.validate()
        cfg=load_config(ROOT/'configs/smoke.json')
        cfg.initial_architecture['predictor_type']='transformer'
        with self.assertRaises(ValueError):TopoNaderEngine(cfg,ROOT)

    def test_infeasible_pool_exhausts_bounded_refills(self):
        cfg=load_config(ROOT/'configs/smoke.json')
        cfg.optimization.refill_attempts=1
        cfg.optimization.refill_pool_size=4
        g=TopoNaderGeometry(cfg.optimization)
        g.valid=lambda x:False
        data=TopoNaderDataset(np.zeros((3,40,16)),np.arange(1.,4.),cfg.data)
        def predictor(x):return np.ones(len(x)),np.zeros(len(x))
        with self.assertRaisesRegex(RuntimeError,'bounded refills'):
            search_batch(cfg,g,data,1,predictor,lambda x:np.ones(len(x)),lambda x:np.ones(len(x)))

    def run_cycle(self, config_name):
        import torch
        from topo_nader.pipeline.engine import TopoNaderEngine
        torch.set_num_threads(1)
        config=load_config(ROOT/f'configs/{config_name}.json')
        engine=TopoNaderEngine(config,ROOT)
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';initial=ROOT/'data/initial_641.mat'
            request=engine.prepare_iteration(run,initial,1)
            payload=loadmat(request);x=payload['inputs']
            self.assertEqual(x.shape,(4,40,16))
            self.assertTrue(all(engine.geometry.valid(a) for a in x))
            self.assertEqual(len({a.tobytes() for a in x}),4)
            historical={a.tobytes() for a in loadmat(initial)['inputs'].astype(float)}
            self.assertTrue(all(a.tobytes() not in historical for a in x))
            self.assertEqual(payload['source'].shape,(4,1))
            # Synthetic oracle exercises plumbing only, never represents COMSOL truth.
            outputs=(1.5+.1*x.mean(axis=(1,2))).reshape(-1,1)
            result=run/'iteration_0001/fe_result.mat'
            savemat(result,{'inputs':x,'outputs':outputs,'status':np.array(['synthetic']*4,dtype=object)})
            next_data=run/'datasets/dataset_0002.mat'
            wrong=run/'wrong.mat'
            savemat(wrong,{'inputs':x,'outputs':outputs})
            with self.assertRaisesRegex(ValueError,'feedback dataset differs'):
                engine.apply_feedback(run,wrong,1,result,next_data)
            savemat(result,{'inputs':1-x,'outputs':outputs})
            with self.assertRaisesRegex(ValueError,'do not match'):
                engine.apply_feedback(run,initial,1,result,next_data)
            savemat(result,{'inputs':x,'outputs':outputs,'status':np.array(['synthetic']*4,dtype=object)})
            engine.apply_feedback(run,initial,1,result,next_data)
            self.assertEqual(loadmat(next_data)['inputsX'].shape,(645,40,16))
            self.assertEqual(json.loads((run/'state.json').read_text())['next_iteration'],2)
            self.assertTrue((run/'iteration_0001/fluid_diagnostics.json').is_file())
            # Resume on appended data, not sorted legacy fields.
            engine.prepare_iteration(run,next_data,2)
            with self.assertRaises(FileExistsError):engine.prepare_iteration(run,next_data,2)
            state=json.loads((run/'state.json').read_text())
            self.assertEqual(state['next_iteration'],2)
            self.assertTrue((run/'runtime/fluid.mph').is_file())
            if config_name=='smoke_audit':
                trace=json.loads((run/'iteration_0001/audit/trace.json').read_text())
                self.assertGreater(len(trace['candidate_attempts']),0)

    def test_mean_only_exact_batch(self):
        from topo_nader.pipeline.engine import TopoNaderEngine
        import torch
        torch.set_num_threads(1)
        cfg=load_config(ROOT/'configs/smoke.json')
        cfg.method.uncertainty_sampling=False
        cfg.optimization.elite_local_search.enabled=False
        engine=TopoNaderEngine(cfg,ROOT)
        with tempfile.TemporaryDirectory() as folder:
            request=engine.prepare_iteration(Path(folder),ROOT/'data/initial_641.mat',1)
            payload=loadmat(request)
            self.assertEqual(payload['inputs'].shape,(4,40,16))
            self.assertTrue(np.all(payload['predicted_uncertainty']==0))

    def test_fixed_prepare_feedback_resume(self):self.run_cycle('smoke')

    def test_architecture_audit_and_feedback(self):self.run_cycle('smoke_audit')


if __name__=='__main__':unittest.main()
