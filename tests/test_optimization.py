import unittest

import numpy as np

from topo_nader.config import TopoNaderOptimizationConfig
from topo_nader.optimization import (
    TopoNaderGeometry,
    TopoNaderPSO,
    refresh_batch_predictions,
    select_active_batch,
)


class TopoNaderOptimizationTests(unittest.TestCase):
    def test_pso_maximizes_objective(self):
        target = np.full(16, 0.75)
        pso = TopoNaderPSO(particles=20, iterations=20, seed=7)
        result = pso.optimize(lambda x: -np.sum((x - target) ** 2, axis=1))
        self.assertGreater(float(np.max(result.scores)), -1.0)

    def test_active_batch_contains_exploitation_and_exploration(self):
        config = TopoNaderOptimizationConfig(
            fe_batch_size=4,
            exploitation_count=2,
            exploration_count=2,
            ratio_target=1.0,
        )
        geometry = TopoNaderGeometry(config)
        rng = np.random.default_rng(4)
        pool = rng.random((30, 16))
        mean = np.linspace(0.0, 1.0, 30)
        uncertainty = np.linspace(1.0, 0.0, 30)
        batch = select_active_batch(pool, mean, uncertainty, geometry, 2, 2, 2.0)
        self.assertEqual(batch.designs.shape, (4, 4, 4))
        self.assertEqual(batch.sources.count("exploitation"), 2)
        self.assertEqual(batch.sources.count("exploration"), 2)

    def test_finalized_designs_are_repredicted(self):
        config = TopoNaderOptimizationConfig(
            fe_batch_size=2,
            exploitation_count=1,
            exploration_count=1,
            ratio_target=1.0,
        )
        geometry = TopoNaderGeometry(config)
        pool = np.full((4, 16), 0.5137)
        pool[1:] += np.arange(1, 4).reshape(-1, 1) * 0.05
        batch = select_active_batch(
            pool,
            np.array([4.0, 3.0, 2.0, 1.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            geometry,
            1,
            1,
            2.0,
        )

        def exact_predictor(normalized):
            mean = normalized.sum(axis=1)
            return mean, np.full(mean.shape, 0.25)

        refreshed = refresh_batch_predictions(batch, geometry, exact_predictor, 2.0)
        exact_normalized = geometry.normalize(refreshed.designs).reshape(-1, 16)
        np.testing.assert_allclose(refreshed.predicted_mean, exact_normalized.sum(axis=1))
        np.testing.assert_allclose(refreshed.acquisition, refreshed.predicted_mean + 0.5)

    def test_active_batch_excludes_designs_already_in_dataset(self):
        config = TopoNaderOptimizationConfig(
            fe_batch_size=2,
            exploitation_count=1,
            exploration_count=1,
            ratio_target=1.0,
        )
        geometry = TopoNaderGeometry(config)
        pool = np.full((5, 16), 0.35)
        pool[1:] += np.arange(1, 5).reshape(-1, 1) * 0.08
        historical = geometry.finalize(
            geometry.denormalize(pool[0].reshape(1, 4, 4))[0]
        ).design.reshape(1, 4, 4)
        batch = select_active_batch(
            pool,
            np.array([10.0, 4.0, 3.0, 2.0, 1.0]),
            np.array([1.0, 0.4, 0.3, 0.2, 0.1]),
            geometry,
            1,
            1,
            2.0,
            excluded_designs=historical,
        )
        self.assertEqual(batch.designs.shape[0], 2)
        self.assertFalse(any(np.array_equal(design, historical[0]) for design in batch.designs))


if __name__ == "__main__":
    unittest.main()
