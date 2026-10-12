import unittest

import numpy as np

from topo_nader.config import TopoNaderCoverageConfig, TopoNaderOptimizationConfig
from topo_nader.optimization import (
    TopoNaderGeometry,
    TopoNaderPSO,
    finalize_candidate_pool,
    refresh_batch_predictions,
    select_active_batch,
)


class TopoNaderOptimizationTests(unittest.TestCase):
    def test_pso_maximizes_objective(self):
        target = np.full(16, 0.75)
        pso = TopoNaderPSO(particles=20, iterations=20, seed=7)
        result = pso.optimize(lambda x: -np.sum((x - target) ** 2, axis=1), dimensions=16)
        self.assertGreater(float(np.max(result.scores)), -1.0)

    def test_pso_respects_local_bounds_and_initial_positions(self):
        initial = np.array(
            [
                [0.20, 0.40, 0.60],
                [0.25, 0.45, 0.65],
                [0.30, 0.50, 0.70],
            ]
        )
        lower = np.array([0.15, 0.35, 0.55])
        upper = np.array([0.35, 0.55, 0.75])
        pso = TopoNaderPSO(particles=3, iterations=4, mutation_rate=1.0, seed=13)
        result = pso.optimize(
            lambda x: x.sum(axis=1),
            dimensions=3,
            initial_positions=initial,
            lower_bounds=lower,
            upper_bounds=upper,
        )
        np.testing.assert_allclose(result.trajectory_positions[:3], initial)
        self.assertTrue(np.all(result.trajectory_positions >= lower))
        self.assertTrue(np.all(result.trajectory_positions <= upper))

    def test_active_batch_contains_exploitation_and_exploration(self):
        config = TopoNaderOptimizationConfig(
            fe_batch_size=4,
            exploitation_count=2,
            exploration_count=2,
            coverage=TopoNaderCoverageConfig(upper_bound=1.0, upper_tolerance=0.0),
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
            coverage=TopoNaderCoverageConfig(upper_bound=1.0, upper_tolerance=0.0),
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

    def test_finalized_pool_reports_repair_duplicates_and_rank_stability(self):
        config = TopoNaderOptimizationConfig(
            coverage=TopoNaderCoverageConfig(
                upper_bound_enabled=True,
                upper_bound=0.20,
                upper_tolerance=0.01,
            )
        )
        geometry = TopoNaderGeometry(config)
        pool = np.full((3, 16), 0.5)
        pool[:, 3::4] = 1.0
        pool[2, 0] = 0.7

        def predictor(normalized):
            mean = normalized.sum(axis=1)
            return mean, np.zeros_like(mean)

        raw_priority = predictor(pool)[0]
        finalized = finalize_candidate_pool(
            pool,
            geometry,
            predictor,
            ranking_mode="mean",
            raw_priority=raw_priority,
        )
        stats = finalized.finalization_stats
        self.assertEqual(stats["input_candidate_count"], 3)
        self.assertGreater(stats["coverage_repaired_count"], 0)
        self.assertGreater(stats["width_absolute_change_max"], 0.0)
        self.assertGreater(stats["original_coverage_mean"], stats["finalized_coverage_mean"])
        self.assertGreaterEqual(stats["final_structure_duplicate_count"], 1)
        self.assertEqual(
            stats["ranking_stability"]["compared_count"], finalized.designs.shape[0]
        )

    def test_active_batch_excludes_designs_already_in_dataset(self):
        config = TopoNaderOptimizationConfig(
            fe_batch_size=2,
            exploitation_count=1,
            exploration_count=1,
            coverage=TopoNaderCoverageConfig(upper_bound=1.0, upper_tolerance=0.0),
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

    def test_geometry_supports_six_bars(self):
        config = TopoNaderOptimizationConfig(
            coverage=TopoNaderCoverageConfig(upper_bound_enabled=False)
        )
        geometry = TopoNaderGeometry(config, num_bars=6)
        normalized = np.full((3, 24), 0.5)
        designs = geometry.denormalize(normalized)
        self.assertEqual(designs.shape, (3, 6, 4))
        self.assertEqual(geometry.normalize(designs).reshape(3, -1).shape, (3, 24))

    def test_disabled_lower_bound_accepts_low_coverage(self):
        config = TopoNaderOptimizationConfig(
            coverage=TopoNaderCoverageConfig(
                upper_bound_enabled=False,
                lower_bound_enabled=False,
            )
        )
        geometry = TopoNaderGeometry(config, num_bars=2)
        design = geometry.lower.copy()
        result = geometry.finalize(design)
        self.assertTrue(result.valid)

    def test_enabled_coverage_bounds_repair_widths(self):
        config = TopoNaderOptimizationConfig(
            coverage=TopoNaderCoverageConfig(
                upper_bound_enabled=True,
                upper_bound=0.20,
                upper_tolerance=0.01,
                lower_bound_enabled=True,
                lower_bound=0.10,
            )
        )
        geometry = TopoNaderGeometry(config, num_bars=4)
        design = geometry.upper.copy()
        result = geometry.finalize(design)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(result.coverage_ratio, 0.10)
        self.assertLessEqual(result.coverage_ratio, 0.21)


if __name__ == "__main__":
    unittest.main()
