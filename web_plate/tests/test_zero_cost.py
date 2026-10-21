import unittest

import numpy as np

from topo_nader.architecture import TopoNaderArchitectureSpec
from topo_nader.training.zero_cost import TopoNaderZeroCostEvaluator


class TopoNaderZeroCostTests(unittest.TestCase):
    def test_training_free_report_is_finite_for_baseline(self):
        inputs = np.random.default_rng(3).random((8, 16))
        report = TopoNaderZeroCostEvaluator(seed=3, max_samples=6).evaluate(
            TopoNaderArchitectureSpec(), inputs
        )
        self.assertTrue(np.isfinite(report.score))
        self.assertEqual(report.architecture_id, TopoNaderArchitectureSpec().architecture_id)
        self.assertGreater(report.input_gradient_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

