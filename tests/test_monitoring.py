import unittest

import numpy as np

from topo_nader.config import TopoNaderTriggerConfig
from topo_nader.monitoring.trigger import TopoNaderTrigger, TopoNaderTriggerState
from topo_nader.training.evaluator import evaluate_predictions


class TopoNaderMonitoringTests(unittest.TestCase):
    def test_ranking_metrics_reward_correct_order(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate_predictions(truth, truth + 0.01, topk=2)
        self.assertAlmostEqual(metrics.pairwise_accuracy, 1.0)
        self.assertAlmostEqual(metrics.topk_recall, 1.0)
        self.assertAlmostEqual(metrics.topk_regret, 0.0)

    def test_trigger_waits_for_persistent_degradation(self):
        config = TopoNaderTriggerConfig(patience=2, max_architecture_age=20, min_feedback_samples=4)
        trigger = TopoNaderTrigger(config)
        bad = evaluate_predictions(np.arange(4.0), np.arange(4.0)[::-1], topk=2)
        first = trigger.evaluate(1, bad, TopoNaderTriggerState())
        second = trigger.evaluate(2, bad, first.state)
        self.assertFalse(first.audit_required)
        self.assertTrue(second.audit_required)
        self.assertIn("persistent_ranking_degradation", second.reasons)


if __name__ == "__main__":
    unittest.main()

