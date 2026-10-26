import unittest

import numpy as np

from topo_nader.config import TopoNaderTriggerConfig, TopoNaderInheritanceConfig
from topo_nader.monitoring.inheritance import decide_inheritance
from dataclasses import replace
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

    def test_cooldown_uses_next_prepare_and_does_not_mark_scheduled_audit_completed(self):
        config = TopoNaderTriggerConfig(patience=1, min_audit_interval=5)
        trigger = TopoNaderTrigger(config)
        bad = evaluate_predictions(np.arange(4.), np.arange(4.)[::-1])
        state = TopoNaderTriggerState(last_audit_iteration=10)
        blocked = trigger.evaluate(13, bad, state)  # next prepare 14 is too soon
        self.assertFalse(blocked.audit_required)
        self.assertIn('minimum_audit_interval_not_met', blocked.reasons)
        ready = trigger.evaluate(14, bad, blocked.state)  # prepare 15 is eligible
        self.assertTrue(ready.audit_required)
        self.assertEqual(ready.state.last_audit_iteration, 10)

    def test_first_audit_obeys_cooldown_and_max_age_still_triggers(self):
        good = evaluate_predictions(np.arange(4.), np.arange(4.))
        trigger = TopoNaderTrigger(TopoNaderTriggerConfig(min_audit_interval=5))
        decision = trigger.evaluate(10, good, TopoNaderTriggerState(architecture_age=9))
        self.assertTrue(decision.audit_required)
        self.assertIn('maximum_architecture_age', decision.reasons)
        bad = evaluate_predictions(np.arange(4.), np.arange(4.)[::-1])
        trigger = TopoNaderTrigger(TopoNaderTriggerConfig(patience=1, min_audit_interval=5))
        self.assertFalse(trigger.evaluate(1, bad, TopoNaderTriggerState()).audit_required)

    def test_fe_inheritance_gain_and_topk_protection(self):
        parent = replace(evaluate_predictions(np.arange(20.), np.arange(20.)),
                         pairwise_accuracy=.6, topk_regret=.1)
        config = TopoNaderInheritanceConfig(enabled=True)
        cases = [
            (.62, .1, False, 'pairwise_gain_too_small'),
            (.7, .13, False, 'topk_regret_degraded'),
            (.63, .12, True, 'ranking_gain_and_topk_protection_passed'),
        ]
        for accuracy, regret, accepted, reason in cases:
            with self.subTest(accuracy=accuracy, regret=regret):
                child = replace(parent, pairwise_accuracy=accuracy, topk_regret=regret)
                result = decide_inheritance(parent, child, config)
                self.assertEqual(result['accepted'], accepted)
                self.assertIn(reason, result['reasons'])
        child = replace(parent, pairwise_accuracy=.8, sample_count=1)
        self.assertIn('insufficient_valid_feedback', decide_inheritance(parent, child, config)['reasons'])
        self.assertFalse(decide_inheritance(parent, replace(parent, rmse=float('nan')), config)['accepted'])
        self.assertFalse(decide_inheritance(parent, parent, config, distinct_child=False)['accepted'])

    def test_absolute_quality_gate_optional_and_legacy_switch_supported(self):
        parent = replace(evaluate_predictions(np.arange(20.), np.arange(20.)), pairwise_accuracy=.3)
        child = replace(parent, pairwise_accuracy=.4)
        self.assertTrue(decide_inheritance(parent, child, TopoNaderInheritanceConfig(enabled=True))['accepted'])
        self.assertFalse(decide_inheritance(parent, child, TopoNaderInheritanceConfig(
            enabled=True, min_pairwise_accuracy=.55))['accepted'])
        child = replace(parent, pairwise_accuracy=.301, topk_regret=.5)
        self.assertTrue(decide_inheritance(parent, child, TopoNaderInheritanceConfig(enabled=False))['accepted'])


if __name__ == "__main__":
    unittest.main()

