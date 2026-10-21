import json
import tempfile
import unittest
from pathlib import Path

from topo_nader.reporting import TopoNaderRunReporter


def _row(iteration, before, batch, requested=4, valid=4, failed=0):
    return {
        "iteration": iteration,
        "dataset_size_before": 10 + (iteration - 1) * valid,
        "requested_fe_samples": requested,
        "valid_fe_samples": valid,
        "failed_fe_samples": failed,
        "batch_best_performance": batch,
        "best_performance_before": before,
        "best_performance_so_far": max(before, batch),
        "pairwise_accuracy": 0.75,
        "kendall_tau": 0.5,
        "topk_regret": 0.1,
        "audit_performed": False,
        "architecture_changed": False,
        "rollback": False,
        "parent_architecture_id": "parent",
        "provisional_architecture_id": "parent",
        "inherited_architecture_id": "parent",
        "next_audit_due": False,
    }


class TopoNaderReportingTests(unittest.TestCase):
    def test_each_feedback_refreshes_standalone_and_cumulative_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = TopoNaderRunReporter(root, target_performance=2.0)
            reporter.update(root / "iteration_0001", _row(1, 1.0, 1.5))
            summary = reporter.update(root / "iteration_0002", _row(2, 1.5, 2.2))
            first = json.loads((root / "iteration_0001" / "iteration_summary.json").read_text())
        self.assertEqual(first["cumulative_fe_calls"], 14)
        self.assertEqual(summary["cumulative_fe_calls"], 18)
        self.assertEqual(summary["fe_calls_to_target_upper_bound"], 18)


if __name__ == "__main__":
    unittest.main()

