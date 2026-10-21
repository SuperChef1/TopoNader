from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from ..logging.provenance import write_json_atomic


class TopoNaderRunReporter:
    """Write a standalone iteration report and refresh cumulative run reports."""

    def __init__(self, run_dir: str | Path, target_performance: float | None = None):
        self.run_dir = Path(run_dir)
        self.target_performance = target_performance

    def _load_rows(self) -> list[dict]:
        path = self.run_dir / "iterations.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def update(self, iteration_dir: str | Path, record: dict) -> dict:
        iteration_dir = Path(iteration_dir)
        rows_by_iteration = {int(row["iteration"]): row for row in self._load_rows()}
        rows_by_iteration[int(record["iteration"])] = record
        rows = [rows_by_iteration[key] for key in sorted(rows_by_iteration)]
        initial_calls = int(rows[0]["dataset_size_before"])
        running_calls = initial_calls
        for row in rows:
            running_calls += int(row["requested_fe_samples"])
            row["cumulative_fe_calls"] = running_calls
        write_json_atomic(iteration_dir / "iteration_summary.json", rows_by_iteration[int(record["iteration"])])
        jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self._atomic_text(self.run_dir / "iterations.jsonl", jsonl)

        columns = [
            "iteration",
            "requested_fe_samples",
            "valid_fe_samples",
            "failed_fe_samples",
            "cumulative_fe_calls",
            "batch_best_performance",
            "best_performance_so_far",
            "pairwise_accuracy",
            "kendall_tau",
            "spearman_rho",
            "rmse",
            "mae",
            "r2",
            "topk_recall",
            "topk_regret",
            "uncertainty_error_correlation",
            "audit_performed",
            "architecture_changed",
            "rollback",
            "parent_architecture_id",
            "provisional_architecture_id",
            "inherited_architecture_id",
            "next_audit_due",
            "coverage_actual_min",
            "coverage_actual_max",
            "coverage_actual_mean",
            "coverage_actual_std",
        ]
        csv_temp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", delete=False, dir=self.run_dir, suffix=".tmp"
        )
        try:
            writer = csv.DictWriter(csv_temp, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            csv_temp.close()
            os.replace(csv_temp.name, self.run_dir / "iterations.csv")
        finally:
            if not csv_temp.closed:
                csv_temp.close()
            if os.path.exists(csv_temp.name):
                os.unlink(csv_temp.name)

        cumulative_new_calls = sum(int(row["requested_fe_samples"]) for row in rows)
        cumulative_calls = initial_calls + cumulative_new_calls
        cumulative_valid = sum(int(row["valid_fe_samples"]) for row in rows)
        cumulative_failed = sum(int(row["failed_fe_samples"]) for row in rows)
        calls_so_far = initial_calls
        calls_to_target = None
        if self.target_performance is not None:
            if float(rows[0]["best_performance_before"]) >= self.target_performance:
                calls_to_target = initial_calls
            else:
                for row in rows:
                    calls_so_far += int(row["requested_fe_samples"])
                    if float(row["best_performance_so_far"]) >= self.target_performance:
                        calls_to_target = calls_so_far
                        break
        coverage_count = sum(int(row.get("coverage_actual_count", 0)) for row in rows)
        coverage_sum = sum(float(row.get("coverage_actual_sum", 0.0)) for row in rows)
        coverage_sum_squares = sum(float(row.get("coverage_actual_sum_squares", 0.0)) for row in rows)
        coverage_mean = coverage_sum / coverage_count if coverage_count else None
        coverage_variance = (
            max(0.0, coverage_sum_squares / coverage_count - coverage_mean**2)
            if coverage_count
            else None
        )
        coverage_mins = [row.get("coverage_actual_min") for row in rows if row.get("coverage_actual_min") is not None]
        coverage_maxes = [row.get("coverage_actual_max") for row in rows if row.get("coverage_actual_max") is not None]
        summary = {
            "iterations_completed": len(rows),
            "last_iteration": int(rows[-1]["iteration"]),
            "initial_dataset_fe_samples": initial_calls,
            "cumulative_new_fe_calls": cumulative_new_calls,
            "cumulative_fe_calls": cumulative_calls,
            "cumulative_valid_fe_results": cumulative_valid,
            "cumulative_failed_fe_results": cumulative_failed,
            "best_performance": max(float(row["best_performance_so_far"]) for row in rows),
            "audits_performed": sum(bool(row["audit_performed"]) for row in rows),
            "architecture_changes": sum(bool(row["architecture_changed"]) for row in rows),
            "rollbacks": sum(bool(row["rollback"]) for row in rows),
            "target_performance": self.target_performance,
            "target_reached": calls_to_target is not None,
            "fe_calls_to_target_upper_bound": calls_to_target,
            "coverage_actual_min": min(coverage_mins) if coverage_mins else None,
            "coverage_actual_max": max(coverage_maxes) if coverage_maxes else None,
            "coverage_actual_mean": coverage_mean,
            "coverage_actual_std": coverage_variance**0.5 if coverage_variance is not None else None,
        }
        write_json_atomic(self.run_dir / "summary.json", summary)
        return summary
