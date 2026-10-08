from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..agents.client import TopoNaderOpenAICompatibleClient, resolve_toponader_llm_runtime
from ..agents.orchestrator import TopoNaderAgentOrchestrator
from ..architecture.spec import TopoNaderArchitectureSpec
from ..architecture.validator import TopoNaderArchitectureValidator
from ..bridge.matlab import load_fe_result, save_fe_request
from ..config import TopoNaderConfig, save_config
from ..data.dataset import TopoNaderDataset, TopoNaderSplit
from ..logging.experience import TopoNaderExperienceStore
from ..logging.provenance import file_sha256, write_json_atomic
from ..monitoring.selection import select_provisional_architecture
from ..monitoring.trigger import TopoNaderTrigger
from ..optimization.acquisition import refresh_batch_predictions, select_active_batch
from ..optimization.geometry import TopoNaderGeometry
from ..optimization.pso import TopoNaderPSO
from ..reporting.summary import TopoNaderRunReporter
from ..training.evaluator import TopoNaderEvaluation, evaluate_predictions
from ..training.trainer import TopoNaderTrainer
from ..training.uncertainty import TopoNaderDeterministicPredictor, TopoNaderMCDropoutPredictor
from ..training.zero_cost import TopoNaderZeroCostEvaluator
from .state import TopoNaderRunState


def _evaluation_from_dict(payload: dict) -> TopoNaderEvaluation:
    return TopoNaderEvaluation(**payload)


def _evaluate_feedback(
    truth: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
) -> TopoNaderEvaluation:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if truth.size >= 2:
        return evaluate_predictions(truth, prediction, uncertainty, topk=min(3, truth.size))
    if truth.size == 1:
        absolute_error = float(abs(truth[0] - prediction[0]))
        return TopoNaderEvaluation(
            sample_count=1,
            rmse=absolute_error,
            mae=absolute_error,
            r2=0.0,
            kendall_tau=0.0,
            spearman_rho=0.0,
            pairwise_accuracy=0.0,
            topk_recall=1.0,
            topk_regret=0.0,
            uncertainty_error_correlation=None,
        )
    raise ValueError("no valid positive FE results were returned")


class TopoNaderEngine:
    def __init__(self, config: TopoNaderConfig, project_root: str | Path, agent_client_factory=None):
        config.validate()
        self.config = config
        self.project_root = Path(project_root).resolve()
        excluded = (self.project_root / config.paths.source_topology_dir / "Coord_True_coords.mat").resolve()
        self.geometry = TopoNaderGeometry(config.optimization, excluded)
        self.validator = TopoNaderArchitectureValidator(config.architecture)
        self.trainer = TopoNaderTrainer(config.training)
        self.agent_client_factory = agent_client_factory or TopoNaderOpenAICompatibleClient

    @property
    def default_architecture(self) -> TopoNaderArchitectureSpec:
        return TopoNaderArchitectureSpec()

    def initialize_run(self, run_dir: str | Path, force: bool = False) -> TopoNaderRunState:
        run = Path(run_dir)
        state_path = run / "state.json"
        if state_path.exists() and not force:
            return TopoNaderRunState.load(state_path)
        run.mkdir(parents=True, exist_ok=True)
        spec_path = run / "state" / "current_architecture.json"
        self.default_architecture.save(spec_path)
        state = TopoNaderRunState(
            current_architecture_id=self.default_architecture.architecture_id,
            audit_due=(
                self.config.method.architecture_adaptation
                and self.config.method.architecture_review_mode == "every_round"
            ),
        )
        write_json_atomic(state_path, state.to_dict())
        runtime_model, runtime_base_url = resolve_toponader_llm_runtime(self.config.agents)
        snapshot_config = replace(
            self.config,
            agents=replace(
                self.config.agents,
                model=runtime_model,
                base_url=runtime_base_url,
            ),
        )
        save_config(snapshot_config, run / "config.snapshot.json")
        return state

    def _normalized_split(self, dataset: TopoNaderDataset) -> TopoNaderSplit:
        split = dataset.chronological_split()
        return replace(
            split,
            x_train=self.geometry.normalize(split.x_train).reshape(-1, 16),
            x_validation=self.geometry.normalize(split.x_validation).reshape(-1, 16),
        )

    def _train_architecture_set(
        self,
        parent_bundle,
        child_specs: list[TopoNaderArchitectureSpec],
        split: TopoNaderSplit,
        iteration_dir: Path,
    ):
        quick = {}
        for spec in child_specs:
            model, transform, result = self.trainer.train(
                spec,
                split,
                iteration_dir / "audit" / "quick" / f"{spec.architecture_id}.pth",
                epochs=self.config.training.quick_epochs,
                output_log10=self.config.data.output_log10,
            )
            quick[spec.architecture_id] = (spec, result.validation)
        ordered = sorted(
            quick,
            key=lambda key: (
                quick[key][1].pairwise_accuracy,
                quick[key][1].kendall_tau,
                -quick[key][1].topk_regret,
                -quick[key][1].rmse,
            ),
            reverse=True,
        )
        parent_spec, parent_model, parent_transform, parent_result = parent_bundle
        selected = ordered[: max(0, self.config.architecture.fully_train_count - 1)]
        full = {
            parent_spec.architecture_id: (parent_spec, parent_model, parent_transform, parent_result)
        }
        for architecture_id in selected:
            spec = quick[architecture_id][0]
            model, transform, result = self.trainer.train(
                spec,
                split,
                iteration_dir / "audit" / "full" / f"{architecture_id}.pth",
                output_log10=self.config.data.output_log10,
            )
            full[architecture_id] = (spec, model, transform, result)
        winner_id = select_provisional_architecture({key: value[3].validation for key, value in full.items()})
        return full, winner_id, {key: value[1].to_dict() for key, value in quick.items()}

    def _audit(
        self,
        current: TopoNaderArchitectureSpec,
        split: TopoNaderSplit,
        iteration_dir: Path,
        run: Path,
        parent_bundle,
    ):
        if not self.config.agents.enabled:
            trace = {"skipped": "agents_disabled"}
            write_json_atomic(iteration_dir / "audit" / "trace.json", trace)
            return {current.architecture_id: parent_bundle}, current.architecture_id, trace
        metrics_path = run / "state" / "last_metrics.json"
        previous_feedback = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        _, _, _, parent_result = parent_bundle
        parent_report = self.validator.validate(current)
        metrics = {
            "previous_fe_feedback": previous_feedback,
            "current_parent_training": {
                "architecture_id": current.architecture_id,
                "parameter_count": parent_report.parameter_count,
                "best_epoch": parent_result.best_epoch,
                "best_validation_loss": parent_result.best_validation_loss,
                "validation": parent_result.validation.to_dict(),
                "last_training_history": parent_result.training_history[-5:],
            },
        }
        experience_path = run / self.config.paths.experience_file
        experiences = TopoNaderExperienceStore(experience_path).recent()
        client = self.agent_client_factory(self.config.agents)
        agents = TopoNaderAgentOrchestrator(client, iteration_dir / "agents")
        candidates, trace = agents.generate_candidates(
            current,
            metrics,
            experiences,
            self.config.architecture.candidate_count,
            validator=self.validator.validate,
            retry_limit=self.config.architecture.candidate_retry_limit,
        )
        zero_cost_reports = []
        if self.config.architecture.zero_cost_enabled and candidates:
            evaluator = TopoNaderZeroCostEvaluator(seed=self.config.training.seed)
            scored = []
            zero_cost_inputs = np.vstack([split.x_train, split.x_validation])
            for candidate in candidates:
                report = evaluator.evaluate(candidate, zero_cost_inputs)
                zero_cost_reports.append(report.to_dict())
                if report.valid:
                    scored.append((report.score, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)
            candidates = [
                candidate for _, candidate in scored[: self.config.architecture.zero_cost_keep_count]
            ]
        full, winner_id, quick = self._train_architecture_set(parent_bundle, candidates, split, iteration_dir)
        trace.update({
            "parent_diagnostics": metrics["current_parent_training"],
            "zero_cost_enabled": self.config.architecture.zero_cost_enabled,
            "zero_cost_reports": zero_cost_reports,
            "zero_cost_survivors": [candidate.architecture_id for candidate in candidates],
            "quick_screen": quick,
            "provisional_winner": winner_id,
        })
        write_json_atomic(iteration_dir / "audit" / "trace.json", trace)
        return full, winner_id, trace

    def prepare_iteration(self, run_dir: str | Path, dataset_path: str | Path, iteration: int) -> Path:
        run = Path(run_dir)
        state = self.initialize_run(run)
        if iteration != state.next_iteration:
            raise ValueError(f"expected iteration {state.next_iteration}, received {iteration}")
        iteration_dir = run / f"iteration_{iteration:04d}"
        if (iteration_dir / "fe_request.mat").exists():
            raise FileExistsError(f"iteration {iteration} already has an FE request")
        iteration_dir.mkdir(parents=True, exist_ok=True)
        dataset = TopoNaderDataset.from_mat(dataset_path, self.config.data)
        split = self._normalized_split(dataset)
        current_path = run / state.current_architecture_file
        current = TopoNaderArchitectureSpec.load(current_path)
        parent_model = None
        parent_transform = None
        audit_trace = None
        audit_requested = bool(state.audit_due and self.config.method.architecture_adaptation)
        if audit_requested:
            parent_model, parent_transform, parent_result = self.trainer.train(
                current,
                split,
                iteration_dir / "audit" / "full" / f"{current.architecture_id}.pth",
                output_log10=self.config.data.output_log10,
            )
            parent_bundle = (current, parent_model, parent_transform, parent_result)
            trained, winner_id, audit_trace = self._audit(
                current, split, iteration_dir, run, parent_bundle
            )
            provisional, model, transform, result = trained[winner_id]
            if current.architecture_id in trained:
                _, parent_model, parent_transform, _ = trained[current.architecture_id]
        else:
            model, transform, result = self.trainer.train(
                current,
                split,
                iteration_dir / "incumbent.pth",
                output_log10=self.config.data.output_log10,
            )
            provisional = current
        deterministic_predictor = TopoNaderDeterministicPredictor(model, transform)
        uncertainty_predictor = TopoNaderMCDropoutPredictor(
            model, transform, self.config.uncertainty.mc_samples, self.config.training.seed
        )

        def predict_mean(positions: np.ndarray) -> np.ndarray:
            return deterministic_predictor.predict(positions)[0]

        def predict_ucb(positions: np.ndarray) -> np.ndarray:
            mean, uncertainty = uncertainty_predictor.predict(positions)
            return mean + self.config.uncertainty.kappa * uncertainty

        seed_order = np.argsort(-dataset.outputs)[: min(10, dataset.outputs.size)]
        seeds = self.geometry.normalize(dataset.inputs[seed_order]).reshape(-1, 16)
        optimization = self.config.optimization
        mean_pso = TopoNaderPSO(
            optimization.particles,
            optimization.iterations,
            optimization.mutation_rate,
            optimization.stall_limit,
            self.config.training.seed,
        ).optimize(predict_mean, seeds=seeds)
        if self.config.method.uncertainty_sampling:
            ucb_pso = TopoNaderPSO(
                optimization.particles,
                optimization.iterations,
                optimization.mutation_rate,
                optimization.stall_limit,
                self.config.training.seed + 1,
            ).optimize(predict_ucb, seeds=seeds)
            mean_count = min(
                max(1, optimization.candidate_pool_size // 2), mean_pso.trajectory_positions.shape[0]
            )
            ucb_count = min(
                max(1, optimization.candidate_pool_size - mean_count), ucb_pso.trajectory_positions.shape[0]
            )
            mean_keep = np.argsort(-mean_pso.trajectory_scores)[:mean_count]
            ucb_keep = np.argsort(-ucb_pso.trajectory_scores)[:ucb_count]
            pool = np.vstack([
                mean_pso.trajectory_positions[mean_keep],
                ucb_pso.trajectory_positions[ucb_keep],
            ])
            mean, uncertainty = uncertainty_predictor.predict(pool)
            exploitation_count = optimization.exploitation_count
            exploration_count = optimization.exploration_count
            final_predictor = uncertainty_predictor.predict
        else:
            keep_count = min(optimization.candidate_pool_size, mean_pso.trajectory_positions.shape[0])
            keep = np.argsort(-mean_pso.trajectory_scores)[:keep_count]
            pool = mean_pso.trajectory_positions[keep]
            mean, uncertainty = deterministic_predictor.predict(pool)
            exploitation_count = optimization.fe_batch_size
            exploration_count = 0
            final_predictor = deterministic_predictor.predict
        batch = select_active_batch(
            pool,
            mean,
            uncertainty,
            self.geometry,
            exploitation_count,
            exploration_count,
            self.config.uncertainty.kappa,
            use_diversity=self.config.method.diversity_sampling,
            excluded_designs=dataset.inputs,
        )
        batch = refresh_batch_predictions(
            batch, self.geometry, final_predictor, self.config.uncertainty.kappa
        )
        extra = {
            "iteration": np.asarray([[iteration]], dtype=np.int64),
            "architecture_id": np.asarray([provisional.architecture_id], dtype=object),
        }
        if audit_requested and parent_model is not None and parent_transform is not None:
            parent_predictor = (
                TopoNaderMCDropoutPredictor(
                    parent_model,
                    parent_transform,
                    self.config.uncertainty.mc_samples,
                    self.config.training.seed,
                )
                if self.config.method.uncertainty_sampling
                else TopoNaderDeterministicPredictor(parent_model, parent_transform)
            )
            parent_mean, parent_uncertainty = parent_predictor.predict(
                self.geometry.normalize(batch.designs).reshape(-1, 16)
            )
            extra["parent_predicted_mean"] = parent_mean.reshape(-1, 1)
            extra["parent_predicted_uncertainty"] = parent_uncertainty.reshape(-1, 1)
            extra["parent_architecture_id"] = np.asarray([current.architecture_id], dtype=object)
        save_fe_request(iteration_dir / "fe_request.mat", batch, extra)
        provisional.save(iteration_dir / "provisional_architecture.json")
        manifest = {
            "iteration": iteration,
            "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_sha256": file_sha256(dataset_path),
            "sample_count": int(dataset.outputs.size),
            "audit": audit_requested,
            "parent_architecture_id": current.architecture_id,
            "provisional_architecture_id": provisional.architecture_id,
            "training": result.to_dict(),
            "method": self.config.method.__dict__,
            "requested_fe_samples": int(batch.designs.shape[0]),
            "audit_trace_file": "audit/trace.json" if audit_trace is not None else None,
        }
        write_json_atomic(iteration_dir / "manifest.json", manifest)
        return iteration_dir / "fe_request.mat"

    def apply_feedback(
        self,
        run_dir: str | Path,
        dataset_path: str | Path,
        iteration: int,
        fe_result_path: str | Path,
        next_dataset_path: str | Path,
    ) -> Path:
        run = Path(run_dir)
        state = TopoNaderRunState.load(run / "state.json")
        iteration_dir = run / f"iteration_{iteration:04d}"
        if iteration != state.next_iteration:
            raise ValueError(f"expected feedback for iteration {state.next_iteration}")
        request = loadmat(iteration_dir / "fe_request.mat")
        outputs = load_fe_result(fe_result_path)
        designs = np.asarray(request["inputs"], dtype=np.float64).reshape(-1, 4, 4)
        predicted = np.asarray(request["predicted_mean"], dtype=np.float64).reshape(-1)
        uncertainty = np.asarray(request["predicted_uncertainty"], dtype=np.float64).reshape(-1)
        if outputs.size != designs.shape[0]:
            raise ValueError("FE result count does not match the request")
        valid = np.isfinite(outputs) & (outputs > 0)
        metrics = _evaluate_feedback(outputs[valid], predicted[valid], uncertainty[valid])
        metric_payload = metrics.to_dict()
        inherited = TopoNaderArchitectureSpec.load(iteration_dir / "provisional_architecture.json")
        manifest = json.loads((iteration_dir / "manifest.json").read_text(encoding="utf-8"))
        parent_metrics = None
        post_reflection = None
        if manifest["audit"] and "parent_predicted_mean" in request:
            parent_prediction = np.asarray(request["parent_predicted_mean"], dtype=np.float64).reshape(-1)
            parent_uncertainty = np.asarray(request["parent_predicted_uncertainty"], dtype=np.float64).reshape(-1)
            parent_metrics = _evaluate_feedback(
                outputs[valid], parent_prediction[valid], parent_uncertainty[valid]
            )
            winner = select_provisional_architecture(
                {"parent": parent_metrics, "child": metrics}
            )
            if winner == "parent":
                inherited = TopoNaderArchitectureSpec.load(run / state.current_architecture_file)
            state.trigger.architecture_age = 0
            state.trigger.consecutive_degraded_rounds = 0
            state.trigger.last_audit_iteration = iteration
            state.audit_due = bool(
                self.config.method.architecture_adaptation
                and self.config.method.architecture_review_mode == "every_round"
            )
        else:
            if not self.config.method.architecture_adaptation:
                state.audit_due = False
                metric_payload["trigger_decision"] = {
                    "audit_required": False,
                    "reasons": ["architecture_adaptation_disabled"],
                }
            elif self.config.method.architecture_review_mode == "every_round":
                state.audit_due = True
                metric_payload["trigger_decision"] = {
                    "audit_required": True,
                    "reasons": ["every_round_review_mode"],
                }
            else:
                decision = TopoNaderTrigger(self.config.trigger).evaluate(iteration, metrics, state.trigger)
                state.trigger = decision.state
                state.audit_due = decision.audit_required
                metric_payload["trigger_decision"] = decision.to_dict()
        dataset = TopoNaderDataset.from_mat(dataset_path, self.config.data)
        updated = dataset.append(designs[valid], outputs[valid])
        updated.save_mat(next_dataset_path)
        inherited.save(run / state.current_architecture_file)
        state.current_architecture_id = inherited.architecture_id
        state.next_iteration = iteration + 1
        state.last_metrics_file = "state/last_metrics.json"
        write_json_atomic(run / "state" / "last_metrics.json", metric_payload)
        if manifest["audit"] and self.config.agents.enabled:
            reflection_input = {
                "iteration": iteration,
                "parent_architecture_id": manifest["parent_architecture_id"],
                "provisional_architecture_id": manifest["provisional_architecture_id"],
                "inherited_architecture_id": inherited.architecture_id,
                "parent_metrics": parent_metrics.to_dict() if parent_metrics else None,
                "provisional_metrics": metric_payload,
            }
            try:
                client = self.agent_client_factory(self.config.agents)
                agents = TopoNaderAgentOrchestrator(client, iteration_dir / "agents")
                post_reflection, reflection_usage = agents.reflector.reflect(reflection_input)
                write_json_atomic(iteration_dir / "audit" / "post_reflection.json", {
                    "reflection": post_reflection,
                    "usage": reflection_usage,
                })
            except Exception as exc:
                post_reflection = {"error": str(exc)}
                write_json_atomic(iteration_dir / "audit" / "post_reflection.json", post_reflection)
        write_json_atomic(iteration_dir / "feedback.json", {
            "valid_fe_samples": int(valid.sum()),
            "rejected_fe_samples": int((~valid).sum()),
            "provisional_metrics": metric_payload,
            "parent_metrics": parent_metrics.to_dict() if parent_metrics else None,
            "inherited_architecture_id": inherited.architecture_id,
            "next_audit_due": state.audit_due,
            "post_reflection": post_reflection,
        })
        write_json_atomic(run / "state.json", state.to_dict())
        TopoNaderExperienceStore(run / self.config.paths.experience_file).append({
            "iteration": iteration,
            "audit": manifest["audit"],
            "parent_architecture_id": manifest["parent_architecture_id"],
            "provisional_architecture_id": manifest["provisional_architecture_id"],
            "inherited_architecture_id": inherited.architecture_id,
            "metrics": metric_payload,
            "parent_metrics": parent_metrics.to_dict() if parent_metrics else None,
            "post_reflection": post_reflection,
        })
        reporter = TopoNaderRunReporter(run, self.config.reporting.target_performance)
        best_before = float(np.max(dataset.outputs))
        batch_best = float(np.max(outputs[valid]))
        reporter.update(iteration_dir, {
            "iteration": iteration,
            "dataset_size_before": int(dataset.outputs.size),
            "requested_fe_samples": int(outputs.size),
            "valid_fe_samples": int(valid.sum()),
            "failed_fe_samples": int((~valid).sum()),
            "batch_best_performance": batch_best,
            "best_performance_before": best_before,
            "best_performance_so_far": max(best_before, batch_best),
            "pairwise_accuracy": metrics.pairwise_accuracy,
            "kendall_tau": metrics.kendall_tau,
            "spearman_rho": metrics.spearman_rho,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2": metrics.r2,
            "topk_recall": metrics.topk_recall,
            "topk_regret": metrics.topk_regret,
            "uncertainty_error_correlation": metrics.uncertainty_error_correlation,
            "audit_performed": bool(manifest["audit"]),
            "architecture_changed": inherited.architecture_id != manifest["parent_architecture_id"],
            "rollback": bool(
                manifest["audit"]
                and manifest["provisional_architecture_id"] != manifest["parent_architecture_id"]
                and inherited.architecture_id == manifest["parent_architecture_id"]
            ),
            "parent_architecture_id": manifest["parent_architecture_id"],
            "provisional_architecture_id": manifest["provisional_architecture_id"],
            "inherited_architecture_id": inherited.architecture_id,
            "next_audit_due": state.audit_due,
            "method": manifest.get("method", {}),
        })
        return Path(next_dataset_path)
