from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..agents.client import TopoNaderOpenAICompatibleClient, resolve_toponader_llm_runtime
from ..agents.orchestrator import TopoNaderAgentOrchestrator
from ..agents.random_proposals import generate_random_candidates
from ..architecture.spec import TopoNaderArchitectureSpec
from ..architecture.validator import TopoNaderArchitectureValidator
from ..bridge.matlab import load_fe_result, save_fe_request
from ..config import TopoNaderConfig, TopoNaderEliteLocalSearchConfig, TopoNaderInheritanceConfig, save_config
from ..data.dataset import TopoNaderDataset, TopoNaderSplit
from ..logging.experience import TopoNaderExperienceStore
from ..logging.provenance import file_sha256, write_json_atomic
from ..monitoring.selection import select_provisional_architecture
from ..monitoring.trigger import TopoNaderTrigger
from ..optimization.acquisition import (
    combine_batches,
    finalize_candidate_pool,
    refresh_batch_predictions,
    select_active_batch,
    select_from_finalized_pool,
)
from ..optimization.geometry import TopoNaderGeometry
from ..optimization.pso import TopoNaderPSO
from ..reporting.summary import TopoNaderRunReporter
from ..training.evaluator import TopoNaderEvaluation, evaluate_predictions
from ..training.trainer import TopoNaderTrainer
from ..training.uncertainty import TopoNaderDeterministicPredictor, TopoNaderMCDropoutPredictor
from ..training.zero_cost import TopoNaderZeroCostEvaluator
from ..monitoring.inheritance import decide_inheritance
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


def _allocate_integer(total: int, weights: tuple[float, ...]) -> list[int]:
    total = int(total)
    values = np.asarray(weights, dtype=np.float64)
    if total < 0 or values.size == 0 or np.any(values < 0) or float(values.sum()) <= 0:
        raise ValueError("invalid weighted integer allocation")
    scaled = values / values.sum() * total
    allocation = np.floor(scaled).astype(int)
    remainder = total - int(allocation.sum())
    if remainder:
        order = np.argsort(-(scaled - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation.tolist()


class TopoNaderEngine:
    def __init__(self, config: TopoNaderConfig, project_root: str | Path, agent_client_factory=None):
        config.validate()
        self.config = config
        self.seeds = config.seed_manifest()
        self.data_config = replace(config.data, split_seed=self.seeds["data_split"])
        self.training_config = replace(config.training, seed=self.seeds["training"])
        self.project_root = Path(project_root).resolve()
        excluded = (self.project_root / config.paths.source_topology_dir / "Coord_True_coords.mat").resolve()
        self.geometry = TopoNaderGeometry(config.optimization, excluded, num_bars=config.data.num_bars)
        self.validator = TopoNaderArchitectureValidator(config.architecture, config.data.input_dim)
        self.trainer = TopoNaderTrainer(self.training_config)
        self.agent_client_factory = agent_client_factory or TopoNaderOpenAICompatibleClient

    @property
    def default_architecture(self) -> TopoNaderArchitectureSpec:
        return TopoNaderArchitectureSpec(input_dim=self.config.data.input_dim)

    def initialize_run(self, run_dir: str | Path, force: bool = False) -> TopoNaderRunState:
        run = Path(run_dir)
        state_path = run / "state.json"
        runtime_model, runtime_base_url = resolve_toponader_llm_runtime(self.config.agents)
        snapshot_config = replace(
            self.config,
            agents=replace(
                self.config.agents,
                model=runtime_model,
                base_url=runtime_base_url,
            ),
        )
        if state_path.exists() and not force:
            snapshot_path = run / "config.snapshot.json"
            if not snapshot_path.exists():
                raise ValueError(f"existing run is missing its config snapshot: {snapshot_path}")
            stored = json.loads(snapshot_path.read_text(encoding="utf-8"))
            # Old snapshots used no cooldown or FE inheritance guard. Normalize
            # to those legacy defaults, never to the requested new policy.
            stored.setdefault("inheritance", asdict(TopoNaderInheritanceConfig()))
            stored.setdefault("trigger", {}).setdefault("min_audit_interval", 0)
            stored.setdefault("agents", {}).setdefault("proposal_source", "llm")
            # Compare the JSON representation because tuple-valued settings are
            # persisted as lists in config.snapshot.json.
            resolved_snapshot = json.loads(json.dumps(snapshot_config.to_dict()))
            stored_agents = stored.get("agents")
            if isinstance(stored_agents, dict) and "structured_output_retries" not in stored_agents:
                stored = dict(stored)
                stored["agents"] = dict(stored_agents)
                stored["agents"]["structured_output_retries"] = resolved_snapshot["agents"][
                    "structured_output_retries"
                ]
            stored_optimization = stored.get("optimization")
            if isinstance(stored_optimization, dict) and "elite_local_search" not in stored_optimization:
                stored = dict(stored)
                stored["optimization"] = dict(stored_optimization)
                stored["optimization"]["elite_local_search"] = json.loads(
                    json.dumps(asdict(TopoNaderEliteLocalSearchConfig()))
                )
            if stored != resolved_snapshot:
                raise ValueError(
                    f"experiment directory already contains a different configuration: {run}"
                )
            return TopoNaderRunState.load(state_path)
        run.mkdir(parents=True, exist_ok=True)
        spec_path = run / "state" / "current_architecture.json"
        initial_architecture = replace(
            self.default_architecture,
            input_dim=self.config.data.input_dim,
        )
        initial_architecture.save(spec_path)
        state = TopoNaderRunState(
            current_architecture_id=initial_architecture.architecture_id,
            audit_due=(
                self.config.method.architecture_adaptation
                and self.config.method.architecture_review_mode == "every_round"
            ),
        )
        write_json_atomic(state_path, state.to_dict())
        save_config(snapshot_config, run / "config.snapshot.json")
        write_json_atomic(run / "seed_manifest.json", self.seeds)
        return state

    def _normalized_split(self, dataset: TopoNaderDataset) -> TopoNaderSplit:
        split = dataset.chronological_split()
        return replace(
            split,
            x_train=self.geometry.normalize(split.x_train).reshape(-1, self.geometry.input_dim),
            x_validation=self.geometry.normalize(split.x_validation).reshape(-1, self.geometry.input_dim),
        )

    def _local_initial_positions(
        self,
        elite_design: np.ndarray,
        count: int,
        rank: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        settings = self.config.optimization.elite_local_search
        center = self.geometry.normalize(
            np.asarray(elite_design, dtype=np.float64).reshape(1, *self.geometry.design_shape)
        ).reshape(-1)
        if rank == 0:
            physical_radius = np.tile(
                [
                    settings.small_position_radius,
                    settings.small_position_radius,
                    settings.small_angle_radius,
                    settings.small_width_radius,
                ],
                self.geometry.num_bars,
            )
        else:
            physical_radius = np.tile(
                [
                    settings.medium_position_radius,
                    settings.medium_position_radius,
                    settings.medium_angle_radius,
                    settings.medium_width_radius,
                ],
                self.geometry.num_bars,
            )
        span = (self.geometry.upper - self.geometry.lower).reshape(-1)
        radius = np.asarray(physical_radius, dtype=np.float64) / span
        lower = np.maximum(0.0, center - radius)
        upper = np.minimum(1.0, center + radius)
        rng = np.random.default_rng(seed)
        positions = np.repeat(center.reshape(1, -1), int(count), axis=0)
        for row in range(1, int(count)):
            draw = float(rng.random())
            if draw < settings.single_parameter_fraction:
                dimensions = [int(rng.integers(0, self.geometry.input_dim))]
            elif draw < (
                settings.single_parameter_fraction
                + settings.single_bar_two_parameter_fraction
            ):
                bar = int(rng.integers(0, self.geometry.num_bars))
                parameters = rng.choice(self.geometry.parameters_per_bar, size=2, replace=False)
                dimensions = [bar * self.geometry.parameters_per_bar + int(value) for value in parameters]
            else:
                bars = rng.choice(self.geometry.num_bars, size=min(2, self.geometry.num_bars), replace=False)
                dimensions = [
                    int(bar) * self.geometry.parameters_per_bar
                    + int(rng.integers(0, self.geometry.parameters_per_bar))
                    for bar in bars
                ]
            for dimension in dimensions:
                positions[row, dimension] += rng.uniform(-radius[dimension], radius[dimension])
        return np.clip(positions, lower, upper), lower, upper

    def _elite_local_exploitation(
        self,
        dataset: TopoNaderDataset,
        iteration: int,
        predictor,
        predict_mean,
        target_count: int,
    ):
        settings = self.config.optimization.elite_local_search
        elite_count = min(settings.elite_count, dataset.outputs.size)
        elite_order = np.argsort(-dataset.outputs)[:elite_count]
        particle_allocation = _allocate_integer(
            self.config.optimization.particles,
            tuple(settings.particle_weights[:elite_count]),
        )
        selected_allocation = _allocate_integer(
            target_count,
            tuple(settings.selected_weights[:elite_count]),
        )
        pools = []
        trace = {
            "enabled": True,
            "elite_dataset_indices": [int(index) for index in elite_order],
            "elite_true_outputs": [float(dataset.outputs[index]) for index in elite_order],
            "particle_allocation": particle_allocation,
            "selected_allocation": selected_allocation,
            "candidate_counts": [],
            "candidate_pool_statistics": [],
        }
        selected_batches = []
        selected_designs = np.empty((0, *self.geometry.design_shape), dtype=np.float64)
        rejection_counts = {"coverage": 0, "historical_duplicate": 0, "batch_duplicate": 0}
        for rank, (dataset_index, particle_count, quota) in enumerate(
            zip(elite_order, particle_allocation, selected_allocation)
        ):
            if particle_count <= 0:
                pools.append(None)
                trace["candidate_counts"].append(0)
                trace["candidate_pool_statistics"].append(None)
                continue
            local_seed = self.config.seed_for(f"pso_mean:{iteration}:elite:{rank}")
            initial, lower, upper = self._local_initial_positions(
                dataset.inputs[dataset_index], particle_count, rank, local_seed
            )
            result = TopoNaderPSO(
                particle_count,
                self.config.optimization.iterations,
                self.config.optimization.mutation_rate,
                self.config.optimization.stall_limit,
                local_seed,
            ).optimize(
                predict_mean,
                dimensions=self.geometry.input_dim,
                initial_positions=initial,
                lower_bounds=lower,
                upper_bounds=upper,
            )
            keep_count = min(
                max(settings.candidates_per_elite, quota * 20),
                result.trajectory_positions.shape[0],
            )
            keep = np.argsort(-result.trajectory_scores)[:keep_count]
            raw_pool = np.vstack([initial, result.trajectory_positions[keep]])
            raw_priority = np.concatenate(
                [result.trajectory_scores[:particle_count], result.trajectory_scores[keep]]
            )
            pool = finalize_candidate_pool(
                raw_pool,
                self.geometry,
                predictor,
                excluded_designs=dataset.inputs,
                ranking_mode="mean",
                raw_priority=raw_priority,
            )
            pools.append(pool)
            trace["candidate_counts"].append(int(pool.designs.shape[0]))
            trace["candidate_pool_statistics"].append(pool.finalization_stats)
            for key, value in pool.rejection_counts.items():
                rejection_counts[key] = rejection_counts.get(key, 0) + int(value)
            if quota <= 0:
                continue
            batch = select_from_finalized_pool(
                pool,
                quota,
                self.config.uncertainty.kappa,
                "exploitation",
                use_uncertainty=False,
                use_diversity=self.config.method.diversity_sampling,
                excluded_designs=selected_designs,
            )
            if batch.designs.shape[0]:
                selected_batches.append(replace(batch, rejection_counts=None))
                selected_designs = np.concatenate([selected_designs, batch.designs], axis=0)
        while selected_designs.shape[0] < target_count:
            progress = False
            for pool in pools:
                if pool is None or selected_designs.shape[0] >= target_count:
                    continue
                batch = select_from_finalized_pool(
                    pool,
                    target_count - selected_designs.shape[0],
                    self.config.uncertainty.kappa,
                    "exploitation",
                    use_uncertainty=False,
                    use_diversity=self.config.method.diversity_sampling,
                    excluded_designs=selected_designs,
                )
                if batch.designs.shape[0]:
                    selected_batches.append(replace(batch, rejection_counts=None))
                    selected_designs = np.concatenate([selected_designs, batch.designs], axis=0)
                    progress = True
            if not progress:
                break
        if selected_designs.shape[0] != target_count:
            raise RuntimeError(
                f"elite-local EPSO produced {selected_designs.shape[0]} unique exploitation designs; "
                f"expected {target_count}"
            )
        trace["selected_count"] = int(selected_designs.shape[0])
        combined = combine_batches(*selected_batches)
        combined.rejection_counts = rejection_counts
        return combined, trace

    def _global_exploration(
        self,
        dataset: TopoNaderDataset,
        iteration: int,
        predictor,
        predict_ucb,
        seeds: np.ndarray,
        excluded_designs: np.ndarray,
        target_count: int,
    ):
        seed = self.config.seed_for(f"pso_ucb:{iteration}")
        result = TopoNaderPSO(
            self.config.optimization.particles,
            self.config.optimization.iterations,
            self.config.optimization.mutation_rate,
            self.config.optimization.stall_limit,
            seed,
        ).optimize(predict_ucb, dimensions=self.geometry.input_dim, seeds=seeds)
        keep_count = min(
            max(self.config.optimization.candidate_pool_size * 10, target_count * 50),
            result.trajectory_positions.shape[0],
        )
        keep = np.argsort(-result.trajectory_scores)[:keep_count]
        pool = finalize_candidate_pool(
            result.trajectory_positions[keep],
            self.geometry,
            predictor,
            excluded_designs=np.concatenate([dataset.inputs, excluded_designs], axis=0),
            ranking_mode="ucb",
            kappa=self.config.uncertainty.kappa,
            raw_priority=result.trajectory_scores[keep],
        )
        batches = []
        selected = np.empty((0, *self.geometry.design_shape), dtype=np.float64)
        batch = select_from_finalized_pool(
            pool,
            target_count,
            self.config.uncertainty.kappa,
            "exploration",
            use_uncertainty=True,
            use_diversity=self.config.method.diversity_sampling,
        )
        if batch.designs.shape[0]:
            batches.append(batch)
            selected = batch.designs
        rng = np.random.default_rng(self.config.seed_for(f"exploration_refill:{iteration}"))
        attempts = 0
        refill_statistics = []
        while selected.shape[0] < target_count and attempts < 4:
            attempts += 1
            random_pool = rng.random((max(500, target_count * 100), self.geometry.input_dim))
            finalized = finalize_candidate_pool(
                random_pool,
                self.geometry,
                predictor,
                excluded_designs=np.concatenate(
                    [dataset.inputs, excluded_designs, selected], axis=0
                ),
                ranking_mode="ucb",
                kappa=self.config.uncertainty.kappa,
            )
            refill_statistics.append(finalized.finalization_stats)
            refill = select_from_finalized_pool(
                finalized,
                target_count - selected.shape[0],
                self.config.uncertainty.kappa,
                "exploration",
                use_uncertainty=True,
                use_diversity=self.config.method.diversity_sampling,
            )
            if refill.designs.shape[0]:
                batches.append(refill)
                selected = np.concatenate([selected, refill.designs], axis=0)
        if selected.shape[0] != target_count:
            raise RuntimeError(
                f"global UCB-EPSO produced {selected.shape[0]} unique exploration designs; "
                f"expected {target_count}"
            )
        return combine_batches(*batches), {
            "candidate_count": int(pool.designs.shape[0]),
            "refill_attempts": attempts,
            "selected_count": int(selected.shape[0]),
            "candidate_pool_statistics": pool.finalization_stats,
            "refill_pool_statistics": refill_statistics,
        }

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
        if self.config.agents.proposal_source == "random":
            iteration = int(iteration_dir.name.rsplit("_", 1)[1])
            candidates, trace = generate_random_candidates(
                current, self.config.architecture,
                self.config.seed_for(f"random_architecture:{iteration}"),
                self.config.architecture.candidate_count, self.validator.validate,
                self.config.architecture.candidate_retry_limit,
            )
        else:
            experience_path = run / self.config.paths.experience_file
            experiences = TopoNaderExperienceStore(experience_path).recent()
            client = self.agent_client_factory(self.config.agents)
            agents = TopoNaderAgentOrchestrator(
                client,
                iteration_dir / "agents",
                structured_output_retries=self.config.agents.structured_output_retries,
            )
            candidates, trace = agents.generate_candidates(
                current,
                metrics,
                experiences,
                self.config.architecture.candidate_count,
                validator=self.validator.validate,
                retry_limit=self.config.architecture.candidate_retry_limit,
            )
            trace["proposal_source"] = "llm"
        zero_cost_reports = []
        if self.config.architecture.zero_cost_enabled and candidates:
            evaluator = TopoNaderZeroCostEvaluator(seed=self.seeds["zero_cost"])
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
        source_dataset_path = Path(dataset_path).resolve()
        dataset_snapshot = run / "datasets" / f"dataset_{iteration:04d}.mat"
        if iteration == 1:
            dataset_snapshot.parent.mkdir(parents=True, exist_ok=True)
            if dataset_snapshot.exists():
                if file_sha256(dataset_snapshot) != file_sha256(source_dataset_path):
                    raise ValueError(f"initial dataset snapshot differs from the supplied dataset: {dataset_snapshot}")
            elif source_dataset_path != dataset_snapshot.resolve():
                shutil.copy2(source_dataset_path, dataset_snapshot)
        dataset = TopoNaderDataset.from_mat(source_dataset_path, self.data_config)
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
            model, transform, self.config.uncertainty.mc_samples, self.seeds["mc_dropout"]
        )

        def predict_mean(positions: np.ndarray) -> np.ndarray:
            return deterministic_predictor.predict(positions)[0]

        def predict_ucb(positions: np.ndarray) -> np.ndarray:
            mean, uncertainty = uncertainty_predictor.predict(positions)
            return mean + self.config.uncertainty.kappa * uncertainty

        seed_order = np.argsort(-dataset.outputs)[: min(10, dataset.outputs.size)]
        seeds = self.geometry.normalize(dataset.inputs[seed_order]).reshape(-1, self.geometry.input_dim)
        optimization = self.config.optimization
        search_trace = {"elite_local_search": {"enabled": False}}
        if optimization.elite_local_search.enabled:
            exploitation_target = (
                optimization.exploitation_count
                if self.config.method.uncertainty_sampling
                else optimization.fe_batch_size
            )
            exploitation_batch, elite_trace = self._elite_local_exploitation(
                dataset,
                iteration,
                uncertainty_predictor.predict,
                predict_mean,
                exploitation_target,
            )
            search_trace["elite_local_search"] = elite_trace
            batches = [exploitation_batch]
            if self.config.method.uncertainty_sampling and optimization.exploration_count:
                exploration_batch, exploration_trace = self._global_exploration(
                    dataset,
                    iteration,
                    uncertainty_predictor.predict,
                    predict_ucb,
                    seeds,
                    exploitation_batch.designs,
                    optimization.exploration_count,
                )
                batches.append(exploration_batch)
                search_trace["global_exploration"] = exploration_trace
            batch = combine_batches(*batches)
            if batch.designs.shape[0] != optimization.fe_batch_size:
                raise RuntimeError(
                    f"elite-local search prepared {batch.designs.shape[0]} FE designs; "
                    f"expected {optimization.fe_batch_size}"
                )
        else:
            mean_pso = TopoNaderPSO(
                optimization.particles,
                optimization.iterations,
                optimization.mutation_rate,
                optimization.stall_limit,
                self.config.seed_for(f"pso_mean:{iteration}"),
            ).optimize(predict_mean, dimensions=self.geometry.input_dim, seeds=seeds)
            if self.config.method.uncertainty_sampling:
                ucb_pso = TopoNaderPSO(
                    optimization.particles,
                    optimization.iterations,
                    optimization.mutation_rate,
                    optimization.stall_limit,
                    self.config.seed_for(f"pso_ucb:{iteration}"),
                ).optimize(predict_ucb, dimensions=self.geometry.input_dim, seeds=seeds)
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
            "num_bars": np.asarray([[self.config.data.num_bars]], dtype=np.int64),
            "parameters_per_bar": np.asarray(
                [[self.config.data.parameters_per_bar]], dtype=np.int64
            ),
            "random_seed": np.asarray([[self.config.random_seed]], dtype=np.int64),
        }
        if audit_requested and parent_model is not None and parent_transform is not None:
            parent_predictor = (
                TopoNaderMCDropoutPredictor(
                    parent_model,
                    parent_transform,
                    self.config.uncertainty.mc_samples,
                    self.seeds["mc_dropout"],
                )
                if self.config.method.uncertainty_sampling
                else TopoNaderDeterministicPredictor(parent_model, parent_transform)
            )
            parent_mean, parent_uncertainty = parent_predictor.predict(
                self.geometry.normalize(batch.designs).reshape(-1, self.geometry.input_dim)
            )
            extra["parent_predicted_mean"] = parent_mean.reshape(-1, 1)
            extra["parent_predicted_uncertainty"] = parent_uncertainty.reshape(-1, 1)
            extra["parent_architecture_id"] = np.asarray([current.architecture_id], dtype=object)
        save_fe_request(iteration_dir / "fe_request.mat", batch, extra)
        provisional.save(iteration_dir / "provisional_architecture.json")
        manifest = {
            "iteration": iteration,
            "dataset_path": str(source_dataset_path),
            "dataset_snapshot": str(dataset_snapshot.resolve()) if dataset_snapshot.exists() else None,
            "dataset_sha256": file_sha256(source_dataset_path),
            "sample_count": int(dataset.outputs.size),
            "design_space": {
                "num_bars": self.config.data.num_bars,
                "parameters_per_bar": self.config.data.parameters_per_bar,
                "input_dim": self.config.data.input_dim,
            },
            "seeds": self.seeds,
            "audit": audit_requested,
            "parent_architecture_id": current.architecture_id,
            "provisional_architecture_id": provisional.architecture_id,
            "training": result.to_dict(),
            "method": self.config.method.__dict__,
            "requested_fe_samples": int(batch.designs.shape[0]),
            "coverage": {
                "constraint": self.config.optimization.coverage.__dict__,
                "actual_min": float(np.min(batch.coverage_ratios)),
                "actual_max": float(np.max(batch.coverage_ratios)),
                "actual_mean": float(np.mean(batch.coverage_ratios)),
                "actual_std": float(np.std(batch.coverage_ratios)),
                "rejection_counts": batch.rejection_counts,
            },
            "search": search_trace,
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
        state = self.initialize_run(run)
        iteration_dir = run / f"iteration_{iteration:04d}"
        if iteration != state.next_iteration:
            raise ValueError(f"expected feedback for iteration {state.next_iteration}")
        request = loadmat(iteration_dir / "fe_request.mat")
        outputs = load_fe_result(fe_result_path)
        designs = np.asarray(request["inputs"], dtype=np.float64).reshape(-1, *self.geometry.design_shape)
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
        inheritance_decision = None
        inherited_metric_payload = metric_payload
        if manifest["audit"] and "parent_predicted_mean" in request:
            parent_prediction = np.asarray(request["parent_predicted_mean"], dtype=np.float64).reshape(-1)
            parent_uncertainty = np.asarray(request["parent_predicted_uncertainty"], dtype=np.float64).reshape(-1)
            parent_metrics = _evaluate_feedback(
                outputs[valid], parent_prediction[valid], parent_uncertainty[valid]
            )
            inheritance_decision = decide_inheritance(
                parent_metrics, metrics, self.config.inheritance,
                distinct_child=(inherited.architecture_id != manifest["parent_architecture_id"]),
            )
            if not inheritance_decision["accepted"]:
                inherited = TopoNaderArchitectureSpec.load(run / state.current_architecture_file)
                inherited_metric_payload = parent_metrics.to_dict()
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
        dataset = TopoNaderDataset.from_mat(dataset_path, self.data_config)
        updated = dataset.append(designs[valid], outputs[valid])
        updated.save_mat(next_dataset_path)
        inherited.save(run / state.current_architecture_file)
        state.current_architecture_id = inherited.architecture_id
        state.next_iteration = iteration + 1
        state.last_metrics_file = "state/last_metrics.json"
        # Subsequent diagnosis must describe the inherited model, including rollback.
        write_json_atomic(run / "state" / "last_metrics.json", inherited_metric_payload)
        if (manifest["audit"] and self.config.agents.enabled
                and self.config.agents.proposal_source == "llm"):
            reflection_input = {
                "iteration": iteration,
                "parent_architecture_id": manifest["parent_architecture_id"],
                "provisional_architecture_id": manifest["provisional_architecture_id"],
                "inherited_architecture_id": inherited.architecture_id,
                "parent_metrics": parent_metrics.to_dict() if parent_metrics else None,
                "provisional_metrics": metric_payload,
                "inheritance_decision": inheritance_decision,
            }
            try:
                client = self.agent_client_factory(self.config.agents)
                agents = TopoNaderAgentOrchestrator(
                    client,
                    iteration_dir / "agents",
                    structured_output_retries=self.config.agents.structured_output_retries,
                )
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
            "inheritance_decision": inheritance_decision,
            "inherited_metrics": inherited_metric_payload,
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
            "inheritance_decision": inheritance_decision,
            "inherited_metrics": inherited_metric_payload,
            "parent_metrics": parent_metrics.to_dict() if parent_metrics else None,
            "post_reflection": post_reflection,
        })
        reporter = TopoNaderRunReporter(run, self.config.reporting.target_performance)
        best_before = float(np.max(dataset.outputs))
        batch_best = float(np.max(outputs[valid]))
        requested_coverage = np.asarray(request.get("coverage_ratios", []), dtype=np.float64).reshape(-1)
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
            "inheritance_decision": inheritance_decision,
            "inheritance_accepted": inheritance_decision["accepted"] if inheritance_decision else None,
            "inheritance_reasons": ";".join(inheritance_decision["reasons"]) if inheritance_decision else None,
            "inheritance_pairwise_improvement": inheritance_decision["pairwise_improvement"] if inheritance_decision else None,
            "inheritance_topk_regret_increase": inheritance_decision["topk_regret_increase"] if inheritance_decision else None,
            "coverage_actual_min": float(np.min(requested_coverage)) if requested_coverage.size else None,
            "coverage_actual_max": float(np.max(requested_coverage)) if requested_coverage.size else None,
            "coverage_actual_mean": float(np.mean(requested_coverage)) if requested_coverage.size else None,
            "coverage_actual_std": float(np.std(requested_coverage)) if requested_coverage.size else None,
            "coverage_actual_count": int(requested_coverage.size),
            "coverage_actual_sum": float(np.sum(requested_coverage)) if requested_coverage.size else 0.0,
            "coverage_actual_sum_squares": float(np.sum(requested_coverage**2)) if requested_coverage.size else 0.0,
        })
        return Path(next_dataset_path)
