import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from topo_nader.architecture import TopoNaderArchitectureSpec
from topo_nader.config import (
    TopoNaderAgentConfig,
    TopoNaderArchitectureConfig,
    TopoNaderConfig,
    TopoNaderCoverageConfig,
    TopoNaderDataConfig,
    TopoNaderEliteLocalSearchConfig,
    TopoNaderMethodConfig,
    TopoNaderOptimizationConfig,
    TopoNaderTrainingConfig,
    TopoNaderUncertaintyConfig,
)
from topo_nader.pipeline import TopoNaderEngine
from topo_nader.pipeline.engine import _allocate_integer
from topo_nader.training.trainer import load_toponader_checkpoint
from topo_nader.training.uncertainty import TopoNaderDeterministicPredictor


class TinyTopoNaderEngine(TopoNaderEngine):
    @property
    def default_architecture(self):
        return TopoNaderArchitectureSpec(
            name="TopoNaderTiny",
            hidden_widths=(16,),
            latent_dim=8,
            predictor_type="mlp",
            predictor_width=16,
            predictor_depth=1,
            attention_heads=1,
        )


class TopoNaderPipelineTests(unittest.TestCase):
    def test_formal_elite_allocations_are_exact(self):
        self.assertEqual(_allocate_integer(300, (0.40, 0.20, 0.16, 0.12, 0.12)), [120, 60, 48, 36, 36])
        self.assertEqual(_allocate_integer(10, (0.40, 0.20, 0.20, 0.10, 0.10)), [4, 2, 2, 1, 1])

    def test_elite_local_prepare_keeps_exact_exploitation_and_exploration_counts(self):
        config = TopoNaderConfig(
            experiment_name="elite-local-test",
            data=TopoNaderDataConfig(validation_fraction=0.25, output_log10=False),
            training=TopoNaderTrainingConfig(
                epochs=1, quick_epochs=1, batch_size=8, patience=1, device="cpu"
            ),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(
                particles=8,
                iterations=1,
                candidate_pool_size=10,
                fe_batch_size=4,
                exploitation_count=2,
                exploration_count=2,
                coverage=TopoNaderCoverageConfig(upper_bound_enabled=False),
                elite_local_search=TopoNaderEliteLocalSearchConfig(
                    enabled=True,
                    elite_count=2,
                    particle_weights=(0.6, 0.4),
                    selected_weights=(0.5, 0.5),
                    candidates_per_elite=10,
                ),
            ),
            method=TopoNaderMethodConfig(
                architecture_adaptation=False,
                uncertainty_sampling=True,
                diversity_sampling=True,
            ),
            agents=TopoNaderAgentConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = TinyTopoNaderEngine(config, Path(__file__).parents[1])
            normalized = np.random.default_rng(31).random((16, 4, 4))
            designs = engine.geometry.denormalize(normalized)
            dataset_path = root / "initial.mat"
            savemat(dataset_path, {"inputs": designs, "outputs": np.linspace(1.0, 2.0, 16)})
            request_path = engine.prepare_iteration(root / "run", dataset_path, 1)
            request = loadmat(request_path)
            manifest = json.loads(
                (root / "run" / "iteration_0001" / "manifest.json").read_text()
            )
        sources = [str(value[0]) for value in request["source"].reshape(-1)]
        self.assertEqual(request["inputs"].shape[0], 4)
        self.assertEqual(sources.count("exploitation"), 2)
        self.assertEqual(sources.count("exploration"), 2)
        self.assertTrue(manifest["search"]["elite_local_search"]["enabled"])
        elite_stats = manifest["search"]["elite_local_search"]["candidate_pool_statistics"]
        self.assertEqual(len(elite_stats), 2)
        self.assertIn("coverage_repaired_fraction", elite_stats[0])
        self.assertIn("ranking_stability", elite_stats[0])
        exploration_stats = manifest["search"]["global_exploration"][
            "candidate_pool_statistics"
        ]
        self.assertIn("final_structure_duplicate_fraction", exploration_stats)

    def test_prepare_feedback_and_per_iteration_summary(self):
        config = TopoNaderConfig(
            data=TopoNaderDataConfig(validation_fraction=0.25, output_log10=False),
            training=TopoNaderTrainingConfig(
                epochs=1, quick_epochs=1, batch_size=8, patience=1, device="cpu"
            ),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(
                particles=6,
                iterations=1,
                candidate_pool_size=10,
                fe_batch_size=2,
                exploitation_count=1,
                exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound=1.0, upper_tolerance=0.0),
            ),
            method=TopoNaderMethodConfig(
                architecture_adaptation=False,
                uncertainty_sampling=False,
                diversity_sampling=True,
            ),
            agents=TopoNaderAgentConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = TinyTopoNaderEngine(config, Path(__file__).parents[1])
            rng = np.random.default_rng(5)
            normalized = rng.random((12, 4, 4))
            designs = engine.geometry.denormalize(normalized)
            outputs = np.linspace(1.0, 2.0, 12)
            dataset_path = root / "initial.mat"
            savemat(dataset_path, {"inputs": designs, "outputs": outputs.reshape(-1, 1)})
            request_path = engine.prepare_iteration(root / "run", dataset_path, 1)
            request = loadmat(request_path)
            checkpoint = root / "run" / "iteration_0001" / "incumbent.pth"
            model, transform, _ = load_toponader_checkpoint(checkpoint)
            predictor = TopoNaderDeterministicPredictor(model, transform)
            exact, _ = predictor.predict(
                engine.geometry.normalize(request["inputs"]).reshape(-1, engine.geometry.input_dim)
            )
            np.testing.assert_allclose(request["predicted_mean"].reshape(-1), exact, rtol=1e-6)
            result_path = root / "result.mat"
            savemat(result_path, {"outputs": (exact * np.array([0.98, 1.02])).reshape(-1, 1)})
            next_dataset = root / "next.mat"
            engine.apply_feedback(root / "run", dataset_path, 1, result_path, next_dataset)
            summary = json.loads((root / "run" / "summary.json").read_text())
            iteration_summary = json.loads(
                (root / "run" / "iteration_0001" / "iteration_summary.json").read_text()
            )
        self.assertEqual(summary["iterations_completed"], 1)
        self.assertEqual(summary["cumulative_fe_calls"], 14)
        self.assertEqual(iteration_summary["requested_fe_samples"], 2)

    def test_audit_reflector_receives_current_parent_training_metrics(self):
        class AuditClient:
            def __init__(self):
                self.calls = 0
                self.saw_parent_training = False

            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    payload = json.loads(messages[-1]["content"])
                    self.saw_parent_training = "current_parent_training" in payload["measured_metrics"]
                    content = json.dumps({"diagnosis": "test", "evidence": [], "risks": [], "recommended_focus": "test"})
                elif self.calls == 2:
                    content = json.dumps({"proposals": [{"title": "dropout", "rationale": "test", "changes": {}}]})
                else:
                    content = json.dumps(
                        TopoNaderArchitectureSpec(
                            name="TopoNaderTinyChild",
                            hidden_widths=(16,),
                            latent_dim=8,
                            predictor_type="mlp",
                            predictor_width=16,
                            predictor_depth=1,
                            attention_heads=1,
                            dropout=0.2,
                        ).to_dict()
                    )
                return {"content": content, "model": "stub", "prompt_tokens": 0, "completion_tokens": 0}

        config = TopoNaderConfig(
            data=TopoNaderDataConfig(validation_fraction=0.25, output_log10=False),
            training=TopoNaderTrainingConfig(
                epochs=1, quick_epochs=1, batch_size=8, patience=1, device="cpu"
            ),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(
                particles=4,
                iterations=1,
                candidate_pool_size=8,
                fe_batch_size=2,
                exploitation_count=1,
                exploration_count=1,
                coverage=TopoNaderCoverageConfig(upper_bound=1.0, upper_tolerance=0.0),
            ),
            architecture=TopoNaderArchitectureConfig(
                candidate_count=1,
                fully_train_count=2,
                candidate_retry_limit=2,
                zero_cost_enabled=False,
            ),
            method=TopoNaderMethodConfig(architecture_review_mode="every_round"),
            agents=TopoNaderAgentConfig(enabled=True),
        )
        client = AuditClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = TinyTopoNaderEngine(config, Path(__file__).parents[1], lambda _: client)
            designs = engine.geometry.denormalize(np.random.default_rng(8).random((12, 4, 4)))
            dataset_path = root / "initial.mat"
            savemat(dataset_path, {"inputs": designs, "outputs": np.linspace(1.0, 2.0, 12)})
            engine.prepare_iteration(root / "run", dataset_path, 1)
            trace = json.loads(
                (root / "run" / "iteration_0001" / "audit" / "trace.json").read_text()
            )
        self.assertTrue(client.saw_parent_training)
        self.assertEqual(trace["accepted_candidate_count"], 1)
        self.assertIn("validation", trace["parent_diagnostics"])

    def test_six_bar_prepare_uses_24_input_dimensions(self):
        config = TopoNaderConfig(
            experiment_name="six-bar-test",
            data=TopoNaderDataConfig(
                num_bars=6,
                validation_fraction=0.25,
                output_log10=False,
            ),
            training=TopoNaderTrainingConfig(
                epochs=1, quick_epochs=1, batch_size=8, patience=1, device="cpu"
            ),
            uncertainty=TopoNaderUncertaintyConfig(mc_samples=2),
            optimization=TopoNaderOptimizationConfig(
                particles=5,
                iterations=1,
                candidate_pool_size=8,
                fe_batch_size=2,
                exploitation_count=2,
                exploration_count=0,
                coverage=TopoNaderCoverageConfig(upper_bound_enabled=False),
            ),
            method=TopoNaderMethodConfig(
                architecture_adaptation=False,
                uncertainty_sampling=False,
                diversity_sampling=True,
            ),
            agents=TopoNaderAgentConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = TinyTopoNaderEngine(config, Path(__file__).parents[1])
            normalized = np.random.default_rng(9).random((12, 6, 4))
            designs = engine.geometry.denormalize(normalized)
            dataset_path = root / "initial.mat"
            savemat(dataset_path, {"inputs": designs, "outputs": np.linspace(1.0, 2.0, 12)})
            request_path = engine.prepare_iteration(root / "run", dataset_path, 1)
            request = loadmat(request_path)
            architecture = json.loads(
                (root / "run" / "state" / "current_architecture.json").read_text()
            )
        self.assertEqual(request["inputs"].shape[1:], (6, 4))
        self.assertEqual(int(request["num_bars"][0, 0]), 6)
        self.assertEqual(architecture["input_dim"], 24)

    def test_existing_experiment_rejects_changed_configuration(self):
        base = TopoNaderConfig(
            experiment_name="config-lock-test",
            random_seed=10,
            agents=TopoNaderAgentConfig(enabled=False),
        )
        changed = TopoNaderConfig(
            experiment_name="config-lock-test",
            random_seed=11,
            agents=TopoNaderAgentConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            TinyTopoNaderEngine(base, Path(__file__).parents[1]).initialize_run(run)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                TinyTopoNaderEngine(changed, Path(__file__).parents[1]).initialize_run(run)

    def test_existing_experiment_accepts_snapshot_from_before_structured_retry_setting(self):
        config = TopoNaderConfig(
            experiment_name="legacy-agent-config-test",
            agents=TopoNaderAgentConfig(enabled=False, structured_output_retries=2),
        )
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            engine = TinyTopoNaderEngine(config, Path(__file__).parents[1])
            expected = engine.initialize_run(run)
            snapshot_path = run / "config.snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            del snapshot["agents"]["structured_output_retries"]
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            restored = engine.initialize_run(run)
        self.assertEqual(restored.to_dict(), expected.to_dict())


if __name__ == "__main__":
    unittest.main()
