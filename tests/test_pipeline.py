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
    TopoNaderDataConfig,
    TopoNaderMethodConfig,
    TopoNaderOptimizationConfig,
    TopoNaderTrainingConfig,
    TopoNaderUncertaintyConfig,
)
from topo_nader.pipeline import TopoNaderEngine
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
                ratio_target=1.0,
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
                engine.geometry.normalize(request["inputs"]).reshape(-1, 16)
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
                ratio_target=1.0,
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


if __name__ == "__main__":
    unittest.main()
