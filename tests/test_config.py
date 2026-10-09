import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from topo_nader.cli import main
from topo_nader.config import TopoNaderConfig, load_config


class TopoNaderConfigTests(unittest.TestCase):
    def test_baseline_config_inherits_default_and_overrides_switches(self):
        path = Path(__file__).parents[1] / "configs" / "baseline_fixed_mean.json"
        config = load_config(path)
        self.assertFalse(config.method.architecture_adaptation)
        self.assertFalse(config.method.uncertainty_sampling)
        self.assertEqual(config.optimization.fe_batch_size, 20)
        self.assertEqual(config.optimization.exploitation_count, 10)
        self.assertEqual(config.optimization.exploration_count, 10)

    def test_resolved_config_cli_includes_default_paths(self):
        path = Path(__file__).parents[1] / "configs" / "smoke.json"
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--config", str(path), "resolved-config"])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertIn("paths", payload)
        self.assertEqual(
            payload["paths"]["source_topology_dir"],
            "matlab",
        )
        self.assertEqual(payload["paths"]["comsol_model"], "matlab/fuban_Beam.mph")
        self.assertEqual(payload["random_seed"], 3)
        self.assertEqual(payload["agents"]["structured_output_retries"], 2)
        self.assertIn("matlab_inputs", payload["resolved_seeds"])
        self.assertTrue(payload["optimization"]["coverage"]["upper_bound_enabled"])

    def test_default_runtime_files_are_inside_project(self):
        project_root = Path(__file__).parents[1].resolve()
        config = load_config(project_root / "configs" / "default.json")
        source_dir = (project_root / config.paths.source_topology_dir).resolve()
        model_path = (project_root / config.paths.comsol_model).resolve()

        self.assertTrue(source_dir.is_relative_to(project_root))
        self.assertTrue(model_path.is_relative_to(project_root))
        self.assertTrue((source_dir / "func_outputs_nowrite.m").is_file())
        self.assertTrue((source_dir / "Coord_True_coords.mat").is_file())
        self.assertTrue(model_path.is_file())

    def test_random_seed_derives_stable_independent_module_seeds(self):
        project_root = Path(__file__).parents[1].resolve()
        config = load_config(project_root / "configs" / "default.json")
        first = config.seed_manifest()
        second = config.seed_manifest()
        self.assertEqual(first, second)
        self.assertNotEqual(first["training"], first["pso_mean"])
        self.assertEqual(config.data.input_dim, 16)
        self.assertEqual(config.default_run_dir(project_root).name, config.experiment_name)

    def test_legacy_ratio_fields_map_to_upper_coverage_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({
                "experiment_name": "legacy-coverage",
                "optimization": {"ratio_target": 0.42, "ratio_tolerance": 0.03},
            }), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.optimization.coverage.upper_bound, 0.42)
        self.assertEqual(config.optimization.coverage.upper_tolerance, 0.03)

    def test_experiment_name_accepts_unicode_but_rejects_paths(self):
        TopoNaderConfig(experiment_name="六杆-覆盖率030-seed3").validate()
        with self.assertRaisesRegex(ValueError, "path-safe"):
            TopoNaderConfig(experiment_name="group/experiment").validate()


if __name__ == "__main__":
    unittest.main()
