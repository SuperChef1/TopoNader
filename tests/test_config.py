import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from topo_nader.cli import main
from topo_nader.config import load_config


class TopoNaderConfigTests(unittest.TestCase):
    def test_baseline_config_inherits_default_and_overrides_switches(self):
        path = Path(__file__).parents[1] / "configs" / "baseline_fixed_mean.json"
        config = load_config(path)
        self.assertFalse(config.method.architecture_adaptation)
        self.assertFalse(config.method.uncertainty_sampling)
        self.assertEqual(config.optimization.fe_batch_size, 10)

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


if __name__ == "__main__":
    unittest.main()
