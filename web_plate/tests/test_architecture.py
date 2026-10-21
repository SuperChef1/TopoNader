import unittest

import torch

from topo_nader.architecture import (
    TopoNaderArchitectureSpec,
    TopoNaderArchitectureValidator,
    TopoNaderModelFactory,
)
from topo_nader.config import TopoNaderArchitectureConfig


class TopoNaderArchitectureTests(unittest.TestCase):
    def test_baseline_is_valid_and_has_expected_shape(self):
        spec = TopoNaderArchitectureSpec()
        report = TopoNaderArchitectureValidator(TopoNaderArchitectureConfig()).validate(spec)
        self.assertTrue(report.valid, report.errors)
        output = TopoNaderModelFactory.build(spec)(torch.zeros(3, 16))
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(spec.architecture_id.startswith("toponader-"))

    def test_invalid_attention_width_is_rejected(self):
        spec = TopoNaderArchitectureSpec(predictor_width=127, attention_heads=4)
        report = TopoNaderArchitectureValidator(TopoNaderArchitectureConfig()).validate(spec)
        self.assertFalse(report.valid)

    def test_equivalent_mlp_ignores_transformer_only_identity_fields(self):
        left = TopoNaderArchitectureSpec(
            predictor_type="mlp", predictor_depth=1, attention_heads=1
        )
        right = TopoNaderArchitectureSpec(
            predictor_type="mlp", predictor_depth=7, attention_heads=8
        )
        self.assertEqual(left.architecture_id, right.architecture_id)

    def test_transformer_supports_variable_bar_count(self):
        spec = TopoNaderArchitectureSpec(input_dim=24, latent_dim=10)
        report = TopoNaderArchitectureValidator(
            TopoNaderArchitectureConfig(), expected_input_dim=24
        ).validate(spec)
        self.assertTrue(report.valid, report.errors)
        output = TopoNaderModelFactory.build(spec)(torch.zeros(3, 24))
        self.assertEqual(tuple(output.shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
