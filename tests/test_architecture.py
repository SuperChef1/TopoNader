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


if __name__ == "__main__":
    unittest.main()

