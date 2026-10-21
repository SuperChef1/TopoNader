import unittest

import torch
from torch import nn

from topo_nader.training.trainer import _validation_predictions


class ValidationBatchingTests(unittest.TestCase):
    def test_bounded_batches_preserve_predictions_and_global_mse(self):
        class RecordingModel(nn.Linear):
            def forward(self, x):
                sizes.append(len(x))
                return super().forward(x)

        torch.manual_seed(7)
        sizes = []
        model = RecordingModel(4, 1).eval()
        x = torch.randn(19, 4)
        target = torch.randn(19, 1)
        with torch.no_grad():
            expected = model(x)
        sizes.clear()
        actual = _validation_predictions(model, x, torch.device('cpu'), 8)
        self.assertEqual(sizes, [8, 8, 3])
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(nn.functional.mse_loss(actual, target),
                                   nn.functional.mse_loss(expected, target))


if __name__ == '__main__':
    unittest.main()
