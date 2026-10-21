import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from topo_nader.config import TopoNaderDataConfig
from topo_nader.data import TopoNaderDataset


class TopoNaderDataTests(unittest.TestCase):
    def test_prefers_unsorted_chronological_legacy_fields(self):
        config = TopoNaderDataConfig(validation_fraction=0.25)
        chronological_x = np.arange(8 * 16, dtype=float).reshape(8, 4, 4)
        chronological_y = np.arange(1, 9, dtype=float)
        descending = np.argsort(-chronological_y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.mat"
            savemat(path, {
                "inputs": chronological_x[descending],
                "outputs": chronological_y[descending],
                "inputsX": chronological_x,
                "outputsX": chronological_y,
            })
            dataset = TopoNaderDataset.from_mat(path, config)
        split = dataset.chronological_split()
        self.assertTrue(dataset.chronological_order)
        np.testing.assert_array_equal(split.y_validation, [7.0, 8.0])

    def test_initial_sorted_file_uses_reproducible_random_split(self):
        config = TopoNaderDataConfig(validation_fraction=0.25, split_seed=11)
        dataset = TopoNaderDataset(np.zeros((8, 4, 4)), np.arange(1, 9), config)
        first = dataset.chronological_split().validation_indices
        second = dataset.chronological_split().validation_indices
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, np.array([6, 7])))

    def test_variable_bar_dataset_round_trip_shape(self):
        config = TopoNaderDataConfig(num_bars=6, split_seed=7)
        dataset = TopoNaderDataset(np.zeros((8, 6, 4)), np.arange(1, 9), config)
        appended = dataset.append(np.ones((2, 6, 4)), np.array([9.0, 10.0]))
        self.assertEqual(appended.inputs.shape, (10, 6, 4))
        self.assertEqual(appended.flat_inputs.shape, (10, 24))


if __name__ == "__main__":
    unittest.main()
