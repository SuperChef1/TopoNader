from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from ..config import TopoNaderDataConfig


@dataclass
class TopoNaderTargetTransform:
    mean: float
    std: float
    log10: bool

    @classmethod
    def fit(cls, y: np.ndarray, log10: bool) -> "TopoNaderTargetTransform":
        values = np.log10(y) if log10 else np.asarray(y, dtype=np.float64)
        std = float(np.std(values))
        return cls(float(np.mean(values)), std if std >= 1e-12 else 1.0, log10)

    def transform(self, y: np.ndarray) -> np.ndarray:
        values = np.log10(y) if self.log10 else np.asarray(y, dtype=np.float64)
        return (values - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=np.float64) * self.std + self.mean
        return np.power(10.0, values) if self.log10 else values


@dataclass
class TopoNaderSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_indices: np.ndarray


class TopoNaderDataset:
    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        config: TopoNaderDataConfig,
        chronological_order: bool = False,
    ):
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 3 or x.shape[1:] != config.design_shape:
            raise ValueError("fluid inputs must have shape (N,40,16); no implicit reshape")
        if not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
            raise ValueError("fluid training inputs must be finite binary obstacles")
        y = np.asarray(outputs, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("inputs and outputs have different sample counts")
        keep = np.isfinite(y) & (y > 0)
        if config.output_log10:
            keep &= y > 0
        self.inputs = x[keep]
        self.outputs = y[keep]
        self.config = config
        self.chronological_order = chronological_order
        if self.outputs.size < 3:
            raise ValueError("at least 3 finite samples are required")

    @classmethod
    def from_mat(cls, path: str | Path, config: TopoNaderDataConfig) -> "TopoNaderDataset":
        payload = loadmat(path)
        has_chronological = (
            config.chronological_input_key in payload and config.chronological_output_key in payload
        )
        input_key = config.chronological_input_key if has_chronological else config.input_key
        output_key = config.chronological_output_key if has_chronological else config.output_key
        return cls(payload[input_key], payload[output_key], config, chronological_order=has_chronological)

    @property
    def flat_inputs(self) -> np.ndarray:
        return self.inputs.reshape(self.inputs.shape[0], -1)

    def chronological_split(self) -> TopoNaderSplit:
        count = self.outputs.size
        validation_count = max(1, int(round(count * self.config.validation_fraction)))
        validation_count = min(validation_count, count - 2)
        split_at = count - validation_count
        if self.chronological_order:
            order = np.arange(count)
        else:
            seed = 3 if self.config.split_seed is None else int(self.config.split_seed)
            order = np.random.default_rng(seed).permutation(count)
        train_indices = order[:split_at]
        indices = order[split_at:]
        return TopoNaderSplit(
            self.flat_inputs[train_indices],
            self.outputs[train_indices],
            self.flat_inputs[indices],
            self.outputs[indices],
            indices,
        )

    def append(self, inputs: np.ndarray, outputs: np.ndarray) -> "TopoNaderDataset":
        joined_x = np.concatenate(
            [self.inputs, np.asarray(inputs).reshape(-1, *self.config.design_shape)], axis=0
        )
        joined_y = np.concatenate([self.outputs, np.asarray(outputs).reshape(-1)], axis=0)
        return TopoNaderDataset(joined_x, joined_y, self.config, chronological_order=True)

    def save_mat(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descending = np.argsort(-self.outputs)
        savemat(
            target,
            {
                self.config.input_key: self.inputs[descending],
                self.config.output_key: self.outputs[descending].reshape(-1, 1),
                self.config.chronological_input_key: self.inputs,
                self.config.chronological_output_key: self.outputs.reshape(-1, 1),
            },
        )
