from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .legacy_connectivity import isvalid


@dataclass
class TopoNaderGeometryResult:
    valid: bool
    design: np.ndarray
    coverage_ratio: float
    reason: str | None = None
    original_coverage_ratio: float | None = None
    coverage_repaired: bool = False


@lru_cache(maxsize=32768)
def _legacy_valid(bits: bytes) -> bool:
    design = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), count=640).reshape(40, 16)
    # Original MATLAB: isvalid(reshape(input,40,16)'). Do not widen the ports.
    return bool(isvalid(design.T))


class TopoNaderGeometry:
    """40x16 binary obstacles with the original case's exact connectivity rule."""

    design_shape = (40, 16)
    input_dim = 640

    def __init__(self, config):
        self.config = config

    @property
    def lower(self):
        return np.zeros(self.design_shape)

    @property
    def upper(self):
        return np.ones(self.design_shape)

    def normalize(self, designs):
        x = np.asarray(designs, dtype=np.float64).reshape(-1, *self.design_shape)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite fluid design')
        return np.clip(x, 0, 1)

    denormalize = normalize

    def discretize(self, design):
        x = self.normalize(design)[0]
        return (x >= 0.5).astype(np.float64)

    def valid(self, design):
        x = self.discretize(design)
        return _legacy_valid(np.packbits(x.astype(np.uint8).reshape(-1)).tobytes())

    def coverage(self, design):
        return float(self.discretize(design).mean())

    def finalize(self, design):
        x = self.discretize(design)
        valid = self.valid(x)
        ratio = float(x.mean())
        return TopoNaderGeometryResult(valid, x, ratio,
            None if valid else 'legacy_connectivity', ratio, False)


def relatively_duplicate(left, right, tolerance=0.005):
    # Exact equality on the final binary structures; tolerance intentionally unused.
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))
