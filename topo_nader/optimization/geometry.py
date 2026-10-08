from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..config import TopoNaderOptimizationConfig


@dataclass
class TopoNaderGeometryResult:
    valid: bool
    design: np.ndarray
    coverage_ratio: float
    reason: str | None = None


class TopoNaderGeometry:
    """Variable-bar geometry contract migrated from the original topology project."""

    def __init__(
        self,
        config: TopoNaderOptimizationConfig,
        excluded_coords_mat: str | Path | None = None,
        num_bars: int = 4,
    ):
        self.config = config
        self.num_bars = int(num_bars)
        if self.num_bars < 1:
            raise ValueError("num_bars must be positive")
        self.parameters_per_bar = 4
        x = config.grid_x0 - np.arange(config.grid_nx) * config.grid_step
        y = config.grid_y0 - np.arange(config.grid_ny) * config.grid_step
        self.coordx, self.coordy = np.meshgrid(x, y, indexing="ij")
        self.design_mask = np.ones_like(self.coordx, dtype=bool)
        if excluded_coords_mat and Path(excluded_coords_mat).is_file():
            payload = loadmat(excluded_coords_mat)
            coords = np.asarray(payload.get("coords", []), dtype=float).squeeze()
            if coords.size:
                coords = coords.reshape(-1, 2) if coords.shape[-1] == 2 else coords.T
                for ix, iy in np.rint(coords).astype(int) - 1:
                    if 0 <= ix < config.grid_nx and 0 <= iy < config.grid_ny:
                        self.design_mask[ix, iy] = False

    @property
    def design_shape(self) -> tuple[int, int]:
        return self.num_bars, self.parameters_per_bar

    @property
    def input_dim(self) -> int:
        return self.num_bars * self.parameters_per_bar

    @property
    def lower(self) -> np.ndarray:
        result = np.zeros(self.design_shape, dtype=np.float64)
        result[:, 0] = np.ceil(self.coordx.min())
        result[:, 1] = np.ceil(self.coordy.min())
        result[:, 2] = 0.0
        result[:, 3] = self.config.rod_width_min
        return result

    @property
    def upper(self) -> np.ndarray:
        result = np.zeros(self.design_shape, dtype=np.float64)
        result[:, 0] = np.floor(self.coordx.max())
        result[:, 1] = np.floor(self.coordy.max())
        result[:, 2] = 179.0
        result[:, 3] = self.config.rod_width_max
        return result

    def normalize(self, designs: np.ndarray) -> np.ndarray:
        designs = np.asarray(designs, dtype=np.float64).reshape(-1, *self.design_shape)
        return np.clip((designs - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def denormalize(self, designs: np.ndarray) -> np.ndarray:
        designs = np.asarray(designs, dtype=np.float64).reshape(-1, *self.design_shape)
        return designs * (self.upper - self.lower) + self.lower

    def discretize(self, design: np.ndarray) -> np.ndarray:
        result = np.nan_to_num(np.asarray(design, dtype=np.float64).reshape(self.design_shape)).copy()
        result[:, 0] = np.clip(np.rint(result[:, 0]), self.lower[0, 0], self.upper[0, 0])
        result[:, 1] = np.clip(np.rint(result[:, 1]), self.lower[0, 1], self.upper[0, 1])
        result[:, 2] = np.mod(np.rint(result[:, 2]), 180)
        result[:, 3] = np.clip(np.rint(np.abs(result[:, 3])), self.config.rod_width_min, self.config.rod_width_max)
        return result

    def coverage(self, design: np.ndarray) -> float:
        filled = np.zeros_like(self.coordx, dtype=bool)
        for xc, yc, theta, width in np.asarray(design).reshape(self.design_shape):
            radians = np.deg2rad(theta)
            normal_distance = (self.coordx - xc) * np.cos(radians) + (self.coordy - yc) * np.sin(radians)
            filled |= np.abs(normal_distance) <= width / 2.0
        active = self.design_mask
        return float(filled[active].mean()) if np.any(active) else 0.0

    def finalize(self, design: np.ndarray) -> TopoNaderGeometryResult:
        candidate = self.discretize(design)
        ratio = self.coverage(candidate)
        coverage = self.config.coverage
        if coverage.upper_bound_enabled and ratio > coverage.upper_bound:
            original_widths = candidate[:, 3].copy()
            low, high = 0.0, 1.0
            best = None
            for _ in range(40):
                scale = (low + high) / 2.0
                trial = candidate.copy()
                trial[:, 3] = np.clip(
                    np.rint(original_widths * scale),
                    self.config.rod_width_min,
                    self.config.rod_width_max,
                )
                trial_ratio = self.coverage(trial)
                if trial_ratio <= coverage.upper_bound:
                    best = trial
                    low = scale
                else:
                    high = scale
            if best is not None:
                candidate = best
                ratio = self.coverage(candidate)

        if coverage.lower_bound_enabled and ratio < float(coverage.lower_bound):
            original_widths = candidate[:, 3].copy()
            maximum_scale = float(np.max(self.config.rod_width_max / np.maximum(original_widths, 1e-12)))
            low, high = 1.0, max(1.0, maximum_scale)
            best = None
            for _ in range(40):
                scale = (low + high) / 2.0
                trial = candidate.copy()
                trial[:, 3] = np.clip(
                    np.rint(original_widths * scale),
                    self.config.rod_width_min,
                    self.config.rod_width_max,
                )
                trial_ratio = self.coverage(trial)
                if trial_ratio >= float(coverage.lower_bound):
                    best = trial
                    high = scale
                else:
                    low = scale
            if best is not None:
                candidate = best
                ratio = self.coverage(candidate)

        upper_valid = (
            not coverage.upper_bound_enabled
            or ratio <= coverage.upper_bound + coverage.upper_tolerance
        )
        lower_valid = (
            not coverage.lower_bound_enabled
            or ratio >= float(coverage.lower_bound)
        )
        reason = None
        if not upper_valid:
            reason = "coverage_upper_limit"
        elif not lower_valid:
            reason = "coverage_lower_limit"
        return TopoNaderGeometryResult(
            valid=bool(upper_valid and lower_valid),
            design=candidate,
            coverage_ratio=ratio,
            reason=reason,
        )


def relatively_duplicate(left: np.ndarray, right: np.ndarray, tolerance: float = 0.005) -> bool:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return bool(np.all(np.abs(a - b) / scale < tolerance))
