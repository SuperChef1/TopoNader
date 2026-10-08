from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TopoNaderPSOResult:
    positions: np.ndarray
    scores: np.ndarray
    trajectory_positions: np.ndarray
    trajectory_scores: np.ndarray


class TopoNaderPSO:
    """Maximizing PSO adapted from the topology project's EPSO implementation."""

    def __init__(
        self,
        particles: int,
        iterations: int,
        mutation_rate: float = 0.1,
        stall_limit: int = 15,
        seed: int = 3,
    ):
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.mutation_rate = float(mutation_rate)
        self.stall_limit = int(stall_limit)
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        dimensions: int | None = None,
        seeds: np.ndarray | None = None,
    ) -> TopoNaderPSOResult:
        if dimensions is None:
            if seeds is None:
                raise ValueError("dimensions is required when seeds are not provided")
            seed_values = np.asarray(seeds)
            if seed_values.ndim < 2:
                raise ValueError("seeds must be a two-dimensional array")
            dimensions = int(seed_values.reshape(seed_values.shape[0], -1).shape[1])
        dimensions = int(dimensions)
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        positions = self.rng.random((self.particles, dimensions))
        if seeds is not None:
            seed_array = np.asarray(seeds, dtype=np.float64).reshape(-1, dimensions)
            count = min(seed_array.shape[0], self.particles)
            positions[:count] = np.clip(seed_array[:count], 0.0, 1.0)
        velocities = np.zeros_like(positions)
        scores = np.asarray(objective(positions), dtype=np.float64).reshape(-1)
        personal_positions = positions.copy()
        personal_scores = scores.copy()
        best_index = int(np.argmax(scores))
        global_position = positions[best_index].copy()
        global_score = float(scores[best_index])
        trajectory_x = [positions.copy()]
        trajectory_y = [scores.copy()]
        stall = 0
        for iteration in range(1, self.iterations + 1):
            progress = iteration / max(self.iterations, 1)
            inertia = 0.9 - 0.5 * progress**2
            cognitive = 2.5 - 2.0 * progress
            social = 0.5 + 2.0 * progress
            velocities = (
                inertia * velocities
                + cognitive * self.rng.random(positions.shape) * (personal_positions - positions)
                + social * self.rng.random(positions.shape) * (global_position - positions)
            )
            positions = positions + velocities
            mutate_rows = np.flatnonzero(self.rng.random(self.particles) < self.mutation_rate)
            if mutate_rows.size:
                mutate_columns = self.rng.integers(0, dimensions, size=mutate_rows.size)
                positions[mutate_rows, mutate_columns] = self.rng.random(mutate_rows.size)
            positions = np.clip(positions, 0.0, 1.0)
            scores = np.asarray(objective(positions), dtype=np.float64).reshape(-1)
            improved = scores > personal_scores
            personal_positions[improved] = positions[improved]
            personal_scores[improved] = scores[improved]
            current_best = int(np.argmax(personal_scores))
            if personal_scores[current_best] > global_score:
                global_score = float(personal_scores[current_best])
                global_position = personal_positions[current_best].copy()
                stall = 0
            else:
                stall += 1
            if stall >= self.stall_limit and iteration != self.iterations:
                reset_count = max(1, int(self.particles * 0.2))
                worst = np.argsort(personal_scores)[:reset_count]
                positions[worst] = self.rng.random((reset_count, dimensions))
                stall = 0
            trajectory_x.append(positions.copy())
            trajectory_y.append(scores.copy())
        return TopoNaderPSOResult(
            personal_positions,
            personal_scores,
            np.vstack(trajectory_x),
            np.concatenate(trajectory_y),
        )
