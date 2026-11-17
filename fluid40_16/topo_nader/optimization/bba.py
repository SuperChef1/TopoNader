from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TopoNaderBBAResult:
    """BBA population plus a bounded, score-aligned archive of accepted states."""

    positions: np.ndarray
    scores: np.ndarray
    trajectory_positions: np.ndarray
    trajectory_scores: np.ndarray


class TopoNaderBBA:
    """Maximizing Binary Bat Algorithm for the fluid topology bit grid.

    Adapted from ``fluid_fine/bba.py`` in deng-cy/deep_learning_topology_opt
    (MIT, commit 40600447fd3c36cf3757eb3b8f685bcb2e96a5f0), whose update
    structure follows a binary bat algorithm.  This NumPy implementation uses
    the standard non-negative frequency interval and the source file's noted
    correction ``1 / dimensions`` instead of its reproducibility-only
    ``1 / 160`` flip floor.
    """

    def __init__(
        self,
        particles: int,
        iterations: int,
        frequency_min: float = 0.0,
        frequency_max: float = 2.0,
        loudness: float = 0.25,
        pulse_rate: float = 0.4,
        loudness_decay: float = 0.999,
        pulse_decay: float = 10.0,
        seed: int = 3,
    ):
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.frequency_min = float(frequency_min)
        self.frequency_max = float(frequency_max)
        self.loudness = float(loudness)
        self.pulse_rate = float(pulse_rate)
        self.loudness_decay = float(loudness_decay)
        self.pulse_decay = float(pulse_decay)
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        dimensions: int | None = None,
        seeds: np.ndarray | None = None,
        initial_positions: np.ndarray | None = None,
        lower_bounds: np.ndarray | float = 0.0,
        upper_bounds: np.ndarray | float = 1.0,
        archive_size: int = 1000,
    ) -> TopoNaderBBAResult:
        if self.particles < 1 or self.iterations < 1:
            raise ValueError("particles and iterations must be positive")
        if archive_size < 1:
            raise ValueError("archive_size must be positive")
        if dimensions is None:
            supplied = initial_positions if initial_positions is not None else seeds
            if supplied is None:
                raise ValueError("dimensions is required when seeds or initial_positions are not provided")
            supplied_array = np.asarray(supplied)
            if supplied_array.ndim < 2:
                raise ValueError("seeds and initial_positions must be two-dimensional arrays")
            dimensions = int(supplied_array.reshape(supplied_array.shape[0], -1).shape[1])
        dimensions = int(dimensions)
        if dimensions < 1:
            raise ValueError("dimensions must be positive")

        low = np.broadcast_to(np.asarray(lower_bounds, dtype=np.float64), (dimensions,))
        high = np.broadcast_to(np.asarray(upper_bounds, dtype=np.float64), (dimensions,))
        if not np.all(low == 0.0) or not np.all(high == 1.0):
            raise ValueError("BBA operates on binary bounds [0,1]")
        if initial_positions is not None and seeds is not None:
            raise ValueError("provide either seeds or initial_positions, not both")

        if initial_positions is not None:
            supplied = np.asarray(initial_positions, dtype=np.float64).reshape(-1, dimensions)
            if supplied.shape[0] != self.particles:
                raise ValueError("initial_positions must contain exactly particles rows")
            positions = (supplied >= 0.5).astype(np.float64)
        else:
            # Match the upstream initialization: approximately ten percent 1 bits.
            positions = (self.rng.random((self.particles, dimensions)) > 0.9).astype(np.float64)
        if seeds is not None:
            seed_array = np.asarray(seeds, dtype=np.float64).reshape(-1, dimensions)
            count = min(seed_array.shape[0], self.particles)
            positions[:count] = (seed_array[:count] >= 0.5).astype(np.float64)

        velocities = np.zeros_like(positions)
        scores = self._evaluate(objective, positions)
        best_index = int(np.argmax(scores))
        best_position = positions[best_index].copy()
        best_score = float(scores[best_index])
        archive: dict[bytes, tuple[np.ndarray, float]] = {}

        def record() -> None:
            for position, score in zip(positions, scores):
                binary = position.astype(np.uint8, copy=False)
                key = np.packbits(binary).tobytes()
                if key not in archive or score > archive[key][1]:
                    archive[key] = (binary.astype(np.float64), float(score))
            if len(archive) > archive_size:
                keep = sorted(archive, key=lambda key: archive[key][1], reverse=True)[:archive_size]
                trimmed = {key: archive[key] for key in keep}
                archive.clear()
                archive.update(trimmed)

        record()
        loudness = self.loudness
        for iteration in range(self.iterations):
            pulse = self.pulse_rate * (1.0 - np.exp(-self.pulse_decay * iteration))
            loudness *= self.loudness_decay
            frequency = self.frequency_min + (
                self.frequency_max - self.frequency_min
            ) * self.rng.random((self.particles, 1))
            velocities += (positions - best_position) * frequency
            transfer = np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * velocities))

            proposed = positions.copy()
            flip_probability = np.minimum(1.0, transfer + 1.0 / dimensions)
            flip = self.rng.random((self.particles, dimensions)) < flip_probability
            proposed[flip] = 1.0 - proposed[flip]

            relocate = self.rng.random((self.particles, dimensions)) > pulse
            proposed[relocate] = np.broadcast_to(best_position, proposed.shape)[relocate]
            proposed_scores = self._evaluate(objective, proposed)

            accept = (proposed_scores >= scores) & (self.rng.random(self.particles) < loudness)
            positions[accept] = proposed[accept]
            scores[accept] = proposed_scores[accept]
            current_best = int(np.argmax(scores))
            if scores[current_best] > best_score:
                best_position = positions[current_best].copy()
                best_score = float(scores[current_best])
            record()

        archived = sorted(archive.values(), key=lambda pair: pair[1], reverse=True)
        return TopoNaderBBAResult(
            positions=positions.copy(),
            scores=scores.copy(),
            trajectory_positions=np.stack([pair[0] for pair in archived]),
            trajectory_scores=np.asarray([pair[1] for pair in archived], dtype=np.float64),
        )

    def _evaluate(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        positions: np.ndarray,
    ) -> np.ndarray:
        scores = np.asarray(objective(positions), dtype=np.float64).reshape(-1)
        if scores.shape != (self.particles,):
            raise ValueError("objective must return one score per BBA particle")
        if not np.isfinite(scores).all():
            raise ValueError("objective returned non-finite BBA scores")
        return scores
