from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .geometry import TopoNaderGeometry, relatively_duplicate


@dataclass
class TopoNaderBatchSelection:
    designs: np.ndarray
    predicted_mean: np.ndarray
    predicted_uncertainty: np.ndarray
    acquisition: np.ndarray
    sources: tuple[str, ...]
    coverage_ratios: np.ndarray
    rejection_counts: dict[str, int] | None = None


def _farthest_choice(pool: np.ndarray, candidates: list[int], selected: list[int], priority: np.ndarray) -> int:
    if not selected:
        return max(candidates, key=lambda index: float(priority[index]))
    return max(
        candidates,
        key=lambda index: (
            float(np.min(np.linalg.norm(pool[index] - pool[selected], axis=1))),
            float(priority[index]),
        ),
    )


def select_active_batch(
    normalized_pool: np.ndarray,
    mean: np.ndarray,
    uncertainty: np.ndarray,
    geometry: TopoNaderGeometry,
    exploitation_count: int,
    exploration_count: int,
    kappa: float,
    use_diversity: bool = True,
    excluded_designs: np.ndarray | None = None,
) -> TopoNaderBatchSelection:
    pool = np.asarray(normalized_pool, dtype=np.float64).reshape(-1, geometry.input_dim)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    sigma = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if not (pool.shape[0] == mu.size == sigma.size):
        raise ValueError("pool and prediction arrays must have the same length")
    ucb = mu + float(kappa) * sigma
    accepted: list[int] = []
    sources: list[str] = []
    finalized: list[np.ndarray] = []
    coverage: list[float] = []
    rejection_counts = {"coverage": 0, "historical_duplicate": 0, "batch_duplicate": 0}
    historical = (
        np.asarray(excluded_designs, dtype=np.float64).reshape(-1, *geometry.design_shape)
        if excluded_designs is not None
        else np.empty((0, *geometry.design_shape), dtype=np.float64)
    )

    def fill(count: int, priority: np.ndarray, source: str) -> None:
        remaining = [index for index in np.argsort(-priority) if int(index) not in accepted]
        while len([item for item in sources if item == source]) < count and remaining:
            chosen = (
                _farthest_choice(pool, [int(i) for i in remaining], accepted, priority)
                if use_diversity
                else int(remaining[0])
            )
            remaining.remove(chosen)
            result = geometry.finalize(
                geometry.denormalize(pool[chosen].reshape(1, *geometry.design_shape))[0]
            )
            if not result.valid:
                rejection_counts["coverage"] += 1
                continue
            if any(relatively_duplicate(result.design, previous) for previous in historical):
                rejection_counts["historical_duplicate"] += 1
                continue
            if any(relatively_duplicate(result.design, previous) for previous in finalized):
                rejection_counts["batch_duplicate"] += 1
                continue
            accepted.append(chosen)
            sources.append(source)
            finalized.append(result.design)
            coverage.append(result.coverage_ratio)

    fill(int(exploitation_count), mu, "exploitation")
    fill(int(exploration_count), ucb, "exploration")
    if not accepted:
        raise RuntimeError("no valid candidates remained after geometry validation")
    idx = np.asarray(accepted, dtype=int)
    return TopoNaderBatchSelection(
        designs=np.stack(finalized),
        predicted_mean=mu[idx],
        predicted_uncertainty=sigma[idx],
        acquisition=ucb[idx],
        sources=tuple(sources),
        coverage_ratios=np.asarray(coverage),
        rejection_counts=rejection_counts,
    )


def refresh_batch_predictions(
    batch: TopoNaderBatchSelection,
    geometry: TopoNaderGeometry,
    predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    kappa: float,
) -> TopoNaderBatchSelection:
    """Re-predict the exact discretized/repaired designs sent to FE."""
    normalized = geometry.normalize(batch.designs).reshape(-1, geometry.input_dim)
    mean, uncertainty = predictor(normalized)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if mean.size != batch.designs.shape[0] or uncertainty.size != batch.designs.shape[0]:
        raise ValueError("refreshed prediction count does not match finalized FE batch")
    return replace(
        batch,
        predicted_mean=mean,
        predicted_uncertainty=uncertainty,
        acquisition=mean + float(kappa) * uncertainty,
    )
