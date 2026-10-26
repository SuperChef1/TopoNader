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


@dataclass
class TopoNaderFinalizedPool:
    designs: np.ndarray
    normalized: np.ndarray
    predicted_mean: np.ndarray
    predicted_uncertainty: np.ndarray
    coverage_ratios: np.ndarray
    rejection_counts: dict[str, int]
    source_indices: np.ndarray
    finalization_stats: dict[str, object]


def _empty_rejection_counts() -> dict[str, int]:
    return {"connectivity": 0, "historical_duplicate": 0, "batch_duplicate": 0}


def _merge_rejection_counts(*counts: dict[str, int] | None) -> dict[str, int]:
    result = _empty_rejection_counts()
    for payload in counts:
        if payload:
            for key, value in payload.items():
                result[key] = result.get(key, 0) + int(value)
    return result


def _farthest_choice(pool: np.ndarray, candidates: list[int], selected: list[int], priority: np.ndarray) -> int:
    if not selected:
        return max(candidates, key=lambda index: float(priority[index]))
    return max(
        candidates,
        key=lambda index: (
            float(np.min(np.mean(pool[index] != pool[selected], axis=1))),
            float(priority[index]),
        ),
    )


def finalize_candidate_pool(normalized_pool, geometry, predictor, excluded_designs=None,
                            ranking_mode=None, kappa=0.0, raw_priority=None):
    raw = np.asarray(normalized_pool, float).reshape(-1, 640)
    binary = (raw >= 0.5).astype(float).reshape(-1, 40, 16)
    key = lambda x: np.packbits(np.asarray(x, dtype=np.uint8).reshape(-1)).tobytes()
    historical = {key(x) for x in np.asarray(excluded_designs).reshape(-1,40,16)} if excluded_designs is not None else set()
    seen, accepted, indices = set(), [], []
    rejection = _empty_rejection_counts()
    for i,x in enumerate(binary):
        k = key(x)
        if not geometry.valid(x): rejection['connectivity'] += 1; continue
        if k in historical: rejection['historical_duplicate'] += 1; continue
        if k in seen: rejection['batch_duplicate'] += 1; continue
        seen.add(k); accepted.append(x); indices.append(i)
    designs = np.asarray(accepted, dtype=float).reshape(-1,40,16)
    normalized = designs.reshape(-1,640)
    if len(designs):
        mean, sigma = predictor(normalized)
        mean, sigma = np.asarray(mean).reshape(-1), np.asarray(sigma).reshape(-1)
        if len(mean) != len(designs) or len(sigma) != len(designs) or not np.isfinite(mean).all() or not np.isfinite(sigma).all():
            raise ValueError('invalid predictions for binary candidate pool')
    else: mean, sigma = np.empty(0), np.empty(0)
    count = len(raw)
    stats = {'input_candidate_count': count,'accepted_unique_count': len(designs),
        'connectivity_rejected_count': rejection['connectivity'],
        'connectivity_rejected_fraction': rejection['connectivity']/max(count,1),
        'historical_duplicate_count': rejection['historical_duplicate'],
        'final_structure_duplicate_count': rejection['batch_duplicate'],
        'unique_yield_fraction': len(designs)/max(count,1),
        'connectivity_repaired_count': 0,
        'binarization_mean_absolute_change': float(np.mean(np.abs(raw-binary.reshape(-1,640)))) if count else 0.0,
        'solid_fraction_mean': float(designs.mean()) if len(designs) else None}
    return TopoNaderFinalizedPool(designs, normalized, mean, sigma,
        designs.mean(axis=(1,2)) if len(designs) else np.empty(0), rejection,
        np.asarray(indices,dtype=int), stats)


def select_from_finalized_pool(
    pool: TopoNaderFinalizedPool,
    count: int,
    kappa: float,
    source: str,
    use_uncertainty: bool,
    use_diversity: bool = True,
    excluded_designs: np.ndarray | None = None,
) -> TopoNaderBatchSelection:
    """Select already-finalized candidates without changing their geometry again."""
    count = int(count)
    if count < 0:
        raise ValueError("selection count must be non-negative")
    existing = (
        np.asarray(excluded_designs, dtype=np.float64).reshape(-1, *pool.designs.shape[1:])
        if excluded_designs is not None and pool.designs.size
        else np.empty((0, *pool.designs.shape[1:]), dtype=np.float64)
        if pool.designs.ndim >= 3
        else np.empty((0, 0, 0), dtype=np.float64)
    )
    priority = (
        pool.predicted_mean + float(kappa) * pool.predicted_uncertainty
        if use_uncertainty
        else pool.predicted_mean
    )
    candidates = [int(index) for index in np.argsort(-priority)]
    selected: list[int] = []
    while len(selected) < count and candidates:
        chosen = (
            _farthest_choice(pool.normalized, candidates, selected, priority)
            if use_diversity
            else candidates[0]
        )
        candidates.remove(chosen)
        if any(relatively_duplicate(pool.designs[chosen], previous) for previous in existing):
            continue
        selected.append(chosen)
    if not selected:
        shape = pool.designs.shape[1:] if pool.designs.ndim >= 3 else (0, 0)
        return TopoNaderBatchSelection(
            designs=np.empty((0, *shape), dtype=np.float64),
            predicted_mean=np.empty(0, dtype=np.float64),
            predicted_uncertainty=np.empty(0, dtype=np.float64),
            acquisition=np.empty(0, dtype=np.float64),
            sources=(),
            coverage_ratios=np.empty(0, dtype=np.float64),
            rejection_counts=pool.rejection_counts,
        )
    idx = np.asarray(selected, dtype=int)
    return TopoNaderBatchSelection(
        designs=pool.designs[idx],
        predicted_mean=pool.predicted_mean[idx],
        predicted_uncertainty=pool.predicted_uncertainty[idx],
        acquisition=(pool.predicted_mean + float(kappa) * pool.predicted_uncertainty)[idx],
        sources=tuple(source for _ in selected),
        coverage_ratios=pool.coverage_ratios[idx],
        rejection_counts=pool.rejection_counts,
    )


def combine_batches(*batches: TopoNaderBatchSelection) -> TopoNaderBatchSelection:
    nonempty = [batch for batch in batches if batch.designs.shape[0] > 0]
    if not nonempty:
        raise RuntimeError("cannot combine empty FE batches")
    return TopoNaderBatchSelection(
        designs=np.concatenate([batch.designs for batch in nonempty], axis=0),
        predicted_mean=np.concatenate([batch.predicted_mean for batch in nonempty]),
        predicted_uncertainty=np.concatenate([batch.predicted_uncertainty for batch in nonempty]),
        acquisition=np.concatenate([batch.acquisition for batch in nonempty]),
        sources=tuple(source for batch in nonempty for source in batch.sources),
        coverage_ratios=np.concatenate([batch.coverage_ratios for batch in nonempty]),
        rejection_counts=_merge_rejection_counts(
            *(batch.rejection_counts for batch in nonempty)
        ),
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
