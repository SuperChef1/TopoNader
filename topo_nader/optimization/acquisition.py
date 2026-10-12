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
    return {"coverage": 0, "historical_duplicate": 0, "batch_duplicate": 0}


def _merge_rejection_counts(*counts: dict[str, int] | None) -> dict[str, int]:
    result = _empty_rejection_counts()
    for payload in counts:
        if payload:
            for key, value in payload.items():
                result[key] = result.get(key, 0) + int(value)
    return result


def _ranking_stability(raw: np.ndarray, final: np.ndarray, top_k: int = 10) -> dict[str, object]:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    final = np.asarray(final, dtype=np.float64).reshape(-1)
    if raw.size != final.size:
        raise ValueError("raw and finalized priority arrays must have the same length")
    count = int(raw.size)
    if count == 0:
        return {
            "compared_count": 0,
            "spearman_rho": None,
            "pairwise_order_agreement": None,
            "top_k": 0,
            "top_k_overlap_fraction": None,
            "mean_absolute_priority_change": None,
            "max_absolute_priority_change": None,
        }
    absolute_change = np.abs(final - raw)
    selected_k = min(int(top_k), count)
    raw_top = set(np.argsort(-raw, kind="stable")[:selected_k].tolist())
    final_top = set(np.argsort(-final, kind="stable")[:selected_k].tolist())
    if count >= 2 and float(np.std(raw)) > 0.0 and float(np.std(final)) > 0.0:
        def average_ranks(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="stable")
            ranks = np.empty(count, dtype=np.float64)
            start = 0
            while start < count:
                end = start + 1
                while end < count and values[order[end]] == values[order[start]]:
                    end += 1
                ranks[order[start:end]] = (start + end - 1) / 2.0
                start = end
            return ranks

        raw_rank = average_ranks(raw)
        final_rank = average_ranks(final)
        spearman = float(np.corrcoef(raw_rank, final_rank)[0, 1])
    else:
        spearman = None
    agreements = 0
    comparisons = 0
    for left in range(count - 1):
        raw_difference = raw[left] - raw[left + 1 :]
        final_difference = final[left] - final[left + 1 :]
        comparable = (raw_difference != 0.0) | (final_difference != 0.0)
        if np.any(comparable):
            agreements += int(
                np.count_nonzero(
                    np.sign(raw_difference[comparable]) == np.sign(final_difference[comparable])
                )
            )
            comparisons += int(np.count_nonzero(comparable))
    return {
        "compared_count": count,
        "spearman_rho": spearman,
        "pairwise_order_agreement": (
            float(agreements / comparisons) if comparisons else None
        ),
        "top_k": selected_k,
        "top_k_overlap_fraction": float(len(raw_top & final_top) / selected_k),
        "mean_absolute_priority_change": float(np.mean(absolute_change)),
        "max_absolute_priority_change": float(np.max(absolute_change)),
    }


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
    rejection_counts = _empty_rejection_counts()
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


def finalize_candidate_pool(
    normalized_pool: np.ndarray,
    geometry: TopoNaderGeometry,
    predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    excluded_designs: np.ndarray | None = None,
    ranking_mode: str | None = None,
    kappa: float = 0.0,
    raw_priority: np.ndarray | None = None,
) -> TopoNaderFinalizedPool:
    """Finalize, deduplicate, then predict the exact structures eligible for FE."""
    pool = np.asarray(normalized_pool, dtype=np.float64).reshape(-1, geometry.input_dim)
    supplied_raw_priority = (
        np.asarray(raw_priority, dtype=np.float64).reshape(-1)
        if raw_priority is not None
        else None
    )
    if supplied_raw_priority is not None and supplied_raw_priority.size != pool.shape[0]:
        raise ValueError("raw_priority must contain one score per input candidate")
    historical = (
        np.asarray(excluded_designs, dtype=np.float64).reshape(-1, *geometry.design_shape)
        if excluded_designs is not None
        else np.empty((0, *geometry.design_shape), dtype=np.float64)
    )
    designs: list[np.ndarray] = []
    coverage: list[float] = []
    source_indices: list[int] = []
    projection_distances: list[float] = []
    coverage_repair_distances: list[float] = []
    repaired_width_changes: list[float] = []
    repaired_width_relative_changes: list[float] = []
    original_coverage_ratios: list[float] = []
    finalized_coverage_ratios: list[float] = []
    valid_count = 0
    coverage_repaired_count = 0
    rejection_counts = _empty_rejection_counts()
    for source_index, candidate in enumerate(pool):
        physical = geometry.denormalize(candidate.reshape(1, *geometry.design_shape))[0]
        discretized = geometry.discretize(physical)
        result = geometry.finalize(physical)
        original_coverage_ratios.append(float(result.original_coverage_ratio))
        finalized_coverage_ratios.append(float(result.coverage_ratio))
        discretized_normalized = geometry.normalize(discretized.reshape(1, *geometry.design_shape)).reshape(-1)
        finalized_normalized = geometry.normalize(result.design.reshape(1, *geometry.design_shape)).reshape(-1)
        projection_distances.append(float(np.max(np.abs(finalized_normalized - candidate))))
        coverage_repair_distance = float(
            np.max(np.abs(finalized_normalized - discretized_normalized))
        )
        if coverage_repair_distance > 0.0:
            coverage_repaired_count += 1
            coverage_repair_distances.append(coverage_repair_distance)
            width_change = np.abs(result.design[:, 3] - discretized[:, 3])
            repaired_width_changes.extend(float(value) for value in width_change)
            repaired_width_relative_changes.extend(
                float(value)
                for value in width_change / np.maximum(np.abs(discretized[:, 3]), 1.0)
            )
        if not result.valid:
            rejection_counts["coverage"] += 1
            continue
        valid_count += 1
        if any(relatively_duplicate(result.design, previous) for previous in historical):
            rejection_counts["historical_duplicate"] += 1
            continue
        if any(relatively_duplicate(result.design, previous) for previous in designs):
            rejection_counts["batch_duplicate"] += 1
            continue
        designs.append(result.design)
        coverage.append(result.coverage_ratio)
        source_indices.append(source_index)
    input_count = int(pool.shape[0])
    accepted_count = len(designs)
    finalization_stats: dict[str, object] = {
        "input_candidate_count": input_count,
        "constraint_valid_count": int(valid_count),
        "accepted_unique_count": int(accepted_count),
        "coverage_repaired_count": int(coverage_repaired_count),
        "coverage_repaired_fraction": (
            float(coverage_repaired_count / input_count) if input_count else 0.0
        ),
        "historical_duplicate_count": int(rejection_counts["historical_duplicate"]),
        "final_structure_duplicate_count": int(rejection_counts["batch_duplicate"]),
        "final_structure_duplicate_fraction": (
            float(rejection_counts["batch_duplicate"] / input_count) if input_count else 0.0
        ),
        "unique_yield_fraction": float(accepted_count / input_count) if input_count else 0.0,
        "original_coverage_min": (
            float(np.min(original_coverage_ratios)) if original_coverage_ratios else None
        ),
        "original_coverage_mean": (
            float(np.mean(original_coverage_ratios)) if original_coverage_ratios else None
        ),
        "original_coverage_max": (
            float(np.max(original_coverage_ratios)) if original_coverage_ratios else None
        ),
        "finalized_coverage_min": (
            float(np.min(finalized_coverage_ratios)) if finalized_coverage_ratios else None
        ),
        "finalized_coverage_mean": (
            float(np.mean(finalized_coverage_ratios)) if finalized_coverage_ratios else None
        ),
        "finalized_coverage_max": (
            float(np.max(finalized_coverage_ratios)) if finalized_coverage_ratios else None
        ),
        "projection_linf_mean": (
            float(np.mean(projection_distances)) if projection_distances else None
        ),
        "projection_linf_max": (
            float(np.max(projection_distances)) if projection_distances else None
        ),
        "coverage_repair_linf_mean_among_repaired": (
            float(np.mean(coverage_repair_distances)) if coverage_repair_distances else None
        ),
        "coverage_repair_linf_max": (
            float(np.max(coverage_repair_distances)) if coverage_repair_distances else None
        ),
        "width_absolute_change_mean_among_repaired": (
            float(np.mean(repaired_width_changes)) if repaired_width_changes else None
        ),
        "width_absolute_change_max": (
            float(np.max(repaired_width_changes)) if repaired_width_changes else None
        ),
        "width_relative_change_mean_among_repaired": (
            float(np.mean(repaired_width_relative_changes))
            if repaired_width_relative_changes
            else None
        ),
        "width_relative_change_max": (
            float(np.max(repaired_width_relative_changes))
            if repaired_width_relative_changes
            else None
        ),
    }
    if not designs:
        finalization_stats["ranking_stability"] = _ranking_stability(
            np.empty(0), np.empty(0)
        )
        return TopoNaderFinalizedPool(
            designs=np.empty((0, *geometry.design_shape), dtype=np.float64),
            normalized=np.empty((0, geometry.input_dim), dtype=np.float64),
            predicted_mean=np.empty(0, dtype=np.float64),
            predicted_uncertainty=np.empty(0, dtype=np.float64),
            coverage_ratios=np.empty(0, dtype=np.float64),
            rejection_counts=rejection_counts,
            source_indices=np.empty(0, dtype=np.int64),
            finalization_stats=finalization_stats,
        )
    design_array = np.stack(designs)
    normalized = geometry.normalize(design_array).reshape(-1, geometry.input_dim)
    mean, uncertainty = predictor(normalized)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if mean.size != design_array.shape[0] or uncertainty.size != design_array.shape[0]:
        raise ValueError("prediction count does not match finalized candidate pool")
    if ranking_mode is None:
        finalization_stats["ranking_stability"] = None
    else:
        if ranking_mode not in {"mean", "ucb"}:
            raise ValueError("ranking_mode must be mean, ucb or None")
        accepted_sources = np.asarray(source_indices, dtype=np.int64)
        if supplied_raw_priority is not None:
            accepted_raw_priority = supplied_raw_priority[accepted_sources]
            raw_priority_source = "pso_score"
        else:
            raw_mean, raw_uncertainty = predictor(pool[accepted_sources])
            raw_mean = np.asarray(raw_mean, dtype=np.float64).reshape(-1)
            raw_uncertainty = np.asarray(raw_uncertainty, dtype=np.float64).reshape(-1)
            accepted_raw_priority = (
                raw_mean + float(kappa) * raw_uncertainty
                if ranking_mode == "ucb"
                else raw_mean
            )
            raw_priority_source = "recomputed_raw_structure"
        final_priority = (
            mean + float(kappa) * uncertainty if ranking_mode == "ucb" else mean
        )
        ranking_stability = _ranking_stability(
            accepted_raw_priority, final_priority
        )
        ranking_stability["raw_priority_source"] = raw_priority_source
        ranking_stability["final_priority_mode"] = ranking_mode
        finalization_stats["ranking_stability"] = ranking_stability
    return TopoNaderFinalizedPool(
        designs=design_array,
        normalized=normalized,
        predicted_mean=mean,
        predicted_uncertainty=uncertainty,
        coverage_ratios=np.asarray(coverage, dtype=np.float64),
        rejection_counts=rejection_counts,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        finalization_stats=finalization_stats,
    )


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
