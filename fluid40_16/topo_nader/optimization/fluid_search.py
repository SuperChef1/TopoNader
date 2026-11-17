from __future__ import annotations

import numpy as np

from .acquisition import combine_batches, finalize_candidate_pool, select_from_finalized_pool
from .bba import TopoNaderBBA
from .pso import TopoNaderPSO


def sample_designs(rng, count, elites, settings, local=False):
    """Initialization/refill only: no channel carving or connectivity repair."""
    bases = np.asarray(elites).reshape(-1, 640)
    out = np.empty((count, 640))
    for i in range(count):
        if local or rng.random() < 0.65:
            x = bases[rng.integers(len(bases))].copy()
            if rng.random() < settings.block_mutation_fraction:
                a = x.reshape(40, 16)
                w = int(rng.integers(1, settings.block_max_side + 1))
                h = int(rng.integers(1, settings.block_max_side + 1))
                ix, iy = int(rng.integers(41-w)), int(rng.integers(17-h))
                a[ix:ix+w, iy:iy+h] = 1 - a[ix:ix+w, iy:iy+h]
            else:
                n = int(rng.integers(settings.min_flips, settings.max_flips + 1))
                indices = rng.choice(640, n, replace=False)
                x[indices] = 1 - x[indices]
        else:
            density = rng.choice(settings.sampling_densities)
            x = (rng.random(640) < density).astype(float)
        out[i] = x
    return out


def _allocate_counts(total, weights):
    weights = np.asarray(weights, dtype=float)
    raw = weights / weights.sum() * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind='stable')
        counts[order[:remainder]] += 1
    return counts


def sample_bba_stratified_designs(rng, count, elites, settings, local=False):
    """Build an exactly unique binary BBA population around several elite radii.

    This changes only BBA initialization. The optimizer update and the legacy
    refill sampler remain unchanged so the first ablation has one main factor.
    """
    bases = (np.asarray(elites).reshape(-1, 640) >= .5).astype(np.uint8)
    population, labels, seen = [], [], set()

    def append(candidate, label):
        candidate = np.asarray(candidate, dtype=np.uint8).reshape(640)
        key = np.packbits(candidate).tobytes()
        if key in seen:
            return False
        seen.add(key)
        population.append(candidate)
        labels.append(label)
        return True

    def flip_candidate(low, high):
        x = bases[int(rng.integers(len(bases)))].copy()
        flips = int(rng.integers(low, high + 1))
        indices = rng.choice(640, flips, replace=False)
        x[indices] = 1 - x[indices]
        return x

    def block_candidate():
        x = bases[int(rng.integers(len(bases)))].copy().reshape(40, 16)
        max_side = min(2, settings.block_max_side)
        width = int(rng.integers(1, max_side + 1))
        height = int(rng.integers(1, max_side + 1))
        ix = int(rng.integers(41 - width))
        iy = int(rng.integers(17 - height))
        x[ix:ix + width, iy:iy + height] = 1 - x[ix:ix + width, iy:iy + height]
        return x.reshape(-1)

    # Local channels receive their own elite. The global channel receives one
    # anchor per available elite before stratified mutations are allocated.
    anchor_limit = 1 if local else min(len(bases), count)
    for base in bases[:anchor_limit]:
        if len(population) == count:
            break
        append(base, 'anchor')

    remaining = count - len(population)
    categories = (
        [('near_flip', .55, lambda: flip_candidate(1, 4)),
         ('medium_flip', .25, lambda: flip_candidate(5, 8)),
         ('block', .20, block_candidate)]
        if local else
        [('elite_mutation', .55, lambda: flip_candidate(2, 12)),
         ('block', .25, block_candidate),
         ('density', .20, lambda: (rng.random(640) <
                                    float(rng.choice(settings.sampling_densities))).astype(np.uint8))]
    )
    quotas = _allocate_counts(remaining, [item[1] for item in categories])
    for (label, _, generator), quota in zip(categories, quotas):
        target = len(population) + int(quota)
        attempts = 0
        while len(population) < target and attempts < max(1000, count * 100):
            append(generator(), label)
            attempts += 1

    # Extremely sparse density draws can collide (especially in smoke tests).
    # Complete the population with bounded elite mutations without weakening
    # exact uniqueness or the local Hamming constraint.
    attempts = 0
    fallback_high = min(8 if local else 32, settings.local_hamming_limit)
    while len(population) < count and attempts < max(5000, count * 500):
        append(flip_candidate(1, fallback_high), 'fallback_flip')
        attempts += 1
    if len(population) != count:
        raise RuntimeError(f'could not construct {count} unique BBA initial designs')

    result = np.asarray(population, dtype=float)
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    statistics = {
        'mode': 'stratified',
        'unique_count': len({np.packbits(row.astype(np.uint8)).tobytes() for row in result}),
        'category_counts': counts,
        'solid_fraction_mean': float(result.mean()),
    }
    return result, statistics


def search_batch(config, geometry, dataset, iteration, predictor, predict_mean, predict_ucb):
    """Separate elite mean searches and global UCB search, bounded feasible refills."""
    o = config.optimization
    elite_cfg = o.elite_local_search
    best = dataset.inputs[np.argsort(-dataset.outputs)[:elite_cfg.elite_count]]
    historical = dataset.inputs.copy()
    batches, traces = [], []
    total_rejections = {'connectivity': 0, 'historical_duplicate': 0, 'batch_duplicate': 0}

    mean_count = o.exploitation_count if config.method.uncertainty_sampling else o.fe_batch_size
    if elite_cfg.enabled:
        quotas = _allocate_counts(mean_count, elite_cfg.selected_weights[:len(best)])
        particles = _allocate_counts(o.particles, elite_cfg.particle_weights[:len(best)])
        channels = [('exploitation', int(q), int(p), best[j:j+1], True)
                    for j,(q,p) in enumerate(zip(quotas,particles)) if q]
    else:
        channels = [('exploitation', mean_count, o.particles, best, False)]
    if config.method.uncertainty_sampling and o.exploration_count:
        channels.append(('exploration', o.exploration_count, o.particles, best, False))

    for channel, (source, quota, particles, centers, local) in enumerate(channels):
        rng = np.random.default_rng(config.seed_for(f'fluid_search:{iteration}:{channel}'))
        base_objective = predict_mean if source == 'exploitation' else predict_ucb
        def objective(x):
            scores = base_objective(x)
            if local:
                distance = np.count_nonzero((x >= .5) != centers[0].reshape(-1), axis=1)
                scores = np.where(distance <= elite_cfg.local_hamming_limit, scores, -1e30)
            return scores
        if o.optimizer == 'bba' and o.bba.initialization_mode == 'stratified':
            initial, initialization_statistics = sample_bba_stratified_designs(
                rng, particles, centers, elite_cfg, local)
        else:
            initial = sample_designs(rng, particles, centers, elite_cfg, local)
            # Preserve one elite to seed each swarm; historical designs are removed before FE.
            initial[0] = centers[0].reshape(-1)
            initialization_statistics = {
                'mode': 'legacy',
                'unique_count': len({np.packbits((row >= .5).astype(np.uint8)).tobytes()
                                     for row in initial}),
                'solid_fraction_mean': float((initial >= .5).mean()),
            }
        archive_size = max(o.candidate_pool_size * 10, quota * 50)
        if o.optimizer == 'bba':
            bba = o.bba
            optimizer = TopoNaderBBA(
                particles=particles,
                iterations=o.iterations,
                frequency_min=bba.frequency_min,
                frequency_max=bba.frequency_max,
                loudness=bba.loudness,
                pulse_rate=bba.pulse_rate,
                loudness_decay=bba.loudness_decay,
                pulse_decay=bba.pulse_decay,
                seed=config.seed_for(f'fluid_bba:{iteration}:{channel}'),
            )
        else:
            optimizer = TopoNaderPSO(
                particles,
                o.iterations,
                o.mutation_rate,
                o.stall_limit,
                config.seed_for(f'fluid_pso:{iteration}:{channel}'),
            )
        result = optimizer.optimize(
            objective,
            dimensions=640,
            initial_positions=initial,
            archive_size=archive_size,
        )
        raw = np.vstack([initial, result.trajectory_positions])
        selected = np.empty((0,40,16)); stats = []
        for attempt in range(o.refill_attempts + 1):
            if local:
                distance = np.count_nonzero((raw >= .5) != centers[0].reshape(-1), axis=1)
                raw = raw[distance <= elite_cfg.local_hamming_limit]
            pool = finalize_candidate_pool(raw, geometry, predictor,
                excluded_designs=np.concatenate([historical, selected]),
                ranking_mode='ucb' if source == 'exploration' else 'mean',
                kappa=config.uncertainty.kappa)
            stats.append(pool.finalization_stats)
            for k,v in pool.rejection_counts.items():
                total_rejections[k] = total_rejections.get(k,0)+v
            batch = select_from_finalized_pool(pool, quota-len(selected), config.uncertainty.kappa,
                source, source == 'exploration', config.method.diversity_sampling)
            if len(batch.designs):
                batch.rejection_counts = None
                batches.append(batch)
                selected = np.concatenate([selected,batch.designs])
            if len(selected) == quota: break
            raw = sample_designs(rng, o.refill_pool_size, centers, elite_cfg, local)
        if len(selected) != quota:
            raise RuntimeError(f'{source} channel {channel}: only {len(selected)}/{quota} unique '
                               'feasible designs after bounded refills; FE was not called')
        historical = np.concatenate([historical,selected])
        traces.append({'source': source,'quota': quota,'particles': particles,
                       'optimizer': o.optimizer,
                       'initialization_statistics': initialization_statistics,
                       'local_initialization': local,'refill_attempts': attempt,
                       'local_hamming_limit': elite_cfg.local_hamming_limit if local else None,
                       'candidate_pool_statistics': stats})
    combined = combine_batches(*batches)
    combined.rejection_counts = total_rejections
    if len(combined.designs) != o.fe_batch_size:
        raise RuntimeError('fluid search did not produce the configured FE batch size')
    optimizer_description = (
        'binary bat algorithm; binary state and corrected dimension-scaled flip floor'
        if o.optimizer == 'bba'
        else 'inherited continuous EPSO; binary evaluation'
    )
    return combined, {'connectivity_mode': 'legacy_unchanged',
                      'optimizer': o.optimizer,
                      'optimizer_update': optimizer_description,
                      'channels': traces,
                      'rejection_counts': total_rejections}
