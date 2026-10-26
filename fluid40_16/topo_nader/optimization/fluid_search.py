from __future__ import annotations

import numpy as np

from .acquisition import combine_batches, finalize_candidate_pool, select_from_finalized_pool
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


def search_batch(config, geometry, dataset, iteration, predictor, predict_mean, predict_ucb):
    """Separate elite mean searches and global UCB search, bounded feasible refills."""
    o = config.optimization
    elite_cfg = o.elite_local_search
    best = dataset.inputs[np.argsort(-dataset.outputs)[:elite_cfg.elite_count]]
    historical = dataset.inputs.copy()
    batches, traces = [], []
    total_rejections = {'connectivity': 0, 'historical_duplicate': 0, 'batch_duplicate': 0}

    def allocate(total, weights):
        w = np.asarray(weights, float); raw = w / w.sum() * total
        n = np.floor(raw).astype(int)
        for i in np.argsort(-(raw-n), kind='stable')[:total-int(n.sum())]: n[i] += 1
        return n

    mean_count = o.exploitation_count if config.method.uncertainty_sampling else o.fe_batch_size
    if elite_cfg.enabled:
        quotas = allocate(mean_count, elite_cfg.selected_weights[:len(best)])
        particles = allocate(o.particles, elite_cfg.particle_weights[:len(best)])
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
        initial = sample_designs(rng, particles, centers, elite_cfg, local)
        # Preserve one elite to seed each swarm; historical designs are removed before FE.
        initial[0] = centers[0].reshape(-1)
        result = TopoNaderPSO(particles, o.iterations, o.mutation_rate, o.stall_limit,
                             config.seed_for(f'fluid_pso:{iteration}:{channel}')).optimize(
                                 objective, dimensions=640, initial_positions=initial,
                                 archive_size=max(o.candidate_pool_size*10, quota*50))
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
                       'local_initialization': local,'refill_attempts': attempt,
                       'local_hamming_limit': elite_cfg.local_hamming_limit if local else None,
                       'candidate_pool_statistics': stats})
    combined = combine_batches(*batches)
    combined.rejection_counts = total_rejections
    if len(combined.designs) != o.fe_batch_size:
        raise RuntimeError('fluid search did not produce the configured FE batch size')
    return combined, {'connectivity_mode': 'legacy_unchanged', 'channels': traces,
                      'eps慰_update': 'inherited continuous EPSO; binary evaluation',
                      'rejection_counts': total_rejections}
