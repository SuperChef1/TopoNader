"""Seeded local architecture mutations for the no-LLM ablation.

One of eight component groups is sampled uniformly per attempt. This is a
documented random-mutation baseline, not a uniform distribution over all legal
architectures and not a claim to match an LLM's candidate distribution.
"""
from dataclasses import replace
import numpy as np

from ..architecture.spec import TopoNaderArchitectureSpec
from ..config import TopoNaderArchitectureConfig
from .change_scope import POLICY_VERSION, check_change_scope


OPERATORS = ('width', 'depth', 'encoder', 'latent', 'activation', 'normalization',
             'dropout', 'predictor')


def generate_random_candidates(current: TopoNaderArchitectureSpec,
                               config: TopoNaderArchitectureConfig, seed: int,
                               count: int, validator, retry_limit: int = 3):
    rng = np.random.default_rng(seed)
    candidates, attempts, proposals = [], [], []
    seen = {current.architecture_id}
    maximum_attempts = count * max(1, retry_limit) ** 2
    before = current.to_dict()
    for index in range(maximum_attempts):
        if len(candidates) >= count:
            break
        operator = str(rng.choice(OPERATORS))
        changes = {}
        if operator == 'width':
            scale = float(rng.choice([.5, .75, 1.25, 1.5]))
            changes['hidden_widths'] = [int(np.clip(round(w*scale/16)*16,
                config.min_hidden_width, config.max_hidden_width)) for w in current.hidden_widths]
        elif operator == 'depth':
            widths = list(current.hidden_widths)
            if len(widths) > 1 and (len(widths) >= config.max_depth or rng.random() < .5):
                widths.pop()
            elif len(widths) < config.max_depth:
                widths.append(int(np.clip(widths[-1]//2, config.min_hidden_width, config.max_hidden_width)))
            changes['hidden_widths'] = widths
        elif operator == 'encoder':
            changes['encoder_type'] = 'residual_mlp' if current.encoder_type == 'mlp' else 'mlp'
        elif operator == 'latent':
            scale = float(rng.choice([.5, .75, 1.25, 1.5, 2.]))
            changes['latent_dim'] = int(np.clip(round(current.latent_dim*scale),
                                               config.min_latent_dim, config.max_latent_dim))
        elif operator == 'activation':
            changes['activation'] = str(rng.choice([a for a in ['relu','gelu','leaky_relu','silu']
                                                   if a != current.activation]))
        elif operator == 'normalization':
            changes['normalization'] = str(rng.choice([a for a in ['layer_norm','batch_norm','none']
                                                      if a != current.normalization]))
        elif operator == 'dropout':
            changes['dropout'] = float(rng.choice([0., .05, .1, .2, .3, .5]))
        else:
            kind = str(rng.choice(['mlp','transformer']))
            width = int(rng.choice([16,32,64,128,256]))
            changes.update(predictor_type=kind, predictor_width=width)
            if kind == 'transformer':
                changes['predictor_depth'] = int(rng.choice([1,2,3,4]))
                changes['attention_heads'] = int(rng.choice([h for h in [1,2,4,8] if width % h == 0]))
            else:
                changes['predictor_activation'] = str(rng.choice(['relu','gelu','leaky_relu','silu']))
        changes = {k:v for k,v in changes.items() if before[k] != v}
        proposal = {'title': f'Random {operator} mutation',
                    'design_intent': f'Explore the {operator} component without performance-guided proposal selection',
                    'rationale': 'Seeded random-mutation ablation; no LLM or experience retrieval',
                    'changes': changes}
        proposals.append(proposal)
        payload = {**before, **changes}
        payload['metadata'] = {'proposal': proposal, 'supporting_changes': {},
                               'proposal_source': 'random', 'operator': operator}
        child = TopoNaderArchitectureSpec.from_dict(payload)
        child = replace(child, name=f'Random_{operator}_{child.architecture_id}')
        scope = check_change_scope(current, child, proposal)
        errors = list(scope['errors'])
        if child.architecture_id in seen:
            errors.append('duplicate_or_parent_architecture')
        report = None
        if not errors:
            report = validator(child)
            if not report.valid:
                errors.extend(report.errors)
        accepted = not errors
        if accepted:
            child.metadata['change_scope'] = scope
            candidates.append(child)
            seen.add(child.architecture_id)
        attempts.append({'proposal': proposal, 'attempt': index+1,
                         'architecture_id': child.architecture_id, 'change_scope': scope,
                         'validation': report.to_dict() if report else None,
                         'accepted': accepted, 'errors': errors})
    return candidates, {
        'proposal_source': 'random', 'random_policy_version': 'local_mutation_v1',
        'proposal_seed': int(seed), 'operators': list(OPERATORS),
        'operator_sampling': 'uniform per attempt; legal unique candidates accepted',
        'maximum_attempts': maximum_attempts, 'change_scope_policy': POLICY_VERSION,
        'diagnosis': {'skipped': 'random_ablation_no_llm'},
        'proposal_rounds': [{'proposals': proposals}], 'candidate_attempts': attempts,
        'requested_candidate_count': count, 'accepted_candidate_count': len(candidates),
        'failures': [a for a in attempts if not a['accepted']], 'usage': [],
    }
