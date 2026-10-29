"""Frozen expert suggestions; program chooses exact targets, LLM implements only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..architecture.spec import TopoNaderArchitectureSpec
from ..config import TopoNaderArchitectureConfig
from .change_scope import check_change_scope

LIBRARY_PATH = Path(__file__).with_name('human_suggestions_v1.json')


def load_human_library(expected_sha256: str):
    raw = LIBRARY_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('human suggestion library SHA256 mismatch; use the original library or a new experiment')
    library = json.loads(raw)
    rules = library['rules']
    if library['policy_version'] != 'surrogate_human_v1' or len({r['id'] for r in rules}) != len(rules):
        raise ValueError('invalid human suggestion library version or duplicate IDs')
    return library, digest


def _targets(rule: dict, current: TopoNaderArchitectureSpec, config: TopoNaderArchitectureConfig):
    operation = rule['operation']
    if operation == 'set':
        return dict(rule['targets'])
    if operation == 'scale_widths':
        return {'hidden_widths': [int(np.clip(round(w * rule['factor'] / 16) * 16,
            config.min_hidden_width, config.max_hidden_width)) for w in current.hidden_widths]}
    if operation == 'depth_down':
        return {'hidden_widths': list(current.hidden_widths[:-1])} if len(current.hidden_widths) > 1 else {}
    if operation == 'depth_up':
        if len(current.hidden_widths) >= config.max_depth:
            return {}
        width = int(np.clip(current.hidden_widths[-1] // 2, config.min_hidden_width, config.max_hidden_width))
        return {'hidden_widths': [*current.hidden_widths, width]}
    if operation == 'scale_latent':
        return {'latent_dim': int(np.clip(round(current.latent_dim * rule['factor']),
            config.min_latent_dim, config.max_latent_dim))}
    if operation == 'predictor_transformer':
        targets = {'predictor_type': 'transformer'}
        # Inactive MLP fields can otherwise be incompatible when attention returns.
        if current.attention_heads < 1 or current.predictor_width % current.attention_heads:
            targets['attention_heads'] = max(h for h in [1, 2, 4, 8] if current.predictor_width % h == 0)
        if current.predictor_depth < 1:
            targets['predictor_depth'] = 1
        return targets
    raise ValueError(f'unknown human rule operation: {operation}')


def generate_human_proposals(current: TopoNaderArchitectureSpec, config: TopoNaderArchitectureConfig,
                             seed: int, expected_sha256: str, validator):
    library, digest = load_human_library(expected_sha256)
    before = current.to_dict()
    proposals, skipped = [], []
    seen = {current.architecture_id}
    order = np.random.default_rng(seed).permutation(len(library['rules']))
    for index in order:
        rule = library['rules'][int(index)]
        changes = {k:v for k,v in _targets(rule, current, config).items() if before[k] != v}
        if not changes:
            skipped.append({'id': rule['id'], 'reason': 'inapplicable_or_no_change'})
            continue
        proposal = {'title': rule['id'], 'design_intent': rule['intent'],
            'rationale': rule['adaptation'], 'changes': changes,
            'human_suggestion_id': rule['id'], 'library_sha256': digest,
            'source_urls': [library['sources'][s] for s in rule['sources']]}
        child = TopoNaderArchitectureSpec.from_dict({**before, **changes, 'metadata': {'supporting_changes': {}}})
        scope = check_change_scope(current, child, proposal)
        # Prechecking suggestions must not consume the training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            report = validator(child)
        if not scope['valid'] or not report.valid or child.architecture_id in seen:
            skipped.append({'id': rule['id'], 'reason': 'invalid_or_duplicate',
                            'errors': scope['errors'] + list(report.errors)})
            continue
        seen.add(child.architecture_id)
        proposals.append(proposal)
    return proposals, {'policy_version': library['policy_version'], 'library_sha256': digest,
        'proposal_seed': int(seed), 'selection': library['selection'],
        'rule_count': len(library['rules']), 'applicable_count': len(proposals),
        'ordered_suggestion_ids': [p['human_suggestion_id'] for p in proposals], 'skipped': skipped}
