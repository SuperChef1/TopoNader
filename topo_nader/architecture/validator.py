from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..config import TopoNaderArchitectureConfig
from .factory import TopoNaderModelFactory
from .spec import TopoNaderArchitectureSpec


@dataclass
class TopoNaderValidationReport:
    valid: bool
    architecture_id: str
    parameter_count: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class TopoNaderArchitectureValidator:
    def __init__(self, config: TopoNaderArchitectureConfig):
        self.config = config

    def validate(self, spec: TopoNaderArchitectureSpec) -> TopoNaderValidationReport:
        errors: list[str] = []
        if spec.input_dim != 16:
            errors.append("input_dim must be 16")
        if not spec.hidden_widths or len(spec.hidden_widths) > self.config.max_depth:
            errors.append("hidden_widths depth is outside the configured range")
        if any(w < self.config.min_hidden_width or w > self.config.max_hidden_width for w in spec.hidden_widths):
            errors.append("a hidden width is outside the configured range")
        if not 0 <= spec.dropout < 0.8:
            errors.append("dropout must be in [0, 0.8)")
        if spec.predictor_type == "transformer" and spec.predictor_width % spec.attention_heads:
            errors.append("predictor_width must be divisible by attention_heads")
        count = 0
        if not errors:
            try:
                model = TopoNaderModelFactory.build(spec)
                count = sum(p.numel() for p in model.parameters())
                output = model(torch.zeros(2, spec.input_dim))
                if output.shape != (2, 1) or not torch.isfinite(output).all():
                    errors.append(f"invalid smoke-test output: {tuple(output.shape)}")
                output.sum().backward()
            except Exception as exc:
                errors.append(f"model build/smoke test failed: {exc}")
        if count > self.config.max_parameters:
            errors.append(f"parameter count {count} exceeds {self.config.max_parameters}")
        return TopoNaderValidationReport(not errors, spec.architecture_id, count, tuple(errors))

