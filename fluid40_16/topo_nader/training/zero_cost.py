from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..architecture.factory import TopoNaderModelFactory
from ..architecture.spec import TopoNaderArchitectureSpec


@dataclass
class TopoNaderZeroCostReport:
    architecture_id: str
    valid: bool
    score: float
    jacobian_rank: int
    jacobian_condition: float
    input_gradient_norm: float
    output_std: float
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class TopoNaderZeroCostEvaluator:
    """Training-free regression proxy based on input-output Jacobian diversity.

    The score is a screening heuristic, not evidence of final model quality. It
    performs forward/backward passes only and never updates model parameters.
    """

    def __init__(self, seed: int = 3, max_samples: int = 12):
        self.seed = int(seed)
        self.max_samples = int(max_samples)

    def evaluate(self, spec: TopoNaderArchitectureSpec, normalized_inputs: np.ndarray) -> TopoNaderZeroCostReport:
        errors: list[str] = []
        try:
            torch.manual_seed(self.seed)
            model = TopoNaderModelFactory.build(spec).cpu().eval()
            values = np.asarray(normalized_inputs, dtype=np.float32).reshape(-1, spec.input_dim)
            if values.shape[0] < 2:
                raise ValueError("zero-cost screening requires at least two inputs")
            if values.shape[0] > self.max_samples:
                indices = np.linspace(0, values.shape[0] - 1, self.max_samples, dtype=int)
                values = values[indices]
            x = torch.tensor(values, dtype=torch.float32, requires_grad=True)
            output = model(x).reshape(-1)
            rows = []
            for index in range(output.shape[0]):
                gradient = torch.autograd.grad(output[index], x, retain_graph=True)[0][index]
                rows.append(gradient.detach().cpu().numpy())
            jacobian = np.stack(rows).astype(np.float64)
            if not np.isfinite(jacobian).all():
                raise ValueError("non-finite input-output Jacobian")
            singular = np.linalg.svd(jacobian, compute_uv=False)
            tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * max(float(singular[0]), 1e-12)
            rank = int(np.count_nonzero(singular > tolerance))
            positive = singular[singular > tolerance]
            condition = float(positive[0] / positive[-1]) if positive.size >= 2 else 1e12
            gradient_norm = float(np.linalg.norm(jacobian) / math.sqrt(jacobian.size))
            output_std = float(output.detach().std(unbiased=False).cpu())
            if rank < 2:
                errors.append("degenerate_jacobian_rank")
            if gradient_norm <= 1e-12:
                errors.append("vanishing_input_gradient")
            if output_std <= 1e-12:
                errors.append("collapsed_output")
            score = (
                math.log1p(rank)
                + 0.25 * math.log1p(max(gradient_norm, 0.0))
                + 0.10 * math.log1p(max(output_std, 0.0))
                - 0.10 * math.log1p(max(condition, 1.0))
            )
            return TopoNaderZeroCostReport(
                spec.architecture_id,
                not errors,
                float(score),
                rank,
                condition,
                gradient_norm,
                output_std,
                tuple(errors),
            )
        except Exception as exc:
            return TopoNaderZeroCostReport(
                spec.architecture_id,
                False,
                -1e30,
                0,
                1e30,
                0.0,
                0.0,
                (str(exc),),
            )
