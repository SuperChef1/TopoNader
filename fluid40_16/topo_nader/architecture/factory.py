from __future__ import annotations

import torch
from torch import nn

from .spec import TopoNaderArchitectureSpec


def _activation(name: str) -> nn.Module:
    choices = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "silu": nn.SiLU,
    }
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[name]()


def _normalization(name: str, width: int) -> nn.Module:
    if name == "layer_norm":
        return nn.LayerNorm(width)
    if name == "batch_norm":
        return nn.BatchNorm1d(width)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


class TopoNaderResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, spec: TopoNaderArchitectureSpec):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            _normalization(spec.normalization, out_features),
            _activation(spec.activation),
            nn.Dropout(spec.dropout),
        )
        self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class TopoNaderSurrogate(nn.Module):
    def __init__(self, spec: TopoNaderArchitectureSpec):
        super().__init__()
        layers: list[nn.Module] = []
        previous = spec.input_dim
        for width in spec.hidden_widths:
            if spec.encoder_type == "residual_mlp":
                layers.append(TopoNaderResidualBlock(previous, width, spec))
            elif spec.encoder_type == "mlp":
                layers.extend([
                    nn.Linear(previous, width),
                    _normalization(spec.normalization, width),
                    _activation(spec.activation),
                    nn.Dropout(spec.dropout),
                ])
            else:
                raise ValueError(f"Unsupported encoder_type: {spec.encoder_type}")
            previous = width
        layers.append(nn.Linear(previous, spec.latent_dim))
        self.encoder = nn.Sequential(*layers)
        if spec.predictor_type == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(spec.latent_dim, spec.predictor_width),
                _activation(spec.predictor_activation),
                nn.Dropout(spec.dropout),
                nn.Linear(spec.predictor_width, 1),
            )
        else:
            raise ValueError(f"Unsupported predictor_type: {spec.predictor_type}")
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encoder(x.reshape(-1, self.spec.input_dim)))


class TopoNaderModelFactory:
    @staticmethod
    def build(spec: TopoNaderArchitectureSpec) -> TopoNaderSurrogate:
        return TopoNaderSurrogate(spec)
