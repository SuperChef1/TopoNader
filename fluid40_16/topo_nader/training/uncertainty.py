from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data.dataset import TopoNaderTargetTransform


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(module, nn.Dropout):
        module.train()


class TopoNaderMCDropoutPredictor:
    def __init__(self, model: nn.Module, transform: TopoNaderTargetTransform, samples: int = 20, seed: int = 3):
        if samples < 2:
            raise ValueError("MC dropout requires at least two samples")
        self.model = model
        self.transform = transform
        self.samples = samples
        self.seed = int(seed)
        self.device = next(model.parameters()).device

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_dim = int(self.model.spec.input_dim)
        x = torch.as_tensor(
            np.asarray(inputs).reshape(-1, input_dim), dtype=torch.float32, device=self.device
        )
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model.eval()
        self.model.apply(_enable_dropout)
        draws = []
        with torch.no_grad():
            for _ in range(self.samples):
                prediction = self.model(x).detach().cpu().numpy().reshape(-1)
                draws.append(self.transform.inverse(prediction))
        self.model.eval()
        stacked = np.stack(draws, axis=0)
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


class TopoNaderDeterministicPredictor:
    def __init__(self, model: nn.Module, transform: TopoNaderTargetTransform):
        self.model = model
        self.transform = transform
        self.device = next(model.parameters()).device

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_dim = int(self.model.spec.input_dim)
        x = torch.as_tensor(
            np.asarray(inputs).reshape(-1, input_dim), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x).detach().cpu().numpy().reshape(-1)
        mean = self.transform.inverse(prediction)
        return mean, np.zeros_like(mean, dtype=np.float64)
