from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..architecture.factory import TopoNaderModelFactory, TopoNaderSurrogate
from ..architecture.spec import TopoNaderArchitectureSpec
from ..config import TopoNaderTrainingConfig
from ..data.dataset import TopoNaderSplit, TopoNaderTargetTransform
from .evaluator import TopoNaderEvaluation, evaluate_predictions


@dataclass
class TopoNaderTrainingResult:
    architecture_id: str
    best_epoch: int
    best_validation_loss: float
    validation: TopoNaderEvaluation
    checkpoint_path: str | None
    training_history: list[dict[str, float]]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["validation"] = self.validation.to_dict()
        return result


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class TopoNaderTrainer:
    def __init__(self, config: TopoNaderTrainingConfig):
        self.config = config
        self.device = _device(config.device)

    def train(
        self,
        spec: TopoNaderArchitectureSpec,
        split: TopoNaderSplit,
        checkpoint_path: str | Path | None = None,
        epochs: int | None = None,
        output_log10: bool = True,
    ) -> tuple[TopoNaderSurrogate, TopoNaderTargetTransform, TopoNaderTrainingResult]:
        _set_seed(self.config.seed)
        model = TopoNaderModelFactory.build(spec)
        model.to(self.device)
        transform = TopoNaderTargetTransform.fit(split.y_train, log10=output_log10)
        train_x = torch.as_tensor(split.x_train, dtype=torch.float32)
        train_y = torch.as_tensor(transform.transform(split.y_train).reshape(-1, 1), dtype=torch.float32)
        validation_x = torch.as_tensor(split.x_validation, dtype=torch.float32, device=self.device)
        validation_y = torch.as_tensor(
            transform.transform(split.y_validation).reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        drop_last = train_x.shape[0] % self.config.batch_size == 1
        loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=min(self.config.batch_size, train_x.shape[0]),
            shuffle=True,
            drop_last=drop_last,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        loss_function = nn.MSELoss()
        best_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = self.config.patience
        history: list[dict[str, float]] = []
        for epoch in range(1, int(epochs or self.config.epochs) + 1):
            model.train()
            weighted_loss = 0.0
            seen = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.detach()) * batch_x.shape[0]
                seen += batch_x.shape[0]
            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(model(validation_x), validation_y))
            history.append({"epoch": float(epoch), "train_loss": weighted_loss / max(seen, 1), "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-10:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = self.config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_z = model(validation_x).cpu().numpy().reshape(-1)
        pred_physical = transform.inverse(pred_z)
        evaluation = evaluate_predictions(split.y_validation, pred_physical, topk=min(3, split.y_validation.size))
        saved_path = None
        if checkpoint_path is not None:
            target = Path(checkpoint_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": spec.to_dict(),
                    "state_dict": best_state,
                    "target_transform": asdict(transform),
                    "validation": evaluation.to_dict(),
                },
                target,
            )
            saved_path = str(target)
            target.with_suffix(".metrics.json").write_text(
                json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        result = TopoNaderTrainingResult(
            spec.architecture_id, best_epoch, best_loss, evaluation, saved_path, history
        )
        return model, transform, result


def load_toponader_checkpoint(path: str | Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device)
    spec = TopoNaderArchitectureSpec.from_dict(payload["architecture"])
    model = TopoNaderModelFactory.build(spec)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    transform = TopoNaderTargetTransform(**payload["target_transform"])
    return model, transform, spec
