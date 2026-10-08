from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TopoNaderArchitectureSpec:
    """Bounded, serializable topology-surrogate architecture description."""

    name: str = "TopoNaderBaseline"
    input_dim: int = 16
    encoder_type: str = "mlp"
    hidden_widths: tuple[int, ...] = (1024, 512, 256)
    latent_dim: int = 12
    activation: str = "relu"
    normalization: str = "layer_norm"
    dropout: float = 0.10
    predictor_type: str = "transformer"
    predictor_depth: int = 2
    predictor_width: int = 128
    predictor_activation: str = "leaky_relu"
    attention_heads: int = 4
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def architecture_id(self) -> str:
        body = json.dumps(self.to_dict(include_metadata=False), sort_keys=True, separators=(",", ":"))
        return "toponader-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_widths"] = list(self.hidden_widths)
        if not include_metadata:
            result.pop("metadata", None)
            result.pop("name", None)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopoNaderArchitectureSpec":
        data = dict(payload)
        if "hidden_widths" in data:
            data["hidden_widths"] = tuple(int(v) for v in data["hidden_widths"])
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "TopoNaderArchitectureSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
