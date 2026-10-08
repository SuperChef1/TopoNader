from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..monitoring.trigger import TopoNaderTriggerState


@dataclass
class TopoNaderRunState:
    current_architecture_file: str = "state/current_architecture.json"
    current_architecture_id: str | None = None
    next_iteration: int = 1
    audit_due: bool = False
    trigger: TopoNaderTriggerState = field(default_factory=TopoNaderTriggerState)
    last_metrics_file: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TopoNaderRunState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["trigger"] = TopoNaderTriggerState(**payload.get("trigger", {}))
        return cls(**payload)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["trigger"] = asdict(self.trigger)
        return result

