from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from ..optimization.acquisition import TopoNaderBatchSelection


def save_fe_request(path: str | Path, batch: TopoNaderBatchSelection, extra: dict | None = None) -> None:
    payload = {
        "inputs": np.asarray(batch.designs, dtype=np.float64),
        "predicted_mean": np.asarray(batch.predicted_mean, dtype=np.float64).reshape(-1, 1),
        "predicted_uncertainty": np.asarray(batch.predicted_uncertainty, dtype=np.float64).reshape(-1, 1),
        "acquisition": np.asarray(batch.acquisition, dtype=np.float64).reshape(-1, 1),
        "coverage_ratios": np.asarray(batch.coverage_ratios, dtype=np.float64).reshape(-1, 1),
        "source": np.asarray(batch.sources, dtype=object).reshape(-1, 1),
    }
    if extra:
        payload.update(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    savemat(target, payload)


def load_fe_result(path: str | Path) -> np.ndarray:
    payload = loadmat(path)
    for key in ("outputs", "fe_outputs", "results"):
        if key in payload:
            return np.asarray(payload[key], dtype=np.float64).reshape(-1)
    raise KeyError("finite-element result MAT must contain outputs, fe_outputs, or results")

